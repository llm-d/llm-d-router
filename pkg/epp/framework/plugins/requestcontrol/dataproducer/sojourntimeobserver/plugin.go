/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package sojourntimeobserver observes each request's sojourn split into TTFT
// (dispatchedAt to firstChunkAt) and decode (firstChunkAt to endOfStreamAt),
// maintains a paired t-digest per endpoint, and publishes a serialized
// snapshot for the mrl-scorer-hub.
//
// It spans two layers because neither can do the job alone: request-control
// hooks are the only place traffic is visible, while the attribute the scorer
// reads is a datalayer concept. The snapshot is published through a
// DynamicAttribute attached once per endpoint, matching latency-observer-producer-hub.
package sojourntimeobserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/caio/go-tdigest/v5"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrsojourn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/sojourntime"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	observerconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sojourntimeobserver/constants"
)

const (
	SojournTimeObserverProducerType = observerconstants.SojournTimeObserverProducerType
)

// Config holds the observer's parameters. Durations are time.ParseDuration
// strings.
type Config struct {
	// Compression controls the t-digest's centroid budget. Higher means more
	// centroids and finer tail quantiles at greater memory cost. Applied to
	// both the TTFT and decode digests.
	Compression float64 `json:"compression,omitempty"`

	// MinSamples is the warm-up threshold applied to each digest independently.
	// A snapshot is published only when both digests have at least this many
	// samples; before that, the endpoint has no snapshot entry in its
	// AttributeMap and the scorer treats it as cold.
	MinSamples uint64 `json:"minSamples,omitempty"`

	// IntervalDuration is the cadence at which the datalayer calls Dispatch to
	// publish the per-endpoint snapshot.
	IntervalDuration string `json:"intervalDuration,omitempty"`
}

// DefaultConfig is decoded over by the factory.
var DefaultConfig = Config{
	Compression:      200,
	MinSamples:       60,
	IntervalDuration: "5s",
}

// resolvedConfig is Config after parsing and validation.
type resolvedConfig struct {
	compression float64
	minSamples  uint64
	interval    time.Duration
}

func (c Config) resolve() (resolvedConfig, error) {
	var out resolvedConfig
	if c.Compression < 1 {
		return out, fmt.Errorf("compression must be >= 1, got %v", c.Compression)
	}
	if c.MinSamples == 0 {
		return out, errors.New("minSamples must be > 0")
	}
	d, err := time.ParseDuration(c.IntervalDuration)
	if err != nil {
		return out, fmt.Errorf("invalid intervalDuration %q: %w", c.IntervalDuration, err)
	}
	if d <= 0 {
		return out, fmt.Errorf("intervalDuration must be > 0, got %v", d)
	}
	out.compression = c.Compression
	out.minSamples = c.MinSamples
	out.interval = d
	return out, nil
}

var (
	_ fwkplugin.ProducerPlugin = &Observer{}
	_ fwkdl.EndpointExtractor  = (*Observer)(nil)
	_ fwkdl.Registrant         = &Observer{}
	_ fwkdl.PollingDispatcher  = &Observer{}
)

// Observer records per-endpoint TTFT and decode latency samples into per-endpoint
// t-digests and publishes two per-endpoint attributes: a serialized digest
// snapshot under SojournEstimatorSnapshotDataKey, and the endpoint's in-flight
// request list under InFlightRequestsDataKey. Both are refreshed on the flush
// tick and read lock-free through DynamicAttribute closures.
//
// Process-lifetime singleton serving every request, so all shared state is
// guarded.
type Observer struct {
	typedName fwkplugin.TypedName
	cfg       resolvedConfig

	snapshotDataKey         fwkplugin.DataKey // paired-digest snapshot
	inFlightRequestsDataKey fwkplugin.DataKey // per-endpoint in-flight list

	// mu guards the three maps below. Per-endpoint digest mutation takes
	// endpointState.mu; the two locks are never held simultaneously in the
	// same direction, so lock order is: mu (for state, inflight, or
	// requestToEndpoint) then endpointState.mu.
	mu sync.RWMutex

	// state carries the paired digests plus the two published attribute
	// pointers per endpoint.
	state map[string]*endpointState

	// inflight is the fleet-wide in-flight index: for each endpoint, one
	// entry per dispatched-not-completed request on that endpoint. Written
	// on PreRequest, first-chunk, and end-of-stream; snapshotted into
	// state.publishedInFlight on the flush tick.
	inflight map[string]map[string]*inflightEntry

	// requestToEndpoint maps a dispatched-not-completed request ID to the
	// endpoint it was dispatched on. Kept in lock-step with inflight so
	// noteFirstChunk and noteEndOfStream resolve the owning endpoint in
	// O(1) instead of scanning every endpoint's inner map. The invariant:
	// a requestID is present in exactly one inflight[*] inner map iff it
	// is present here with that endpoint as the value.
	requestToEndpoint map[string]string
}

// inflightEntry is one dispatched-not-completed request's timestamps.
// FirstChunkAt is the zero Time when no chunk has arrived yet.
type inflightEntry struct {
	dispatchedAt time.Time
	firstChunkAt time.Time
}

// endpointState carries one endpoint's TTFT and decode digests plus the two
// pointers the DynamicAttribute closures load.
type endpointState struct {
	mu     sync.Mutex
	ttft   *tdigest.TDigest
	decode *tdigest.TDigest

	// published is read lock-free by the SojournEstimatorSnapshotDataKey
	// closure. Set to non-nil only when both digests are warm.
	published atomic.Pointer[attrsojourn.SojournEstimatorSnapshot]

	// publishedInFlight is read lock-free by the InFlightRequestsDataKey
	// closure. Refreshed on the flush tick with a copy of the endpoint's
	// current in-flight index, so its freshness is bounded by
	// intervalDuration. Nil until the first flush.
	publishedInFlight atomic.Pointer[attrsojourn.InFlightRequestsSnapshot]
}

// SojournTimeObserverFactory builds an Observer. The recompute is driven by the
// datalayer once the observer is listed under dataLayer.sources.
func SojournTimeObserverFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}

	cfg := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for plugin %q: %w", name, err)
		}
	}

	observer, err := NewObserver(name, cfg)
	if err != nil {
		return nil, fmt.Errorf("invalid parameters for plugin %q: %w", name, err)
	}
	return observer, nil
}

// NewObserver initializes an Observer. See README.md "Fleet-wide in-flight
// index" for why the in-flight index is kept outside PluginState.
func NewObserver(name string, cfg Config) (*Observer, error) {
	resolved, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	return &Observer{
		typedName:               fwkplugin.TypedName{Type: SojournTimeObserverProducerType, Name: name},
		cfg:                     resolved,
		snapshotDataKey:         attrsojourn.SojournEstimatorSnapshotDataKey.WithNonEmptyProducerName(name),
		inFlightRequestsDataKey: attrsojourn.InFlightRequestsDataKey.WithNonEmptyProducerName(name),
		state:                   map[string]*endpointState{},
		inflight:                map[string]map[string]*inflightEntry{},
		requestToEndpoint:       map[string]string{},
	}, nil
}

// TypedName implements fwkplugin.Plugin.
func (p *Observer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the two attributes this producer publishes on each
// endpoint.
func (p *Observer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.snapshotDataKey:         attrsojourn.SojournEstimatorSnapshot{},
		p.inFlightRequestsDataKey: attrsojourn.InFlightRequestsSnapshot{},
	}
}

// RegisterDependencies subscribes to endpoint lifecycle events, so the observer
// can allocate per-endpoint state and attach its published attribute.
func (p *Observer) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      p.TypedName(),
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType,
		),
	})
}

// Extract handles endpoint lifecycle events. On add it attaches a
// DynamicAttribute whose closure resolves to the latest published snapshot,
// so the AttributeMap is written exactly once per endpoint and each flush
// only swaps the pointer behind it. Nil until the digests are warm.
func (p *Observer) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}
	id := event.Endpoint.GetMetadata().ID.String()
	logger := log.FromContext(ctx).V(logutil.DEFAULT)

	switch event.Type {
	case fwkdl.EventDelete:
		p.mu.Lock()
		delete(p.state, id)
		// Purge the reverse-index entries for every request that was on
		// this endpoint before removing the inner map, so the
		// requestToEndpoint invariant holds across the delete.
		for reqID := range p.inflight[id] {
			delete(p.requestToEndpoint, reqID)
		}
		delete(p.inflight, id)
		p.mu.Unlock()
		logger.Info("Dropped sojourn digests for deleted endpoint", "endpoint", id)

	case fwkdl.EventAddOrUpdate:
		state := p.stateForOrCreate(id)
		event.Endpoint.GetAttributes().Put(p.snapshotDataKey, &fwkdl.DynamicAttribute{
			// A nil typed-pointer boxed into fwkdl.Cloneable is not a nil
			// interface; the explicit return keeps a cold endpoint reading
			// as absent rather than as a non-nil interface wrapping nil.
			Get: func() fwkdl.Cloneable {
				snapshot := state.published.Load()
				if snapshot == nil {
					return nil
				}
				return snapshot
			},
		})
		event.Endpoint.GetAttributes().Put(p.inFlightRequestsDataKey, &fwkdl.DynamicAttribute{
			Get: func() fwkdl.Cloneable {
				snapshot := state.publishedInFlight.Load()
				if snapshot == nil {
					return nil
				}
				return snapshot
			},
		})
		logger.Info("Attached sojourn attributes",
			"snapshotKey", p.snapshotDataKey.String(),
			"inFlightKey", p.inFlightRequestsDataKey.String(),
			"endpoint", id)
	}
	return nil
}

// stateForOrCreate returns the endpoint's state, creating it if absent.
// Digest allocation errors are surfaced via the returned zero pointer; a
// caller that gets nil must skip observation for this endpoint rather than
// panic.
func (p *Observer) stateForOrCreate(endpointID string) *endpointState {
	p.mu.RLock()
	state, ok := p.state[endpointID]
	p.mu.RUnlock()
	if ok {
		return state
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if state, ok = p.state[endpointID]; ok {
		return state
	}
	ttft, err := tdigest.New(tdigest.Compression(p.cfg.compression))
	if err != nil {
		return nil
	}
	decode, err := tdigest.New(tdigest.Compression(p.cfg.compression))
	if err != nil {
		return nil
	}
	state = &endpointState{ttft: ttft, decode: decode}
	p.state[endpointID] = state
	return state
}

// addTTFT ingests one TTFT sample for the endpoint.
func (p *Observer) addTTFT(endpointID string, ttftSeconds float64) {
	if ttftSeconds < 0 {
		return
	}
	state := p.stateForOrCreate(endpointID)
	if state == nil {
		return
	}
	state.mu.Lock()
	_ = state.ttft.Add(ttftSeconds)
	state.mu.Unlock()
}

// addDecode ingests one decode sample for the endpoint.
func (p *Observer) addDecode(endpointID string, decodeSeconds float64) {
	if decodeSeconds < 0 {
		return
	}
	state := p.stateForOrCreate(endpointID)
	if state == nil {
		return
	}
	state.mu.Lock()
	_ = state.decode.Add(decodeSeconds)
	state.mu.Unlock()
}

// noteDispatch records the dispatch of a request to an endpoint. Called from
// PreRequest. Writes both inflight and requestToEndpoint under p.mu so the
// invariant on requestToEndpoint holds.
func (p *Observer) noteDispatch(endpointID, requestID string, dispatchedAt time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()
	byReq, ok := p.inflight[endpointID]
	if !ok {
		byReq = make(map[string]*inflightEntry)
		p.inflight[endpointID] = byReq
	}
	byReq[requestID] = &inflightEntry{dispatchedAt: dispatchedAt}
	p.requestToEndpoint[requestID] = endpointID
}

// noteFirstChunk records the first-chunk timestamp on the in-flight entry.
// Returns the entry's dispatched timestamp and the endpoint ID it belonged to,
// so the caller can emit a TTFT sample; returns zero Time / empty endpointID
// when no in-flight entry exists for the request. O(1) lookup via
// requestToEndpoint; the request stays dispatched-not-completed, so the
// reverse index is not mutated.
func (p *Observer) noteFirstChunk(requestID string, firstChunkAt time.Time) (dispatchedAt time.Time, endpointID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	epID, ok := p.requestToEndpoint[requestID]
	if !ok {
		return time.Time{}, ""
	}
	entry, ok := p.inflight[epID][requestID]
	if !ok {
		return time.Time{}, ""
	}
	entry.firstChunkAt = firstChunkAt
	return entry.dispatchedAt, epID
}

// noteEndOfStream removes the in-flight entry for the request. Returns the
// entry's firstChunkAt and endpointID so the caller can emit a decode sample.
// Returns zero Time / empty endpointID when no entry exists. O(1) lookup
// via requestToEndpoint; both inflight and requestToEndpoint are purged.
func (p *Observer) noteEndOfStream(requestID string) (firstChunkAt time.Time, endpointID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	epID, ok := p.requestToEndpoint[requestID]
	if !ok {
		return time.Time{}, ""
	}
	byReq, ok := p.inflight[epID]
	if !ok {
		// Reverse-index points at an endpoint whose inner map was already
		// reaped (e.g. EventDelete); drop the stale index entry and treat
		// as "no entry."
		delete(p.requestToEndpoint, requestID)
		return time.Time{}, ""
	}
	entry, ok := byReq[requestID]
	if !ok {
		delete(p.requestToEndpoint, requestID)
		return time.Time{}, ""
	}
	firstChunkAt = entry.firstChunkAt
	endpointID = epID
	delete(byReq, requestID)
	if len(byReq) == 0 {
		delete(p.inflight, epID)
	}
	delete(p.requestToEndpoint, requestID)
	return firstChunkAt, endpointID
}

// InFlightRequestsFor returns one entry per dispatched-not-completed request
// on the endpoint. Package tests use it to inspect the in-flight index without
// reaching into unexported fields; the scorer reads the same data through the
// InFlightRequestsDataKey attribute the producer publishes. Order is
// unspecified.
func (p *Observer) InFlightRequestsFor(endpointID string) []attrsojourn.InFlightRequest {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.snapshotInFlightLocked(endpointID)
}

// snapshotInFlightLocked returns a fresh slice of the endpoint's in-flight
// entries. Caller holds p.mu at least for reading. The returned slice is
// independent of p.inflight so callers may sort or modify it, and a
// subsequent write to p.inflight does not race with reads of the slice.
func (p *Observer) snapshotInFlightLocked(endpointID string) []attrsojourn.InFlightRequest {
	byReq, ok := p.inflight[endpointID]
	if !ok {
		return nil
	}
	out := make([]attrsojourn.InFlightRequest, 0, len(byReq))
	for _, entry := range byReq {
		out = append(out, attrsojourn.InFlightRequest{
			DispatchedAt: entry.dispatchedAt,
			FirstChunkAt: entry.firstChunkAt,
		})
	}
	return out
}
