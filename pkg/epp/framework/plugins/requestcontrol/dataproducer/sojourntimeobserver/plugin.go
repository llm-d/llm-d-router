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
	// SojournTimeObserverProducerType is the plugin type of this producer.
	SojournTimeObserverProducerType = observerconstants.SojournTimeObserverProducerType
)

// InFlightRequest is one dispatched-not-completed request on an endpoint, as
// the mrl-scorer-hub reads it via InFlightRequestsFor. Wall-clock timestamps.
// FirstChunkAt is the zero Time when no chunk has arrived yet — the caller
// checks IsZero() to decide which term of the two-term MRL residual applies.
type InFlightRequest struct {
	DispatchedAt time.Time
	FirstChunkAt time.Time
}

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

// Observer records per-endpoint TTFT and decode samples into per-endpoint
// t-digests and publishes serialized snapshots. It also maintains a
// fleet-wide in-flight index (dispatchedAt/firstChunkAt per request per
// endpoint) that the mrl-scorer-hub reads at scoring time via
// InFlightRequestsFor.
//
// Process-lifetime singleton serving every request, so all shared state is
// guarded.
type Observer struct {
	typedName fwkplugin.TypedName
	cfg       resolvedConfig

	snapshotDataKey fwkplugin.DataKey // what this producer publishes

	// mu guards the two maps below. Per-endpoint digest mutation takes
	// endpointState.mu; the two locks are never held simultaneously in the
	// same direction, so lock order is: mu (for state or inflight) then
	// endpointState.mu.
	mu sync.RWMutex

	// state carries the paired digests plus published snapshot pointer per
	// endpoint.
	state map[string]*endpointState

	// inflight is the fleet-wide in-flight index: for each endpoint, one
	// entry per dispatched-not-completed request on that endpoint. Written
	// on PreRequest, first-chunk, and end-of-stream. Read by the scorer via
	// InFlightRequestsFor.
	inflight map[string]map[string]*inflightEntry
}

// inflightEntry is one dispatched-not-completed request's timestamps.
// FirstChunkAt is the zero Time when no chunk has arrived yet.
type inflightEntry struct {
	dispatchedAt time.Time
	firstChunkAt time.Time
}

// endpointState carries one endpoint's TTFT and decode digests plus its
// currently-published snapshot pointer.
type endpointState struct {
	mu     sync.Mutex
	ttft   *tdigest.TDigest
	decode *tdigest.TDigest

	// published is read lock-free by the DynamicAttribute closure the scorer
	// reads through. It is set to non-nil only when both digests are warm.
	published atomic.Pointer[attrsojourn.SojournEstimatorSnapshot]
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

// NewObserver initializes an Observer.
//
// The observer keeps its own fleet-wide in-flight index rather than routing
// per-request timestamps through PluginState, because the mrl-scorer-hub
// needs to enumerate live in-flight requests per endpoint at scoring time
// and PluginState is keyed by request ID.
func NewObserver(name string, cfg Config) (*Observer, error) {
	resolved, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	return &Observer{
		typedName:       fwkplugin.TypedName{Type: SojournTimeObserverProducerType, Name: name},
		cfg:             resolved,
		snapshotDataKey: attrsojourn.SojournEstimatorSnapshotDataKey.WithNonEmptyProducerName(name),
		state:           map[string]*endpointState{},
		inflight:        map[string]map[string]*inflightEntry{},
	}, nil
}

// TypedName implements fwkplugin.Plugin.
func (p *Observer) TypedName() fwkplugin.TypedName { return p.typedName }

// Produces declares the snapshot this producer publishes.
func (p *Observer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.snapshotDataKey: attrsojourn.SojournEstimatorSnapshot{}}
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
		logger.Info("Attached sojourn snapshot attribute", "key", p.snapshotDataKey.String(), "endpoint", id)
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
// PreRequest.
func (p *Observer) noteDispatch(endpointID, requestID string, dispatchedAt time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()
	byReq, ok := p.inflight[endpointID]
	if !ok {
		byReq = make(map[string]*inflightEntry)
		p.inflight[endpointID] = byReq
	}
	byReq[requestID] = &inflightEntry{dispatchedAt: dispatchedAt}
}

// noteFirstChunk records the first-chunk timestamp on the in-flight entry.
// Returns the entry's dispatched timestamp and the endpoint ID it belonged to,
// so the caller can emit a TTFT sample; returns zero Time / empty endpointID
// when no in-flight entry exists for the request.
func (p *Observer) noteFirstChunk(requestID string, firstChunkAt time.Time) (dispatchedAt time.Time, endpointID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for epID, byReq := range p.inflight {
		if entry, ok := byReq[requestID]; ok {
			entry.firstChunkAt = firstChunkAt
			return entry.dispatchedAt, epID
		}
	}
	return time.Time{}, ""
}

// noteEndOfStream removes the in-flight entry for the request. Returns the
// entry's firstChunkAt and endpointID so the caller can emit a decode sample.
// Returns zero Time / empty endpointID when no entry exists.
func (p *Observer) noteEndOfStream(requestID string) (firstChunkAt time.Time, endpointID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for epID, byReq := range p.inflight {
		if entry, ok := byReq[requestID]; ok {
			firstChunkAt = entry.firstChunkAt
			endpointID = epID
			delete(byReq, requestID)
			if len(byReq) == 0 {
				delete(p.inflight, epID)
			}
			return firstChunkAt, endpointID
		}
	}
	return time.Time{}, ""
}

// InFlightRequestsFor returns one entry per dispatched-not-completed request
// on the endpoint. Called by the mrl-scorer-hub at scoring time. The returned
// slice is a fresh allocation; the caller may sort or modify it. Order is
// unspecified — the residual formula is a sum and does not depend on order.
func (p *Observer) InFlightRequestsFor(endpointID string) []InFlightRequest {
	p.mu.RLock()
	defer p.mu.RUnlock()
	byReq, ok := p.inflight[endpointID]
	if !ok {
		return nil
	}
	out := make([]InFlightRequest, 0, len(byReq))
	for _, entry := range byReq {
		out = append(out, InFlightRequest{
			DispatchedAt: entry.dispatchedAt,
			FirstChunkAt: entry.firstChunkAt,
		})
	}
	return out
}
