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

// Package latencyobserver measures each request's time-to-first-token per
// endpoint and publishes a summary for the latency-observation scorer.
//
// It spans two layers because neither can do the job alone: request-control
// hooks are the only place traffic is visible, while the attribute the scorer
// reads is a datalayer concept. The summary is published through a
// DynamicAttribute attached once per endpoint, as inflight-load-producer does.
package latencyobserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	observerconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/latencyobserver/constants"
)

const (
	// LatencyObserverProducerType is the plugin type of this producer.
	LatencyObserverProducerType = observerconstants.LatencyObserverProducerType

	// dispatchStateKey names the per-request record written in PreRequest and
	// read back when the first response chunk arrives.
	dispatchStateKey = fwkplugin.StateKey("dispatch")

	// inflightStateKey names the per-request in-flight readings Produce takes
	// for every candidate, which PreRequest narrows to the winner.
	inflightStateKey = fwkplugin.StateKey("inflight-at-dispatch")

	// maxDebugDumpEndpoints bounds the DumpState payload on large deployments.
	maxDebugDumpEndpoints = 100
)

var (
	_ fwkplugin.StateDumper    = &Observer{}
	_ fwkplugin.ProducerPlugin = &Observer{}
	_ fwkplugin.ConsumerPlugin = &Observer{}
	_ fwkdl.EndpointExtractor  = (*Observer)(nil)
	_ fwkdl.Registrant         = &Observer{}
	_ fwkdl.PollingDispatcher  = &Observer{}
)

// Config holds the observer's parameters; durations are time.ParseDuration
// strings, e.g. "1s". See README.md for what each one does.
type Config struct {
	// Which inflight-load-producer to read. Empty uses the default producer.
	InFlightLoadProducerName string `json:"inFlightLoadProducerName,omitempty"`

	IntervalDuration string `json:"intervalDuration,omitempty"`
	// Ring-buffer capacity, allocated up front. It bounds retention by count,
	// so at high request rates this, rather than MaxObservationAge, is what
	// limits how far back the window actually reaches.
	WindowSize        int    `json:"windowSize,omitempty"`
	MaxObservationAge string `json:"maxObservationAge,omitempty"`
	// Caps the short window to the newest N observations. Must be <= WindowSize.
	MaxRequests int `json:"maxRequests,omitempty"`
	// Evidence threshold below which the scorer treats an endpoint as
	// uncalibrated. Published in the snapshot so consumers need no own copy.
	MinRequests int `json:"minRequests,omitempty"`
	// The two load anchors. Must satisfy 0 < low < typical < 100.
	LowPercentile     int `json:"lowPercentile,omitempty"`
	TypicalPercentile int `json:"typicalPercentile,omitempty"`
	// Window for each floor-history entry's P10. Keep <= MaxObservationAge so a
	// full bucket is retained.
	BucketDuration string `json:"bucketDuration,omitempty"`
	// Per-bucket P10s kept for the floor. Evicted by value rather than age, so
	// this bounds memory and smooths the floor rather than setting a time
	// horizon. Must be >= 2.
	BucketHistorySize int `json:"bucketHistorySize,omitempty"`
}

// DefaultConfig is decoded over by the factory.
var DefaultConfig = Config{
	IntervalDuration:  "1s",
	WindowSize:        5000,
	MaxObservationAge: "3m",
	MaxRequests:       100,
	MinRequests:       10,
	LowPercentile:     25,
	TypicalPercentile: 50,
	BucketDuration:    "1m",
	BucketHistorySize: 1000,
}

// resolvedConfig is Config after parsing and validation. Percentiles are
// fractions here, e.g. 0.25.
type resolvedConfig struct {
	interval          time.Duration
	maxObservationAge time.Duration
	bucketDuration    time.Duration
	windowSize        int
	maxRequests       int
	minRequests       int
	bucketHistorySize int
	lowPercentile     float64
	typicalPercentile float64
}

func (c Config) resolve() (resolvedConfig, error) {
	var out resolvedConfig
	var err error

	if out.interval, err = positiveDuration("intervalDuration", c.IntervalDuration); err != nil {
		return out, err
	}
	if out.maxObservationAge, err = positiveDuration("maxObservationAge", c.MaxObservationAge); err != nil {
		return out, err
	}
	if out.bucketDuration, err = positiveDuration("bucketDuration", c.BucketDuration); err != nil {
		return out, err
	}

	switch {
	case c.WindowSize <= 0:
		return out, fmt.Errorf("windowSize must be > 0, got %d", c.WindowSize)
	case c.MaxRequests <= 0:
		return out, fmt.Errorf("maxRequests must be > 0, got %d", c.MaxRequests)
	case c.MaxRequests > c.WindowSize:
		return out, fmt.Errorf("maxRequests (%d) must be <= windowSize (%d)", c.MaxRequests, c.WindowSize)
	case c.MinRequests <= 0:
		return out, fmt.Errorf("minRequests must be > 0, got %d", c.MinRequests)
	case c.BucketHistorySize < 2:
		return out, fmt.Errorf("bucketHistorySize must be >= 2, got %d", c.BucketHistorySize)
	case c.LowPercentile <= 0 || c.TypicalPercentile >= 100 || c.LowPercentile >= c.TypicalPercentile:
		return out, fmt.Errorf("percentiles must satisfy 0 < lowPercentile (%d) < typicalPercentile (%d) < 100",
			c.LowPercentile, c.TypicalPercentile)
	}

	out.windowSize, out.maxRequests, out.minRequests = c.WindowSize, c.MaxRequests, c.MinRequests
	out.bucketHistorySize = c.BucketHistorySize
	out.lowPercentile = float64(c.LowPercentile) / 100
	out.typicalPercentile = float64(c.TypicalPercentile) / 100
	return out, nil
}

func positiveDuration(field, raw string) (time.Duration, error) {
	d, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid %s %q: %w", field, raw, err)
	}
	if d <= 0 {
		return 0, fmt.Errorf("%s must be > 0, got %v", field, d)
	}
	return d, nil
}

// Observer records per-endpoint TTFT observations. It is a process-lifetime
// singleton serving every request, so all shared state is guarded.
type Observer struct {
	typedName   fwkplugin.TypedName
	PluginState *fwkplugin.PluginState
	cfg         resolvedConfig

	percentilesDataKey  fwkplugin.DataKey // what this producer publishes
	inFlightLoadDataKey fwkplugin.DataKey // the live in-flight signal it reads

	// mu guards the map itself. Per-endpoint mutation takes that endpoint's own
	// lock, so a slow update never blocks another endpoint. Lock order is always
	// mu then endpointState.mu; nothing acquires them the other way round.
	mu    sync.RWMutex
	state map[string]*endpointState
}

// endpointState is one endpoint's observation state.
type endpointState struct {
	mu      sync.Mutex
	tracker *slidingWindowTracker

	// The floor: once per bucketDuration the bucket's P10 joins the history,
	// and floor is a low percentile of that history.
	bucketStart time.Time
	bucketP10s  []float64
	floor       float64

	observations           int64
	lastTTFT               float64
	lastInflightAtDispatch int64

	// Read by the scorer through the DynamicAttribute closure. Nil until the
	// first flush, which reads as cold.
	published atomic.Pointer[attrlatency.TTFTPercentiles]
}

// LatencyObserverFactory builds an Observer. The recompute is driven by the
// datalayer once the observer is listed under dataLayer.sources; see README.md.
func LatencyObserverFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	ctx := handle.Context()
	if ctx == nil {
		ctx = context.Background()
	}

	cfg := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for plugin %q: %w", name, err)
		}
	}

	observer, err := NewObserver(ctx, name, cfg)
	if err != nil {
		return nil, fmt.Errorf("invalid parameters for plugin %q: %w", name, err)
	}
	return observer, nil
}

// NewObserver initializes an Observer whose per-request state is bound to ctx.
func NewObserver(ctx context.Context, name string, cfg Config) (*Observer, error) {
	resolved, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	return &Observer{
		typedName:           fwkplugin.TypedName{Type: LatencyObserverProducerType, Name: name},
		PluginState:         fwkplugin.NewPluginState(ctx),
		cfg:                 resolved,
		percentilesDataKey:  attrlatency.TTFTPercentilesDataKey.WithNonEmptyProducerName(name),
		inFlightLoadDataKey: attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(cfg.InFlightLoadProducerName),
		state:               map[string]*endpointState{},
	}, nil
}

func (p *Observer) TypedName() fwkplugin.TypedName { return p.typedName }

// RegisterDependencies asks for endpoint lifecycle events, so the observer can
// allocate per-endpoint state and attach its published attribute. The source is
// auto-created when the config omits it.
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
// DynamicAttribute whose closure resolves to the latest snapshot, so the
// attribute map is written exactly once per endpoint and each flush only swaps
// the pointer behind it.
//
// A delete racing the add of a recreated endpoint of the same name can discard
// fresh state; that endpoint then reads cold and re-calibrates, so the cost is
// lost observations, not a wrong value.
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
		p.mu.Unlock()
		logger.Info("Dropped TTFT observations for deleted endpoint", "endpoint", id)

	case fwkdl.EventAddOrUpdate:
		state := p.stateFor(id)
		event.Endpoint.GetAttributes().Put(p.percentilesDataKey, &fwkdl.DynamicAttribute{
			Get: func() fwkdl.Cloneable {
				if snapshot := state.published.Load(); snapshot != nil {
					return snapshot
				}
				// Absent, not an empty struct, so the endpoint reads as cold.
				return nil
			},
		})
		logger.Info("Attached TTFT percentiles attribute", "key", p.percentilesDataKey.String(), "endpoint", id)
	}
	return nil
}

// stateFor returns the endpoint's state, creating it if absent.
func (p *Observer) stateFor(endpointID string) *endpointState {
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
	state = &endpointState{tracker: newSlidingWindowTracker(p.cfg.windowSize)}
	p.state[endpointID] = state
	return state
}

// record stores one TTFT observation for an endpoint. at is the observation's
// timestamp; see slidingWindowTracker.window for the ordering it assumes.
func (p *Observer) record(endpointID string, ttftSeconds float64, inflightAtDispatch int64, at time.Time) {
	state := p.stateFor(endpointID)
	state.mu.Lock()
	state.tracker.add(ttftSeconds, inflightAtDispatch, at)
	state.lastTTFT, state.lastInflightAtDispatch = ttftSeconds, inflightAtDispatch
	state.observations++
	state.mu.Unlock()
}

type observerDebugState struct {
	Endpoints      []endpointDebugState `json:"endpoints"`
	TotalEndpoints int                  `json:"totalEndpoints"`
	MaxEndpoints   int                  `json:"maxEndpoints"`
	Truncated      bool                 `json:"truncated"`
}

type endpointDebugState struct {
	Endpoint               string  `json:"endpoint"`
	LastTTFTSeconds        float64 `json:"lastTtftSeconds"`
	LastInflightAtDispatch int64   `json:"lastInflightAtDispatch"`
	Observations           int64   `json:"observations"`
	FloorSeconds           float64 `json:"floorSeconds"`
	RecentRequestCount     int     `json:"recentN"`
	LowLoadTTFTSeconds     float64 `json:"lowTtftSeconds"`
	TypicalLoadTTFTSeconds float64 `json:"highTtftSeconds"`
	InflightAtLowLoad      float64 `json:"inflightAtLow"`
	InflightAtTypicalLoad  float64 `json:"inflightAtHigh"`
}

// DumpState implements [fwkplugin.StateDumper] for /debug/plugins/state.
// Endpoints are snapshotted under their own locks, so values are not from a
// single instant: best-effort visibility beats a global lock contending with
// the observation path.
func (p *Observer) DumpState() (json.RawMessage, error) {
	return json.Marshal(p.snapshotDebugState())
}

func (p *Observer) snapshotDebugState() observerDebugState {
	p.mu.RLock()
	dump := observerDebugState{
		Endpoints:    make([]endpointDebugState, 0, len(p.state)),
		MaxEndpoints: maxDebugDumpEndpoints,
	}
	for id, state := range p.state {
		state.mu.Lock()
		entry := endpointDebugState{
			Endpoint:               id,
			LastTTFTSeconds:        state.lastTTFT,
			LastInflightAtDispatch: state.lastInflightAtDispatch,
			Observations:           state.observations,
		}
		state.mu.Unlock()

		if snapshot := state.published.Load(); snapshot != nil {
			entry.FloorSeconds = snapshot.Floor()
			entry.RecentRequestCount = snapshot.RecentRequestCount
			entry.LowLoadTTFTSeconds = snapshot.LowLoadTTFT
			entry.TypicalLoadTTFTSeconds = snapshot.TypicalLoadTTFT
			entry.InflightAtLowLoad = snapshot.InflightAtLowLoad
			entry.InflightAtTypicalLoad = snapshot.InflightAtTypicalLoad
		}
		dump.Endpoints = append(dump.Endpoints, entry)
	}
	p.mu.RUnlock()

	dump.TotalEndpoints = len(dump.Endpoints)
	sort.SliceStable(dump.Endpoints, func(i, j int) bool {
		if dump.Endpoints[i].Observations != dump.Endpoints[j].Observations {
			return dump.Endpoints[i].Observations > dump.Endpoints[j].Observations
		}
		return dump.Endpoints[i].Endpoint < dump.Endpoints[j].Endpoint
	})
	if len(dump.Endpoints) > maxDebugDumpEndpoints {
		dump.Endpoints = dump.Endpoints[:maxDebugDumpEndpoints]
		dump.Truncated = true
	}
	return dump
}
