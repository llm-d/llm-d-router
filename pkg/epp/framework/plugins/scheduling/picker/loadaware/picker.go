/*
Copyright 2026 The Kubernetes Authors.

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

// Package loadaware implements the load-aware picker plugin.
// See README.md for motivation, algorithm overview, and usage guidance.
package loadaware

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// LoadAwarePickerType is the registered type name for the load-aware picker plugin.
	LoadAwarePickerType = "load-aware-picker"

	// defaultEMADecay is the exponential moving average decay factor.
	// 0.1 gives approximately 10 observations to converge: 0.1*new + 0.9*history.
	defaultEMADecay = 0.1

	// concentrationWindowBuckets is the number of 1-second buckets in the sliding pick window.
	concentrationWindowBuckets = 10

	// candidateCap is the maximum number of candidates evaluated under the mutex.
	// For pools larger than this, the remaining endpoints are sampled randomly.
	candidateCap = 32

	// requestCacheTTL is how long a dispatched-request record is kept before expiry.
	requestCacheTTL = 5 * time.Minute
)

// Package-level metric singletons registered once per process via registerMetrics.
var (
	pendingGauge = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "load_aware_picker_pending_requests",
		Help:      "Number of pending (picked but not yet dispatched) requests per endpoint.",
	}, []string{"endpoint"})

	pickCounter = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "load_aware_picker_picks_total",
		Help:      "Total picks made by the load-aware picker per endpoint.",
	}, []string{"endpoint"})
)

// registerMetrics registers the package-level metric collectors with registerer.
// Re-registration of the same collector is silently accepted; any other error is returned.
func registerMetrics(registerer prometheus.Registerer) error {
	for _, collector := range []prometheus.Collector{pendingGauge, pickCounter} {
		if err := registerer.Register(collector); err != nil {
			var already prometheus.AlreadyRegisteredError
			if errors.As(err, &already) && already.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register load-aware picker metric: %w", err)
		}
	}
	return nil
}

// compile-time interface assertions
var _ fwksched.Picker = &LoadAwarePicker{}
var _ fwkrc.PreRequest = &LoadAwarePicker{}
var _ fwkrc.ResponseBodyProcessor = &LoadAwarePicker{}

// LoadAwarePickerFactory is the factory function for LoadAwarePicker.
func LoadAwarePickerFactory(name string, _ *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	p, err := newLoadAwarePicker(name, handle)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s: %w", LoadAwarePickerType, err)
	}
	return p, nil
}

// newLoadAwarePicker constructs a LoadAwarePicker and registers its metrics.
func newLoadAwarePicker(name string, handle fwkplugin.Handle) (*LoadAwarePicker, error) {
	registerer := prometheus.DefaultRegisterer
	if handle != nil {
		registerer = handle.Metrics()
	}
	if err := registerMetrics(registerer); err != nil {
		return nil, err
	}

	cache := ttlcache.New(
		ttlcache.WithTTL[string, *requestRecord](requestCacheTTL),
	)
	go cache.Start()

	if name == "" {
		name = LoadAwarePickerType
	}
	return &LoadAwarePicker{
		typedName:    fwkplugin.TypedName{Type: LoadAwarePickerType, Name: name},
		state:        make(map[string]*endpointLiveState),
		requestCache: cache,
	}, nil
}

// LoadAwarePicker serializes the final endpoint selection step behind a mutex,
// applying concentration and capacity score multipliers derived from live state.
type LoadAwarePicker struct {
	typedName fwkplugin.TypedName

	mu           sync.RWMutex
	state        map[string]*endpointLiveState // keyed by NamespacedName.String(); guarded by mu
	totalBuckets bucketedCounter               // total pool-wide picks, under mu

	requestCache *ttlcache.Cache[string, *requestRecord]
}

// TypedName implements plugin.Plugin.
func (p *LoadAwarePicker) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Pick implements scheduling.Picker.
// Candidates are sampled before the mutex; score adjustment and selection happen inside.
func (p *LoadAwarePicker) Pick(ctx context.Context, scoredPods []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	logger := log.FromContext(ctx).V(logutil.DEBUG)

	poolSize := len(scoredPods)
	if poolSize == 0 {
		return &fwksched.ProfileRunResult{}
	}

	k := min(poolSize, candidateCap)
	candidates := sampleCandidates(scoredPods, k)

	p.mu.Lock()
	defer p.mu.Unlock()

	totalPicks := p.totalBuckets.Count()
	expectedShare := 0.0
	if poolSize > 1 {
		expectedShare = float64(totalPicks) / float64(poolSize)
	}

	var best *fwksched.ScoredEndpoint
	var bestKey string
	var bestSt *endpointLiveState
	bestScore := math.Inf(-1)
	for _, ep := range candidates {
		key := ep.GetMetadata().NamespacedName.String()
		st := p.liveStateFor(key)
		score := ep.Score * p.concentrationFactor(st, expectedShare, poolSize) * p.capacityFactor(ep, st)
		if logger.Enabled() {
			logger.Info("candidate score", "endpoint", key, "rawScore", ep.Score, "adjustedScore", score)
		}
		if score > bestScore {
			bestScore = score
			best = ep
			bestKey = key
			bestSt = st
		}
	}

	bestSt.pendingRequests.Add(1)
	bestSt.pickBuckets.Inc()
	p.totalBuckets.Inc()

	pendingGauge.WithLabelValues(bestKey).Set(float64(bestSt.pendingRequests.Load()))
	pickCounter.WithLabelValues(bestKey).Inc()

	return &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{best}}
}

// PreRequest implements requestcontrol.PreRequest.
// Decrements the pending counter for the selected endpoint and records the dispatch time.
func (p *LoadAwarePicker) PreRequest(ctx context.Context, req *fwksched.InferenceRequest, result *fwksched.SchedulingResult) {
	if req == nil || result == nil {
		return
	}
	primary := result.ProfileResults[result.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return
	}
	key := primary.TargetEndpoints[0].GetMetadata().NamespacedName.String()

	p.mu.RLock()
	st, ok := p.state[key]
	p.mu.RUnlock()
	if ok {
		remaining := st.pendingRequests.Add(-1)
		pendingGauge.WithLabelValues(key).Set(float64(remaining))
	}

	p.requestCache.Set(req.RequestID, &requestRecord{
		dispatchTime: time.Now(),
	}, ttlcache.DefaultTTL)
}

// ResponseBody implements requestcontrol.ResponseBodyProcessor.
// Updates per-endpoint capacity, latency, and token-per-request EMAs on completion.
func (p *LoadAwarePicker) ResponseBody(ctx context.Context, req *fwksched.InferenceRequest, resp *fwkrc.Response, ep *fwkdl.EndpointMetadata) {
	if req == nil || resp == nil || ep == nil {
		return
	}
	if !resp.EndOfStream || resp.Usage.CompletionTokens == 0 {
		// Zero CompletionTokens filters errors and client disconnects.
		return
	}

	item := p.requestCache.Get(req.RequestID)
	if item == nil {
		return
	}
	rec := item.Value()
	p.requestCache.Delete(req.RequestID)

	duration := time.Since(rec.dispatchTime).Seconds()
	if duration <= 0 {
		return
	}

	inputTokens := 0
	if req.Body != nil && req.Body.TokenizedPrompt != nil {
		inputTokens = len(req.Body.TokenizedPrompt.TokenIDs)
	}
	tokens := float64(inputTokens + resp.Usage.CompletionTokens)
	throughput := tokens / duration

	key := ep.NamespacedName.String()
	p.mu.Lock()
	st := p.liveStateFor(key)
	if st.capacityEMA == 0 {
		st.capacityEMA = throughput
		st.avgLatency = duration
		st.avgTokensPerReq = tokens
	} else {
		st.capacityEMA = defaultEMADecay*throughput + (1-defaultEMADecay)*st.capacityEMA
		st.avgLatency = defaultEMADecay*duration + (1-defaultEMADecay)*st.avgLatency
		st.avgTokensPerReq = defaultEMADecay*tokens + (1-defaultEMADecay)*st.avgTokensPerReq
	}
	p.mu.Unlock()
}

// liveStateFor returns the live state for the given endpoint key, lazily initialising it.
// Must be called with p.mu held for writing.
func (p *LoadAwarePicker) liveStateFor(key string) *endpointLiveState {
	if st, ok := p.state[key]; ok {
		return st
	}
	st := &endpointLiveState{}
	p.state[key] = st
	return st
}

// concentrationFactor returns a multiplier in (0, 1] that penalises endpoints selected
// more than their expected share of picks in the current window.
func (p *LoadAwarePicker) concentrationFactor(st *endpointLiveState, expectedShare float64, poolSize int) float64 {
	if poolSize <= 1 || expectedShare <= 0 {
		return 1.0
	}
	actual := float64(st.pickBuckets.Count())
	if actual <= expectedShare {
		return 1.0
	}
	return expectedShare / actual
}

// capacityFactor returns a multiplier in [0, 1] derived from Little's Law.
// Returns 1.0 during cold start (before any EMA data is available).
func (p *LoadAwarePicker) capacityFactor(ep *fwksched.ScoredEndpoint, st *endpointLiveState) float64 {
	if st.capacityEMA <= 0 || st.avgLatency <= 0 || st.avgTokensPerReq <= 0 {
		return 1.0
	}
	maxConcurrency := st.capacityEMA * st.avgLatency / st.avgTokensPerReq
	if maxConcurrency <= 0 {
		return 1.0
	}
	committedRequests := float64(ep.GetMetrics().RunningRequestsSize)
	pendingRequests := float64(st.pendingRequests.Load())
	return math.Max(0, 1-((committedRequests+pendingRequests)/maxConcurrency))
}

// sampleCandidates returns all endpoints when k == len(all), otherwise a random
// sample of k drawn without replacement using the shared PickerRand.
func sampleCandidates(all []*fwksched.ScoredEndpoint, k int) []*fwksched.ScoredEndpoint {
	if k >= len(all) {
		return all
	}
	candidates := make([]*fwksched.ScoredEndpoint, len(all))
	copy(candidates, all)
	picker.ShuffleScoredEndpoints(candidates)
	return candidates[:k]
}

// endpointLiveState holds per-endpoint counters and EMA values updated on every pick/response.
// pendingRequests is an atomic so it can be decremented in PreRequest without the mutex.
// All other fields are read and written under the picker mutex.
type endpointLiveState struct {
	pendingRequests atomic.Int64
	pickBuckets     bucketedCounter
	capacityEMA     float64
	avgLatency      float64
	avgTokensPerReq float64
}

// requestRecord is stored in the TTL cache at PreRequest time.
type requestRecord struct {
	dispatchTime time.Time
}

// bucketedCounter is a sliding-window counter using a fixed ring of 1-second buckets.
// Inc and Count are O(1) and O(buckets) respectively. Not concurrency-safe on its own;
// callers must hold the picker mutex when invoking Inc or Count.
type bucketedCounter struct {
	counts [concentrationWindowBuckets]int64
	times  [concentrationWindowBuckets]int64 // unix seconds, one per slot
	pos    int
	total  int64
}

// Inc increments the counter for the current second, expiring stale buckets.
func (b *bucketedCounter) Inc() {
	now := time.Now().Unix()
	if b.times[b.pos] != now {
		b.pos = (b.pos + 1) % len(b.counts)
		b.total -= b.counts[b.pos]
		b.counts[b.pos] = 0
		b.times[b.pos] = now
	}
	b.counts[b.pos]++
	b.total++
}

// Count returns the total picks across the active window.
func (b *bucketedCounter) Count() int64 { return b.total }
