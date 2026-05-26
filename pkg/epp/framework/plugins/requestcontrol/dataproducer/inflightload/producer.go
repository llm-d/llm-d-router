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

package inflightload

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	inflightloadconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload/constants"
)

const (
	InFlightLoadProducerType = inflightloadconstants.InFlightLoadProducerType
	profilePrefill           = "prefill"
)

// Config controls optional behaviors of InFlightLoadProducer.
type Config struct {
	// AddEstimatedOutputTokens controls whether estimated output tokens are added to
	// the in-flight token counter. Defaults to false.
	AddEstimatedOutputTokens bool `json:"addEstimatedOutputTokens"`
}

func defaultConfig() Config {
	return Config{AddEstimatedOutputTokens: false}
}

func InFlightLoadProducerFactory(name string, decoder *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("handle is nil")
	}
	ctx := handle.Context()
	if ctx == nil {
		ctx = context.Background()
	}

	cfg := defaultConfig()
	if decoder != nil {
		if err := decoder.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to decode inflight-load-producer parameters: %w", err)
		}
	}

	return &InFlightLoadProducer{
		typedName:                fwkplugin.TypedName{Type: InFlightLoadProducerType, Name: name},
		requestTracker:           newConcurrencyTracker(),
		tokenTracker:             newConcurrencyTracker(),
		tokenEstimator:           NewSimpleTokenEstimator(),
		addEstimatedOutputTokens: cfg.AddEstimatedOutputTokens,
		dk:                       attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(name),
		PluginState:              fwkplugin.NewPluginState(ctx),
	}, nil
}

var (
	_ requestcontrol.PreRequest            = &InFlightLoadProducer{}
	_ requestcontrol.ResponseBodyProcessor = &InFlightLoadProducer{}
	_ requestcontrol.DataProducer          = &InFlightLoadProducer{}
	_ datalayer.EndpointExtractor          = (*InFlightLoadProducer)(nil)
	_ datalayer.Registrant                 = &InFlightLoadProducer{}
)

type InFlightLoadProducer struct {
	typedName                fwkplugin.TypedName
	requestTracker           *concurrencyTracker
	tokenTracker             *concurrencyTracker
	tokenEstimator           TokenEstimator
	addEstimatedOutputTokens bool
	PluginState              *fwkplugin.PluginState
	dk                       fwkplugin.DataKey
}

// addedTokensEntry tracks a request's contribution to the global token and
// request counters. OnEvicted rolls back the contribution exactly once,
// whether triggered by explicit release at end-of-stream or by the janitor's
// TTL reaper. The fields are atomic so releaseTokensEarly and OnEvicted
// can race safely: whichever swaps first does the decrement, the other
// sees 0 and is a no-op.
type addedTokensEntry struct {
	endpointID     string
	tokens         atomic.Int64
	tokenTracker   *concurrencyTracker
	requestTracker *concurrencyTracker
	requests       atomic.Int32
}

var _ fwkplugin.EvictableStateData = (*addedTokensEntry)(nil)

// Clone returns a distinct copy of the entry with the current atomic values.
// The tracker references remain shared, but the cloned state object itself is
// independent so later mutation or eviction of the clone does not alias the
// original entry.
func (e *addedTokensEntry) Clone() fwkplugin.StateData {
	if e == nil {
		return nil
	}
	clone := &addedTokensEntry{
		endpointID:     e.endpointID,
		tokenTracker:   e.tokenTracker,
		requestTracker: e.requestTracker,
	}
	clone.tokens.Store(e.tokens.Load())
	clone.requests.Store(e.requests.Load())
	return clone
}

// addIfPresent applies delta only when the endpoint is still tracked.
// This avoids recreating a deleted endpoint with a negative in-flight count
// during delayed eviction cleanup.
func (t *concurrencyTracker) addIfPresent(endpointID string, delta int64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	counter, ok := t.counts[endpointID]
	if !ok {
		return
	}
	counter.Add(delta)
}

// decIfPresent decrements the endpoint only when it is still tracked.
func (t *concurrencyTracker) decIfPresent(endpointID string) {
	t.addIfPresent(endpointID, -1)
}

func (e *addedTokensEntry) OnEvicted(_ string, _ fwkplugin.StateKey) {
	if t := e.tokens.Swap(0); t != 0 {
		e.tokenTracker.addIfPresent(e.endpointID, -t)
	}
	if e.requests.Swap(0) != 0 {
		e.requestTracker.decIfPresent(e.endpointID)
	}
}

func (p *InFlightLoadProducer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// RegisterDependencies declares that this plugin needs an endpoint-notification-source to track
// endpoint lifecycle events. The source is auto-created if not already in the config.
func (p *InFlightLoadProducer) RegisterDependencies(r datalayer.Registrar) error {
	return r.Register(datalayer.PendingRegistration{
		Owner:         p.TypedName(),
		SourceType:    sourcenotifications.EndpointNotificationSourceType,
		Extractor:     p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(sourcenotifications.EndpointNotificationSourceType, sourcenotifications.EndpointNotificationSourceType),
	})
}

// ExpectedInputType defines the type expected by the extractor.
func (p *InFlightLoadProducer) ExpectedInputType() reflect.Type {
	return datalayer.EndpointEventReflectType
}

// ExtractEndpoint handles endpoint deletion events to prune stateful trackers.
func (p *InFlightLoadProducer) ExtractEndpoint(ctx context.Context, event datalayer.EndpointEvent) error {
	if event.Type != datalayer.EventDelete || event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}

	id := event.Endpoint.GetMetadata().NamespacedName.String()

	p.DeleteEndpoint(id)
	log.FromContext(ctx).V(logutil.DEFAULT).Info("Cleaned up in-flight load for deleted endpoint", "endpoint", id)
	return nil
}

func (p *InFlightLoadProducer) Produce(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	var inputTokens int64
	if request != nil {
		inputTokens = p.tokenEstimator.EstimateInput(request)
	}

	for _, e := range endpoints {
		if e == nil || e.GetMetadata() == nil {
			continue
		}
		endpointID := e.GetMetadata().NamespacedName.String()

		load := &attrconcurrency.InFlightLoad{
			Tokens:   p.tokenTracker.get(endpointID),
			Requests: p.requestTracker.get(endpointID),
		}
		if request != nil {
			// Project this request's additional work onto the endpoint: uncached
			// input tokens (per its prefix-cache state) plus estimated output
			// when the producer is configured to include it. Per-cycle, not
			// persisted in the tracker — that happens only on PreRequest.
			load.UncachedRequestTokens = p.estimateRequestTokens(e, inputTokens)
		}
		e.Put(p.dk.String(), load)
	}
	return nil
}

func (p *InFlightLoadProducer) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) {
	if result == nil || len(result.ProfileResults) == 0 {
		return
	}

	if request == nil {
		log.FromContext(ctx).V(logutil.VERBOSE).Info("Skipping in-flight load tracking: request is nil")
		return
	}

	if request.RequestID == "" {
		log.FromContext(ctx).V(logutil.VERBOSE).Info("Skipping in-flight load tracking: missing RequestID")
		return
	}

	if p.PluginState == nil {
		log.FromContext(ctx).V(logutil.VERBOSE).Info("Skipping in-flight load tracking: PluginState is nil", "requestID", request.RequestID)
		return
	}

	inputTokens := p.tokenEstimator.EstimateInput(request)

	for profileName, profileResult := range result.ProfileResults {
		if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
			continue
		}
		// Only track the first endpoint (the primary target), as requested by reviewers.
		endpoint := profileResult.TargetEndpoints[0]
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		eid := endpoint.GetMetadata().NamespacedName.String()
		p.requestTracker.inc(eid)

		// Compute the uncached prompt portion this endpoint must actually compute.
		// Prefer the prefix producer's view (real tokens) when available so the
		// match-length and the input length are in the same units; fall back to
		// the (estimated) input tokens otherwise.
		tokens := p.estimateRequestTokens(endpoint, inputTokens)

		p.tokenTracker.add(eid, tokens)

		entry := &addedTokensEntry{
			endpointID:     eid,
			tokenTracker:   p.tokenTracker,
			requestTracker: p.requestTracker,
		}
		entry.tokens.Store(tokens)
		entry.requests.Store(1)
		p.PluginState.Write(
			request.RequestID,
			fwkplugin.StateKey(addedTokensKey(eid, profileName)),
			entry,
		)
	}
}

func (p *InFlightLoadProducer) estimateRequestTokens(endpoint fwksched.Endpoint, inputTokens int64) int64 {
	adjustedInput := uncachedInputTokens(endpoint, inputTokens)
	tokens := adjustedInput
	if p.addEstimatedOutputTokens {
		// Output tokens are based on the full input, not the cached portion.
		tokens += p.tokenEstimator.EstimateOutput(inputTokens)
	}
	return tokens
}

func (p *InFlightLoadProducer) ResponseBody(
	_ context.Context,
	request *fwksched.InferenceRequest,
	resp *requestcontrol.Response,
	_ *datalayer.EndpointMetadata,
) {
	if request == nil || resp == nil || request.RequestID == "" || p.PluginState == nil {
		return
	}

	result := request.SchedulingResult
	if result == nil {
		return
	}

	// When output tokens are excluded, the in-flight token estimate represents only
	// the prompt cost, which is consumed by prefill. As soon as the first chunk
	// arrives (StartOfStream), prefill is done across all profiles, so free the
	// token counters for every targeted endpoint regardless of profile name.
	// Request counters are still released on EndOfStream below via PluginState.Delete.
	if !p.addEstimatedOutputTokens && resp.StartOfStream {
		for profileName, profileResult := range result.ProfileResults {
			if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
				continue
			}
			endpoint := profileResult.TargetEndpoints[0]
			if endpoint == nil || endpoint.GetMetadata() == nil {
				continue
			}
			p.releaseTokensEarly(endpoint, request, profileName)
		}
	}

	// Early prefill release (on first chunk). Frees the primary profile's
	// prefill contribution as soon as prefill completes, while other profiles'
	// entries remain until EndOfStream.
	if p.addEstimatedOutputTokens && resp.StartOfStream {
		if prefillResult, ok := result.ProfileResults[profilePrefill]; ok && len(prefillResult.TargetEndpoints) > 0 {
			endpoint := prefillResult.TargetEndpoints[0]
			if endpoint != nil && endpoint.GetMetadata() != nil {
				p.release(endpoint, request, profilePrefill)
			}
		}
	}

	// Full cleanup on completion vs. lifetime extension on an intermediate chunk.
	// PluginState.Delete iterates remaining entries via per-key LoadAndDelete,
	// firing OnEvicted at most once per entry; entries already released at
	// StartOfStream are gracefully no-op'd (LoadAndDelete miss / atomic Swap-to-0).
	if resp.EndOfStream {
		p.PluginState.Delete(request.RequestID)
	} else {
		p.PluginState.Touch(request.RequestID)
	}
}

// release surgically deletes a single profile's entry from PluginState,
// triggering OnEvicted to roll back that profile's counter contribution.
// Used at StartOfStream when a single profile needs to be released ahead of
// the EndOfStream bulk Delete.
func (p *InFlightLoadProducer) release(endpoint fwksched.Endpoint, request *fwksched.InferenceRequest, profileName string) {
	if endpoint == nil || request == nil || request.RequestID == "" || p.PluginState == nil {
		return
	}
	meta := endpoint.GetMetadata()
	if meta == nil {
		return
	}
	eid := meta.NamespacedName.String()
	key := fwkplugin.StateKey(addedTokensKey(eid, profileName))

	// DeleteKey triggers OnEvicted, which decrements the counters exactly once.
	// If the janitor already reaped the request, this is a no-op.
	p.PluginState.DeleteKey(request.RequestID, key)
}

// releaseTokensEarly frees only the token portion of a profile's entry
// (request counter stays held), used at StartOfStream for the
// addEstimatedOutputTokens=false path where prefill completion frees tokens
// but the request remains in-flight until EndOfStream.
func (p *InFlightLoadProducer) releaseTokensEarly(endpoint fwksched.Endpoint, request *fwksched.InferenceRequest, profileName string) {
	if endpoint == nil || request == nil || request.RequestID == "" || p.PluginState == nil {
		return
	}
	meta := endpoint.GetMetadata()
	if meta == nil {
		return
	}
	eid := meta.NamespacedName.String()

	key := fwkplugin.StateKey(addedTokensKey(eid, profileName))
	if entry, err := fwkplugin.ReadPluginStateKey[*addedTokensEntry](p.PluginState, request.RequestID, key); err == nil {
		if t := entry.tokens.Swap(0); t != 0 {
			entry.tokenTracker.addIfPresent(entry.endpointID, -t)
		}
	}
}

func addedTokensKey(endpointID, profileName string) string {
	return endpointID + "|" + profileName + "|added"
}

// uncachedInputTokens returns the prompt tokens this endpoint must actually compute,
// excluding any prefix already cached on it.
//
// When the approximate prefix producer has populated PrefixCacheMatchInfo on the
// endpoint, the matched and total block counts are in real (tokenized) units, so
// we use them directly: uncached = (TotalBlocks - MatchBlocks) * BlockSizeTokens.
// For very long prompts where the prefix index is capped (MaxPrefixTokensToMatch),
// any tail beyond the cap is added back from the (estimated) inputTokens so the
// full prompt cost is still reflected.
//
// When the attribute is missing, we fall back to the estimated inputTokens.
func uncachedInputTokens(endpoint fwksched.Endpoint, inputTokens int64) int64 {
	if endpoint == nil {
		return nonNeg(inputTokens)
	}
	raw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoDataKey.String())
	if !ok {
		return nonNeg(inputTokens)
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok || info == nil || info.BlockSizeTokens() <= 0 {
		return nonNeg(inputTokens)
	}

	blockSize := int64(info.BlockSizeTokens())
	matched := int64(info.MatchBlocks()) * blockSize
	indexed := int64(info.TotalBlocks()) * blockSize

	uncachedIndexed := indexed - matched
	if uncachedIndexed < 0 {
		uncachedIndexed = 0
	}

	// Tail beyond the indexed portion (e.g., when MaxPrefixTokensToMatch caps total).
	tail := inputTokens - indexed
	if tail < 0 {
		tail = 0
	}

	return uncachedIndexed + tail
}

func nonNeg(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
}

func (p *InFlightLoadProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.dk: attrconcurrency.InFlightLoad{},
	}
}

func (p *InFlightLoadProducer) Consumes() map[string]any {
	return map[string]any{
		attrprefix.PrefixCacheMatchInfoDataKey.String(): (*attrprefix.PrefixCacheMatchInfo)(nil),
	}
}

// DeleteEndpoint removes an endpoint from the concurrency trackers to prevent memory leaks.
// This matches the design of the previous saturation detector and is called by the
// ExtractNotification hook to ensure deterministic cleanup of stateful data.
func (p *InFlightLoadProducer) DeleteEndpoint(endpointID string) {
	p.requestTracker.delete(endpointID)
	p.tokenTracker.delete(endpointID)
}

// concurrencyTracker manages thread-safe counters for inflight requests.
type concurrencyTracker struct {
	mu     sync.RWMutex
	counts map[string]*atomic.Int64
}

func newConcurrencyTracker() *concurrencyTracker {
	return &concurrencyTracker{
		counts: make(map[string]*atomic.Int64),
	}
}

func (t *concurrencyTracker) get(endpointID string) int64 {
	t.mu.RLock()
	counter, exists := t.counts[endpointID]
	t.mu.RUnlock()

	if !exists {
		return 0
	}
	return counter.Load()
}

func (t *concurrencyTracker) inc(endpointID string) {
	t.add(endpointID, 1)
}

func (t *concurrencyTracker) add(endpointID string, delta int64) {
	t.mu.RLock()
	counter, exists := t.counts[endpointID]
	t.mu.RUnlock()

	if exists {
		counter.Add(delta)
		return
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	if counter, exists = t.counts[endpointID]; exists {
		counter.Add(delta)
		return
	}

	counter = &atomic.Int64{}
	counter.Store(delta)
	t.counts[endpointID] = counter
}

func (t *concurrencyTracker) delete(endpointID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.counts, endpointID)
}
