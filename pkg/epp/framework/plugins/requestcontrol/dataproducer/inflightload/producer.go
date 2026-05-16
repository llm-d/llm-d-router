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
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	sourcenotifications "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/source/notifications"
)

const (
	InFlightLoadProducerType = "inflight-load-producer"
	profilePrefill           = "prefill"
)

// Config controls optional behaviors of InFlightLoadProducer.
type Config struct {
	// IncludeOutputTokens controls whether estimated output tokens are added to
	// the in-flight token counter. Defaults to true to preserve historical behavior.
	IncludeOutputTokens bool `json:"includeOutputTokens"`
}

func defaultConfig() Config {
	return Config{IncludeOutputTokens: true}
}

func InFlightLoadProducerFactory(name string, rawParameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultConfig()
	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &cfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal inflight-load-producer parameters: %w", err)
		}
	}
	return &InFlightLoadProducer{
		typedName:           fwkplugin.TypedName{Type: InFlightLoadProducerType, Name: name},
		requestTracker:      newConcurrencyTracker(),
		tokenTracker:        newConcurrencyTracker(),
		tokenEstimator:      NewSimpleTokenEstimator(),
		includeOutputTokens: cfg.IncludeOutputTokens,
	}, nil
}

var (
	_ requestcontrol.PreRequest            = &InFlightLoadProducer{}
	_ requestcontrol.ResponseBodyProcessor = &InFlightLoadProducer{}
	_ requestcontrol.DataProducer          = &InFlightLoadProducer{}
	_ datalayer.EndpointExtractor          = &InFlightLoadProducer{}
	_ datalayer.Registrant                 = &InFlightLoadProducer{}
)

type InFlightLoadProducer struct {
	typedName           fwkplugin.TypedName
	requestTracker      *concurrencyTracker
	tokenTracker        *concurrencyTracker
	tokenEstimator      TokenEstimator
	includeOutputTokens bool
	// addedTokens tracks the exact token amount added per (requestID, endpointID, profileName)
	// so release subtracts the same value (accounting for prefix-cache discount).
	// The profile name is part of the key so that multiple profiles targeting the
	// same endpoint each track their own increment independently and a release
	// for one profile cannot subtract another profile's value (which would leak
	// or double-count tokens).
	// Key format: "<requestID>|<endpointID>|<profileName>".
	addedTokens sync.Map
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
	if event.Type != datalayer.EventDelete || event.Endpoint == nil {
		return nil
	}

	id := event.Endpoint.GetMetadata().NamespacedName.String()

	p.DeleteEndpoint(id)
	log.FromContext(ctx).V(logutil.DEFAULT).Info("Cleaned up in-flight load for deleted endpoint", "endpoint", id)
	return nil
}

func (p *InFlightLoadProducer) PrepareRequestData(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	for _, e := range endpoints {
		endpointID := e.GetMetadata().NamespacedName.String()
		e.Put(attrconcurrency.InFlightLoadKey, &attrconcurrency.InFlightLoad{
			Tokens:   p.tokenTracker.get(endpointID),
			Requests: p.requestTracker.get(endpointID),
		})
	}
	return nil
}

func (p *InFlightLoadProducer) PreRequest(_ context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) {
	if result == nil || len(result.ProfileResults) == 0 {
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
		adjustedInput := uncachedInputTokens(endpoint, inputTokens)
		tokens := adjustedInput
		if p.includeOutputTokens {
			// Output tokens are based on the full input, not the cached portion.
			tokens += p.tokenEstimator.EstimateOutput(inputTokens)
		}

		p.tokenTracker.add(eid, tokens)
		if request != nil && request.RequestID != "" {
			p.addedTokens.Store(addedTokensKey(request.RequestID, eid, profileName), tokens)
		}
	}
}

func (p *InFlightLoadProducer) ResponseBody(
	ctx context.Context,
	request *fwksched.InferenceRequest,
	resp *requestcontrol.Response,
	_ *datalayer.EndpointMetadata,
) {
	if request == nil || resp == nil {
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
	// Request counters are still released on EndOfStream below.
	if !p.includeOutputTokens && resp.StartOfStream {
		for profileName, profileResult := range result.ProfileResults {
			if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
				continue
			}
			p.releaseTokens(profileResult.TargetEndpoints[0], request, profileName)
		}
	}

	// 1. Early Prefill Release (on first chunk) — original behavior.
	// Uses the new StartOfStream signal provided by the framework.
	if p.includeOutputTokens && resp.StartOfStream {
		if prefillResult, ok := result.ProfileResults[profilePrefill]; ok && len(prefillResult.TargetEndpoints) > 0 {
			p.release(prefillResult.TargetEndpoints[0], request, profilePrefill)
		}
	}

	// 2. Full Cleanup (on completion)
	if resp.EndOfStream {
		for name, profileResult := range result.ProfileResults {
			if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
				continue
			}
			endpoint := profileResult.TargetEndpoints[0]

			if !p.includeOutputTokens {
				// Tokens are normally freed at StartOfStream; also call
				// releaseTokens here as a safety net for non-streaming or
				// error paths where StartOfStream may not be observed. It is
				// a no-op via LoadAndDelete if tokens were already released.
				p.release(endpoint, request, name)
				continue
			}

			// Skip "prefill" as it was already released in the StartOfStream block.
			// This works perfectly even if StartOfStream and EndOfStream are both true (single chunk).
			if name == profilePrefill {
				continue
			}
			p.release(endpoint, request, name)
		}
	}
}

func (p *InFlightLoadProducer) release(endpoint fwksched.Endpoint, request *fwksched.InferenceRequest, profileName string) {
	p.releaseRequest(endpoint)
	p.releaseTokens(endpoint, request, profileName)
}

func (p *InFlightLoadProducer) releaseRequest(endpoint fwksched.Endpoint) {
	if endpoint == nil || endpoint.GetMetadata() == nil {
		return
	}
	eid := endpoint.GetMetadata().NamespacedName.String()
	p.requestTracker.dec(eid)
}

func (p *InFlightLoadProducer) releaseTokens(endpoint fwksched.Endpoint, request *fwksched.InferenceRequest, profileName string) {
	if endpoint == nil || endpoint.GetMetadata() == nil {
		return
	}
	eid := endpoint.GetMetadata().NamespacedName.String()

	// Prefer the exact value stored in PreRequest to keep counters balanced.
	// LoadAndDelete makes this idempotent per (requestID, endpointID, profileName):
	// a second call for the same key finds nothing and is a no-op below (we do
	// NOT fall back to Estimate when the request carries a real RequestID, since
	// the absence of the key means "already released").
	if request != nil && request.RequestID != "" {
		key := addedTokensKey(request.RequestID, eid, profileName)
		if v, ok := p.addedTokens.LoadAndDelete(key); ok {
			if tokens, ok := v.(int64); ok && tokens != 0 {
				p.tokenTracker.add(eid, -tokens)
			}
		}
		// Either we just released the stored value, or the key was already
		// released by a previous call — both cases are no-ops here.
		return
	}

	// Fallback: re-estimate. Covers tests/legacy paths that bypass PreRequest
	// (request is nil or has no RequestID, so nothing was stored to release).
	// Mirror PreRequest's accounting so we subtract the same amount we would
	// have added: uncached input tokens, plus output tokens only when
	// includeOutputTokens is true. Using tokenEstimator.Estimate() here would
	// ignore both the includeOutputTokens=false semantics and any prefix-cache
	// discount applied at PreRequest time, leading to over/under-decrement.
	inputTokens := p.tokenEstimator.EstimateInput(request)
	tokens := uncachedInputTokens(endpoint, inputTokens)
	if p.includeOutputTokens {
		tokens += p.tokenEstimator.EstimateOutput(inputTokens)
	}
	if tokens != 0 {
		p.tokenTracker.add(eid, -tokens)
	}
}

func addedTokensKey(requestID, endpointID, profileName string) string {
	return requestID + "|" + endpointID + "|" + profileName
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
	raw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey)
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

func (p *InFlightLoadProducer) Produces() map[string]any {
	return map[string]any{
		attrconcurrency.InFlightLoadKey: attrconcurrency.InFlightLoad{},
	}
}

func (p *InFlightLoadProducer) Consumes() map[string]any {
	return map[string]any{
		attrprefix.PrefixCacheMatchInfoKey: (*attrprefix.PrefixCacheMatchInfo)(nil),
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

func (ct *concurrencyTracker) get(endpointID string) int64 {
	ct.mu.RLock()
	counter, exists := ct.counts[endpointID]
	ct.mu.RUnlock()

	if !exists {
		return 0
	}
	return counter.Load()
}

func (ct *concurrencyTracker) inc(endpointID string) {
	ct.add(endpointID, 1)
}

func (ct *concurrencyTracker) add(endpointID string, delta int64) {
	ct.mu.RLock()
	counter, exists := ct.counts[endpointID]
	ct.mu.RUnlock()

	if exists {
		counter.Add(delta)
		return
	}

	ct.mu.Lock()
	defer ct.mu.Unlock()

	if counter, exists = ct.counts[endpointID]; exists {
		counter.Add(delta)
		return
	}

	counter = &atomic.Int64{}
	counter.Store(delta)
	ct.counts[endpointID] = counter
}

func (ct *concurrencyTracker) dec(endpointID string) {
	ct.add(endpointID, -1)
}

func (ct *concurrencyTracker) delete(endpointID string) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	delete(ct.counts, endpointID)
}
