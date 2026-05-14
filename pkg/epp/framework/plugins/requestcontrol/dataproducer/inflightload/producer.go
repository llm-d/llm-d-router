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
	sourcenotifications "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/source/notifications"
)

const (
	InFlightLoadProducerType = "inflight-load-producer"
	profilePrefill           = "prefill"
)

func InFlightLoadProducerFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return &InFlightLoadProducer{
		typedName:      fwkplugin.TypedName{Type: InFlightLoadProducerType, Name: name},
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
		tokenEstimator: NewSimpleTokenEstimator(),
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
	typedName      fwkplugin.TypedName
	requestTracker *concurrencyTracker
	tokenTracker   *concurrencyTracker
	tokenEstimator TokenEstimator
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

func (p *InFlightLoadProducer) Produce(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
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

	for _, profileResult := range result.ProfileResults {
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
		tokens := p.tokenEstimator.Estimate(request)
		p.tokenTracker.add(eid, tokens)
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

	// 1. Early Prefill Release (on first chunk)
	// Uses the new StartOfStream signal provided by the framework.
	if resp.StartOfStream {
		if prefillResult, ok := result.ProfileResults[profilePrefill]; ok && len(prefillResult.TargetEndpoints) > 0 {
			p.release(prefillResult.TargetEndpoints[0], request)
		}
	}

	// 2. Full Cleanup (on completion)
	if resp.EndOfStream {
		for name, profileResult := range result.ProfileResults {
			if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
				continue
			}
			// Skip "prefill" as it was already released in the StartOfStream block.
			// This works perfectly even if StartOfStream and EndOfStream are both true (single chunk).
			if name == profilePrefill {
				continue
			}
			p.release(profileResult.TargetEndpoints[0], request)
		}
	}
}

func (p *InFlightLoadProducer) release(endpoint fwksched.Endpoint, request *fwksched.InferenceRequest) {
	if endpoint == nil || endpoint.GetMetadata() == nil {
		return
	}
	eid := endpoint.GetMetadata().NamespacedName.String()
	p.requestTracker.dec(eid)
	tokens := p.tokenEstimator.Estimate(request)
	p.tokenTracker.add(eid, -tokens)
}

func (p *InFlightLoadProducer) Produces() map[string]any {
	return map[string]any{
		attrconcurrency.InFlightLoadKey: attrconcurrency.InFlightLoad{},
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
