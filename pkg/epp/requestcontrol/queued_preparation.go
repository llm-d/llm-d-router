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

package requestcontrol

import (
	"context"
	"errors"
	"maps"
	"slices"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
)

func (d *Director) queuedPreparation(request *fwksched.InferenceRequest, metadata map[string]any) contracts.PrepareRequest {
	var baseline *fwksched.InferenceRequest
	metadata = maps.Clone(metadata)
	plugins := d.requestControlPlugins.dataProducerPlugins
	ownership := &preparationOwnership{running: map[plugin.TypedName]<-chan struct{}{}}
	dataKeys := map[plugin.DataKey]struct{}{}
	for _, p := range plugins {
		if _, ok := p.(fwkrh.AdmissionDataProducer); ok {
			for key, value := range p.Produces() {
				switch value.(type) {
				case attrconcurrency.InFlightLoad, *attrconcurrency.InFlightLoad:
					// Live load is supplied by the data layer, not request preparation.
					continue
				}
				dataKeys[key] = struct{}{}
			}
		}
		for key, value := range p.Produces() {
			switch value.(type) {
			case attrprefix.PrefixCacheMatchInfo, *attrprefix.PrefixCacheMatchInfo:
				dataKeys[key] = struct{}{}
			}
		}
	}
	keys := slices.Collect(maps.Keys(dataKeys))
	locate := func(ctx context.Context) []fwkdl.Endpoint { return d.endpointCandidates.Locate(ctx, metadata) }
	var admissionFilter contracts.AdmissionFilter
	if scheduler, ok := d.scheduler.(interface {
		FilterForAdmission(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint,
			func(string, []fwksched.Endpoint) []fwksched.Endpoint) (bool, map[string][]fwksched.Endpoint)
	}); ok {
		admissionFilter = scheduler.FilterForAdmission
	}
	return func(ctx context.Context, fits func(*contracts.PreparedRequest) bool, publish func(*contracts.PreparedRequest)) *contracts.PreparedRequest {
		failed := map[plugin.TypedName]error{}
		if baseline == nil {
			baseline = copyPreparationRequest(request)
		}
		request := copyPreparationRequest(baseline)
		located := locate(ctx)
		candidates := make(map[fwkdl.ID]*fwkdl.EndpointMetadata, len(located))
		for _, ep := range located {
			meta := ep.GetMetadata()
			candidates[meta.ID] = meta.Clone()
		}
		endpoints := d.toSchedulerEndpoints(located)
		endpoints = d.runScreeners(ctx, request, endpoints)
		prepared := &contracts.PreparedRequest{Request: request, Endpoints: endpoints, Candidates: candidates,
			Filter: admissionFilter, DataKeys: keys, Locate: locate, SealPreparation: ownership.seal}
		if len(endpoints) == 0 {
			message := "screeners eliminated all endpoint candidates"
			if len(located) == 0 {
				message = "failed to find endpoint candidates for serving the request"
			}
			prepared.Err = errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: message,
				Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonNoEndpoints)}}
			return prepared
		}
		run := func(p fwkrh.DataProducer, withoutPrefix bool) {
			if failed[p.TypedName()] != nil || ctx.Err() != nil {
				return
			}
			finished, ok := ownership.begin(ctx, p.TypedName())
			if !ok {
				failed[p.TypedName()] = errPreparationRunning
				return
			}
			nextRequest := copyPreparationRequest(request)
			nextEndpoints := copyPreparationEndpoints(endpoints)
			adapter := admissionProducer{DataProducer: p, withoutPrefix: withoutPrefix}
			err := dataProducerPluginsWithCompletion(ctx, fwkrh.AdmissionDataProducerExtensionPoint, producerTimeout(p),
				[]fwkrh.DataProducer{adapter}, nextRequest, nextEndpoints, finished)
			if err != nil {
				log.FromContext(ctx).Error(err, "failed to prepare admission data")
				failed[p.TypedName()] = err
				return
			}
			request, endpoints = nextRequest, nextEndpoints
		}
		// Tokenization is reusable across endpoint and cache changes.
		for _, p := range plugins {
			if _, ok := p.(fwkrh.AdmissionDataProducer); !ok {
				continue
			}
			if _, tokenProducer := p.Produces()[tokenizer.TokenizedPromptDataKey]; tokenProducer {
				run(p, false)
			}
		}
		if baseline.Body != nil && request.Body != nil {
			baseline.Body.TokenizedRequest = copyPreparationTokens(request.Body.TokenizedRequest)
		}
		// A missing cache estimate means no discount. Publish zero matches so a
		// prefix-based route decision can conservatively evaluate the full input.
		for _, p := range plugins {
			for key, value := range p.Produces() {
				switch value.(type) {
				case attrprefix.PrefixCacheMatchInfo, *attrprefix.PrefixCacheMatchInfo:
					for _, endpoint := range endpoints {
						endpoint.Put(key, attrprefix.NewPrefixCacheMatchInfo(0, 0, 1))
					}
				}
			}
		}
		warmRequest, warmEndpoints := copyPreparationRequest(request), copyPreparationEndpoints(endpoints)
		for _, p := range plugins {
			if _, ok := p.(fwkrh.AdmissionCostProducer); ok {
				run(p, true)
			}
		}
		cold := &contracts.PreparedRequest{Request: request, Endpoints: endpoints, Candidates: candidates,
			Filter: admissionFilter, DataKeys: keys, Locate: locate, ProducerErrors: maps.Clone(failed), SealPreparation: ownership.seal}
		if err := admissionCostError(plugins, failed); err != nil {
			cold.Err = err
			return cold
		}
		// Dispatch can take ownership as soon as the snapshot is published.
		fitsCold := fits(cold)
		publish(cold)
		if fitsCold || ctx.Err() != nil {
			return cold
		}
		request, endpoints = warmRequest, warmEndpoints
		for _, p := range plugins {
			if _, ok := p.(fwkrh.AdmissionDataProducer); !ok {
				continue
			}
			if _, tokenProducer := p.Produces()[tokenizer.TokenizedPromptDataKey]; !tokenProducer {
				run(p, false)
			}
		}
		if err := admissionCostError(plugins, failed); err != nil {
			result := *cold
			result.Err = err
			result.ProducerErrors = maps.Clone(failed)
			return &result
		}
		return &contracts.PreparedRequest{Request: request, Endpoints: endpoints, Candidates: candidates,
			WithoutPrefix: cold, Filter: admissionFilter, DataKeys: keys, Locate: locate, ProducerErrors: maps.Clone(failed), SealPreparation: ownership.seal}
	}
}

var errPreparationRunning = errors.New("admission preparation is still running")

type preparationOwnership struct {
	mu      sync.Mutex
	sealed  bool
	running map[plugin.TypedName]<-chan struct{}
}

func (o *preparationOwnership) begin(ctx context.Context, name plugin.TypedName) (chan struct{}, bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.sealed || ctx.Err() != nil {
		return nil, false
	}
	if previous, ok := o.running[name]; ok {
		select {
		case <-previous:
		default:
			return nil, false
		}
	}
	finished := make(chan struct{})
	o.running[name] = finished
	return finished, true
}

func (o *preparationOwnership) seal() map[plugin.TypedName]error {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.sealed = true
	failures := map[plugin.TypedName]error{}
	for name, finished := range o.running {
		select {
		case <-finished:
		default:
			failures[name] = errPreparationRunning
		}
	}
	return failures
}

func admissionCostError(producers []fwkrh.DataProducer, failures map[plugin.TypedName]error) error {
	for _, producer := range producers {
		if _, ok := producer.(fwkrh.AdmissionCostProducer); ok && failures[producer.TypedName()] != nil {
			return errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "failed to prepare projected request cost"}
		}
	}
	return nil
}

type admissionProducer struct {
	fwkrh.DataProducer
	withoutPrefix bool
}

func (p admissionProducer) Consumes() plugin.DataDependencies {
	if consumer, ok := p.DataProducer.(plugin.ConsumerPlugin); ok {
		return consumer.Consumes()
	}
	return plugin.DataDependencies{}
}

func (p admissionProducer) Produce(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	if p.withoutPrefix {
		return p.DataProducer.(fwkrh.AdmissionCostProducer).PrepareWithoutPrefix(ctx, request, endpoints)
	}
	return p.DataProducer.(fwkrh.AdmissionDataProducer).PrepareForAdmission(ctx, request, endpoints)
}

// Producer inputs are read-only except for attributes and tokenization output.
// Each invocation owns those outputs so a timeout cannot write into a snapshot
// subsequently read by scheduling or by a refresh.
func copyPreparationRequest(request *fwksched.InferenceRequest) *fwksched.InferenceRequest {
	result := &fwksched.InferenceRequest{
		RequestID:        request.RequestID,
		TargetModel:      request.TargetModel,
		Headers:          maps.Clone(request.Headers),
		Objectives:       request.Objectives,
		FairnessID:       request.FairnessID,
		RequestSizeBytes: request.RequestSizeBytes,
		SchedulingResult: request.SchedulingResult,
	}
	if request.Body != nil {
		body := *request.Body
		body.TokenizedRequest = copyPreparationTokens(body.TokenizedRequest)
		result.Body = &body
	}
	result.CopyAttributesFrom(request)
	return result
}

func copyPreparationTokens(source *fwkrequest.TokenizedRequest) *fwkrequest.TokenizedRequest {
	if source == nil {
		return nil
	}
	tokens := *source
	tokens.Prompts = slices.Clone(tokens.Prompts)
	for i := range tokens.Prompts {
		tokens.Prompts[i].TokenIDs = slices.Clone(tokens.Prompts[i].TokenIDs)
		tokens.Prompts[i].MultiModalFeatures = slices.Clone(tokens.Prompts[i].MultiModalFeatures)
	}
	return &tokens
}

func copyPreparationEndpoints(endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	result := make([]fwksched.Endpoint, len(endpoints))
	for i, endpoint := range endpoints {
		result[i] = fwksched.NewEndpoint(endpoint.GetMetadata(), endpoint.GetMetrics(), endpoint)
	}
	return result
}
