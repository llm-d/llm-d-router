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
	"maps"
	"slices"

	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

func (d *Director) queuedPreparation(request *fwksched.InferenceRequest, metadata map[string]any) contracts.PrepareRequest {
	var baseline *fwksched.InferenceRequest
	metadata = maps.Clone(metadata)
	plugins := d.requestControlPlugins.dataProducerPlugins
	failed := map[plugin.TypedName]error{}
	dataKeys := map[plugin.DataKey]struct{}{}
	for _, p := range plugins {
		if _, ok := p.(fwkrc.AdmissionDataProducer); ok {
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
			Filter: admissionFilter, DataKeys: keys, Locate: locate}
		if len(endpoints) == 0 {
			message := "screeners eliminated all endpoint candidates"
			if len(located) == 0 {
				message = "failed to find endpoint candidates for serving the request"
			}
			prepared.Err = errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: message,
				Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonNoEndpoints)}}
			return prepared
		}
		run := func(p fwkrc.DataProducer, withoutPrefix bool) {
			if failed[p.TypedName()] != nil || ctx.Err() != nil {
				return
			}
			nextRequest := copyPreparationRequest(request)
			nextEndpoints := copyPreparationEndpoints(endpoints)
			adapter := admissionProducer{DataProducer: p, withoutPrefix: withoutPrefix}
			if err := dataProducerPluginsWithTimeout(ctx, producerTimeout(p), []fwkrc.DataProducer{adapter}, nextRequest, nextEndpoints, fwkrc.AdmissionDataProducerExtensionPoint); err != nil {
				log.FromContext(ctx).Error(err, "failed to prepare admission data")
				failed[p.TypedName()] = err
				return
			}
			request, endpoints = nextRequest, nextEndpoints
		}
		// Tokenization is reusable across endpoint and cache changes.
		for _, p := range plugins {
			if _, ok := p.(fwkrc.AdmissionDataProducer); !ok {
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
			if _, ok := p.(fwkrc.AdmissionCostProducer); ok {
				run(p, true)
			}
		}
		cold := &contracts.PreparedRequest{Request: request, Endpoints: endpoints, Candidates: candidates,
			Filter: admissionFilter, DataKeys: keys, Locate: locate, ProducerErrors: maps.Clone(failed)}
		if err := admissionCostError(plugins, failed); err != nil {
			cold.Err = err
			return cold
		}
		publish(cold)
		if fits(cold) || ctx.Err() != nil {
			return cold
		}
		request, endpoints = warmRequest, warmEndpoints
		for _, p := range plugins {
			if _, ok := p.(fwkrc.AdmissionDataProducer); !ok {
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
			WithoutPrefix: cold, Filter: admissionFilter, DataKeys: keys, Locate: locate, ProducerErrors: maps.Clone(failed)}
	}
}

func admissionCostError(producers []fwkrc.DataProducer, failures map[plugin.TypedName]error) error {
	for _, producer := range producers {
		if _, ok := producer.(fwkrc.AdmissionCostProducer); ok && failures[producer.TypedName()] != nil {
			return errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "failed to prepare projected request cost"}
		}
	}
	return nil
}

type admissionProducer struct {
	fwkrc.DataProducer
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
		return p.DataProducer.(fwkrc.AdmissionCostProducer).PrepareWithoutPrefix(ctx, request, endpoints)
	}
	return p.DataProducer.(fwkrc.AdmissionDataProducer).PrepareForAdmission(ctx, request, endpoints)
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
	for _, key := range request.AttributeKeys() {
		value, _ := request.GetAttribute(key)
		if cloneable, ok := value.(fwkdl.Cloneable); ok {
			value = cloneable.Clone()
		}
		result.PutAttribute(key, value)
	}
	return result
}

func copyPreparationTokens(source *fwkrh.TokenizedRequest) *fwkrh.TokenizedRequest {
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
