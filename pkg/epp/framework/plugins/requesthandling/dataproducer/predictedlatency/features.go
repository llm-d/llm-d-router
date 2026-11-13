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

package predictedlatency

import (
	"fmt"
	"math"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
)

type featureScope uint8

const (
	requestFeature featureScope = iota
	endpointFeature
	metricFeature
	contextFeature
)

type featureUse uint8

const (
	predictionFeatures featureUse = 1 << iota
	ttftFeatures
	tpotFeatures
	allFeatureUses = predictionFeatures | ttftFeatures | tpotFeatures
)

type featureInput struct {
	producer *PredictedLatency
	state    *predictedLatencyCtx
	request  *fwksched.InferenceRequest
	endpoint fwksched.Endpoint
	metrics  *fwkdl.Metrics
	load     inFlightLoadSnapshot
	use      featureUse
}

type featureProvider struct {
	name         protoreflect.Name
	field        protoreflect.FieldDescriptor
	defaultValue protoreflect.Value
	scope        featureScope
	uses         featureUse
	enabled      func(*PredictedLatency) bool
	read         func(featureInput) protoreflect.Value
	dependency   func(*PredictedLatency) (plugin.DataKey, any)
}

var featureProviders = []featureProvider{
	{name: "kv_cache_percentage", scope: metricFeature, uses: allFeatureUses, read: func(in featureInput) protoreflect.Value {
		return protoreflect.ValueOfFloat64(in.metrics.KVCacheUsagePercent)
	}},
	{name: "input_token_length", scope: requestFeature, uses: allFeatureUses, read: func(in featureInput) protoreflect.Value {
		if in.request != nil && in.request.Body != nil && in.request.Body.TokenizedRequest != nil {
			return protoreflect.ValueOfInt64(int64(in.request.Body.TokenizedRequest.TokenCount()))
		}
		return protoreflect.ValueOfInt64(0)
	}, dependency: func(_ *PredictedLatency) (plugin.DataKey, any) {
		return tokenproducer.TokenizedPromptDataKey, fwkrequest.TokenizedRequest{}
	}},
	{name: "num_request_waiting", scope: metricFeature, uses: allFeatureUses, read: func(in featureInput) protoreflect.Value {
		return protoreflect.ValueOfInt64(int64(in.metrics.WaitingQueueSize))
	}},
	{name: "num_request_running", scope: endpointFeature, uses: allFeatureUses, read: func(in featureInput) protoreflect.Value { return protoreflect.ValueOfInt64(int64(in.load.requests)) }, dependency: inFlightDependency},
	{name: "num_tokens_generated", scope: contextFeature, uses: allFeatureUses, read: func(in featureInput) protoreflect.Value {
		if in.use == predictionFeatures {
			return protoreflect.ValueOfInt64(1)
		}
		return protoreflect.ValueOfInt64(0)
	}},
	{name: "prefix_cache_score", scope: endpointFeature, uses: predictionFeatures | ttftFeatures, read: func(in featureInput) protoreflect.Value {
		if raw, ok := in.endpoint.Get(in.producer.prefixMatchDataKey); ok {
			match := raw.(*attrprefix.PrefixCacheMatchInfo)
			score := float64(match.MatchBlocks()) / float64(match.TotalBlocks())
			if !math.IsNaN(score) {
				return protoreflect.ValueOfFloat64(score)
			}
		}
		return protoreflect.ValueOfFloat64(0)
	}, dependency: func(pl *PredictedLatency) (plugin.DataKey, any) {
		return pl.prefixMatchDataKey, attrprefix.PrefixCacheMatchInfo{}
	}},
	{name: "encoder_input_size", scope: endpointFeature, uses: predictionFeatures | ttftFeatures, enabled: encoderFeaturesEnabled, read: func(in featureInput) protoreflect.Value {
		input, _ := encoderSizes(in)
		return protoreflect.ValueOfInt64(int64(input))
	}, dependency: encoderDependency},
	{name: "encoder_matched_size", scope: endpointFeature, uses: predictionFeatures | ttftFeatures, enabled: encoderFeaturesEnabled, read: func(in featureInput) protoreflect.Value {
		_, matched := encoderSizes(in)
		return protoreflect.ValueOfInt64(int64(matched))
	}, dependency: encoderDependency},
	{name: "pod_type", scope: endpointFeature, uses: allFeatureUses, read: func(in featureInput) protoreflect.Value {
		if in.producer.config.EndpointRoleLabel == "" {
			return protoreflect.ValueOfString("")
		}
		return protoreflect.ValueOfString(in.endpoint.GetMetadata().Labels[in.producer.config.EndpointRoleLabel])
	}},
	{name: "prefill_tokens_in_flight", scope: endpointFeature, uses: allFeatureUses, read: func(in featureInput) protoreflect.Value { return protoreflect.ValueOfInt64(in.load.tokens) }, dependency: inFlightDependency},
	{name: "decode_tokens_in_flight", scope: endpointFeature, uses: allFeatureUses, read: func(_ featureInput) protoreflect.Value { return protoreflect.ValueOfInt64(0) }},
}

func init() {
	if err := validateFeatureProviders(featureProviders); err != nil {
		panic(err)
	}
	fields := (&protocol.Features{}).ProtoReflect().Descriptor().Fields()
	for index := range featureProviders {
		provider := &featureProviders[index]
		provider.field = fields.ByName(provider.name)
		value, err := protocol.DefaultFor(provider.field)
		if err != nil {
			panic(fmt.Errorf("default for %s: %w", provider.name, err))
		}
		provider.defaultValue = value
	}
}

func encoderFeaturesEnabled(pl *PredictedLatency) bool {
	return pl != nil && pl.config.UseEncoderCacheFeatures
}
func encoderDependency(pl *PredictedLatency) (plugin.DataKey, any) {
	return pl.encoderCacheDataKey, attrmm.EncoderCacheMatchInfo{}
}
func inFlightDependency(pl *PredictedLatency) (plugin.DataKey, any) {
	return pl.inFlightLoadDataKey, attrconcurrency.InFlightLoad{}
}

func encoderSizes(in featureInput) (int, int) {
	raw, ok := in.endpoint.Get(in.producer.encoderCacheDataKey)
	if !ok {
		return 0, 0
	}
	match, ok := raw.(*attrmm.EncoderCacheMatchInfo)
	if !ok || match == nil {
		return 0, 0
	}
	input := sumItemSizes(match.RequestItems())
	return input, min(input, sumItemSizes(match.MatchedItems()))
}

func validateFeatureProviders(providers []featureProvider) error {
	fields := (&protocol.Features{}).ProtoReflect().Descriptor().Fields()
	seen := make(map[protoreflect.Name]bool)
	for _, provider := range providers {
		if fields.ByName(provider.name) == nil {
			return fmt.Errorf("feature provider %s has no schema field", provider.name)
		}
		if seen[provider.name] {
			return fmt.Errorf("duplicate feature provider %s", provider.name)
		}
		if provider.read == nil || provider.uses == 0 || provider.scope > contextFeature {
			return fmt.Errorf("incomplete feature provider %s", provider.name)
		}
		seen[provider.name] = true
	}
	for index := range fields.Len() {
		if name := fields.Get(index).Name(); !seen[name] {
			return fmt.Errorf("schema feature %s has no provider", name)
		}
	}
	return nil
}

func captureFeatures(in featureInput, scope featureScope) *protocol.Features {
	features := &protocol.Features{}
	message := features.ProtoReflect()
	for _, provider := range featureProviders {
		if provider.scope != scope {
			continue
		}
		value := provider.defaultValue
		if provider.enabled == nil || provider.enabled(in.producer) {
			value = provider.read(in)
		}
		message.Set(provider.field, value)
	}
	return features
}

func (pl *PredictedLatency) captureEndpointFeatures(endpoint fwksched.Endpoint) *protocol.Features {
	return captureFeatures(featureInput{producer: pl, endpoint: endpoint, load: pl.readInFlightLoad(endpoint)}, endpointFeature)
}

func endpointFeaturesFor(state *predictedLatencyCtx, metadata *fwkdl.EndpointMetadata) *protocol.Features {
	if metadata == nil {
		return &protocol.Features{}
	}
	if features := state.endpointFeatures[metadata.ID.String()]; features != nil {
		return features
	}
	return &protocol.Features{}
}

func composeFeatures(state *predictedLatencyCtx, metrics *fwkdl.Metrics, endpoint *protocol.Features, use featureUse) *protocol.Features {
	features := &protocol.Features{}
	if state.requestFeatures != nil {
		proto.Merge(features, state.requestFeatures)
	}
	if endpoint != nil {
		proto.Merge(features, endpoint)
	}
	message := features.ProtoReflect()
	input := featureInput{state: state, metrics: metrics, use: use}
	for _, provider := range featureProviders {
		if provider.uses&use == 0 {
			message.Set(provider.field, provider.defaultValue)
		} else if provider.scope == metricFeature || provider.scope == contextFeature {
			message.Set(provider.field, provider.read(input))
		}
	}
	return features
}
