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

package loader

import (
	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	configapiv1alpha1 "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
)

// convertV1alpha1ToV1 maps a v1alpha1 configuration onto the v1 types the EPP reads internally.
// The two differ only under dataLayer: the crossReplica* fields group into a subsection
// and the bare discovery.pluginRef is folded into discovery.endpoints. Every other field is a direct copy.
func convertV1alpha1ToV1(logger logr.Logger, in *configapiv1alpha1.EndpointPickerConfig) *configapiv1.EndpointPickerConfig {
	if in == nil {
		return nil
	}

	out := &configapiv1.EndpointPickerConfig{
		TypeMeta: metav1.TypeMeta{
			APIVersion: configapiv1.GroupVersion.String(),
			Kind:       in.Kind,
		},
		FeatureGates:       configapiv1.FeatureGates(in.FeatureGates),
		Plugins:            convertPluginSpecs(in.Plugins),
		SchedulingProfiles: convertSchedulingProfiles(in.SchedulingProfiles),
		DataLayer:          convertDataLayer(logger, in.DataLayer),
		FlowControl:        convertFlowControl(in.FlowControl),
		RequestHandler:     convertRequestHandler(in.RequestHandler),
	}
	return out
}

func convertPluginSpecs(in []configapiv1alpha1.PluginSpec) []configapiv1.PluginSpec {
	if in == nil {
		return nil
	}
	out := make([]configapiv1.PluginSpec, len(in))
	for i, spec := range in {
		out[i] = configapiv1.PluginSpec{
			Name:       spec.Name,
			Type:       spec.Type,
			Parameters: spec.Parameters,
		}
	}
	return out
}

func convertSchedulingProfiles(in []configapiv1alpha1.SchedulingProfile) []configapiv1.SchedulingProfile {
	if in == nil {
		return nil
	}
	out := make([]configapiv1.SchedulingProfile, len(in))
	for i, profile := range in {
		out[i] = configapiv1.SchedulingProfile{Name: profile.Name}
		if profile.Plugins != nil {
			out[i].Plugins = make([]configapiv1.SchedulingPlugin, len(profile.Plugins))
			for j, p := range profile.Plugins {
				out[i].Plugins[j] = configapiv1.SchedulingPlugin{
					PluginRef: p.PluginRef,
					Weight:    copyFloat64(p.Weight),
				}
			}
		}
	}
	return out
}

func convertDataLayer(logger logr.Logger, in *configapiv1alpha1.DataLayerConfig) *configapiv1.DataLayerConfig {
	if in == nil {
		return nil
	}

	out := &configapiv1.DataLayerConfig{
		InjectDefaults: copyBool(in.InjectDefaults),
		Discovery:      convertDiscovery(logger, in.Discovery),
	}

	if in.Sources != nil {
		out.Sources = make([]configapiv1.DataLayerSource, len(in.Sources))
		for i, source := range in.Sources {
			out.Sources[i] = configapiv1.DataLayerSource{PluginRef: source.PluginRef}
			if source.Extractors != nil {
				out.Sources[i].Extractors = make([]configapiv1.DataLayerExtractor, len(source.Extractors))
				for j, extractor := range source.Extractors {
					out.Sources[i].Extractors[j] = configapiv1.DataLayerExtractor{PluginRef: extractor.PluginRef}
				}
			}
		}
	}

	if in.CrossReplicaSyncerPluginRef != "" || in.CrossReplicaSyncInterval != nil || in.CrossReplicaPublishTimeout != nil {
		out.CrossReplica = &configapiv1.CrossReplicaConfig{
			SyncerPluginRef: in.CrossReplicaSyncerPluginRef,
			SyncInterval:    copyDuration(in.CrossReplicaSyncInterval),
			PublishTimeout:  copyDuration(in.CrossReplicaPublishTimeout),
		}
	}

	return out
}

func convertDiscovery(logger logr.Logger, in *configapiv1alpha1.DiscoveryConfig) *configapiv1.DiscoveryConfig {
	if in == nil {
		return nil
	}

	out := &configapiv1.DiscoveryConfig{}
	if in.Endpoints != nil {
		out.Endpoints = &configapiv1.EndpointDiscoveryConfig{PluginRef: in.Endpoints.PluginRef}
	}
	if in.Peers != nil {
		out.Peers = &configapiv1.PeerDiscoveryConfig{PluginRef: in.Peers.PluginRef}
	}

	//nolint:staticcheck // SA1019: reading the deprecated field is the point of the conversion.
	if in.PluginRef != "" {
		logger.Info("DEPRECATION: dataLayer.discovery.pluginRef is deprecated, use dataLayer.discovery.endpoints.pluginRef instead. If both are set, the new field is used.")
		if out.Endpoints == nil {
			//nolint:staticcheck // SA1019: reading the deprecated field is the point of the conversion.
			out.Endpoints = &configapiv1.EndpointDiscoveryConfig{PluginRef: in.PluginRef}
		}
	}

	return out
}

func convertFlowControl(in *configapiv1alpha1.FlowControlConfig) *configapiv1.FlowControlConfig {
	if in == nil {
		return nil
	}

	out := &configapiv1.FlowControlConfig{
		MaxBytes:                    copyQuantity(in.MaxBytes),
		MaxRequests:                 copyQuantity(in.MaxRequests),
		DefaultRequestTTL:           copyDuration(in.DefaultRequestTTL),
		NoEndpointRequestTTL:        copyDuration(in.NoEndpointRequestTTL),
		DefaultPriorityBand:         convertPriorityBandPtr(in.DefaultPriorityBand),
		DefaultNegativePriorityBand: convertPriorityBandPtr(in.DefaultNegativePriorityBand),
		UsageLimitPolicyPluginRef:   in.UsageLimitPolicyPluginRef,
		EnableEviction:              in.EnableEviction,
	}

	if in.PriorityBands != nil {
		out.PriorityBands = make([]configapiv1.PriorityBandConfig, len(in.PriorityBands))
		for i := range in.PriorityBands {
			out.PriorityBands[i] = convertPriorityBand(in.PriorityBands[i])
		}
	}

	if in.SaturationDetector != nil {
		out.SaturationDetector = &configapiv1.SaturationDetectorConfig{PluginRef: in.SaturationDetector.PluginRef}
	}

	return out
}

func convertPriorityBandPtr(in *configapiv1alpha1.PriorityBandConfig) *configapiv1.PriorityBandConfig {
	if in == nil {
		return nil
	}
	out := convertPriorityBand(*in)
	return &out
}

func convertPriorityBand(in configapiv1alpha1.PriorityBandConfig) configapiv1.PriorityBandConfig {
	return configapiv1.PriorityBandConfig{
		Priority:          in.Priority,
		MaxBytes:          copyQuantity(in.MaxBytes),
		MaxRequests:       copyQuantity(in.MaxRequests),
		DefaultRequestTTL: copyDuration(in.DefaultRequestTTL),
		FairnessPolicyRef: in.FairnessPolicyRef,
		OrderingPolicyRef: in.OrderingPolicyRef,
	}
}

func convertRequestHandler(in *configapiv1alpha1.RequestHandlerConfig) *configapiv1.RequestHandlerConfig {
	if in == nil {
		return nil
	}

	out := &configapiv1.RequestHandlerConfig{PropagatePriority: in.PropagatePriority}
	if in.Parsers != nil {
		out.Parsers = make([]configapiv1.ParserConfig, len(in.Parsers))
		for i, parser := range in.Parsers {
			out.Parsers[i] = configapiv1.ParserConfig{PluginRef: parser.PluginRef}
		}
	}
	return out
}

func copyQuantity(in *resource.Quantity) *resource.Quantity {
	if in == nil {
		return nil
	}
	out := in.DeepCopy()
	return &out
}

func copyDuration(in *metav1.Duration) *metav1.Duration {
	if in == nil {
		return nil
	}
	out := *in
	return &out
}

func copyBool(in *bool) *bool {
	if in == nil {
		return nil
	}
	out := *in
	return &out
}

func copyFloat64(in *float64) *float64 {
	if in == nil {
		return nil
	}
	out := *in
	return &out
}
