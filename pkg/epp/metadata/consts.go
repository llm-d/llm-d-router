/*
Copyright 2025 The Kubernetes Authors.

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

package metadata

import "strings"

const (
	// SubsetFilterNamespace is the key for the outer namespace struct in the metadata field of the extproc request that is used to wrap the subset filter.
	SubsetFilterNamespace = "envoy.lb.subset_hint"
	// SubsetFilterKey is the metadata key used by Envoy to specify an array candidate pods for serving the request.
	// If not specified, all the pods that are associated with the pool are candidates.
	SubsetFilterKey = "x-gateway-destination-endpoint-subset"
	// DestinationEndpointNamespace is the key for the outer namespace struct in the metadata field of the extproc response that is used to wrap the target endpoint.
	DestinationEndpointNamespace = "envoy.lb"
	// DestinationEndpointKey is the header and response metadata key used by Envoy to route to the appropriate pod.
	DestinationEndpointKey = "x-gateway-destination-endpoint"
	// DestinationEndpointScoresKey is the response metadata key holding a struct that maps each endpoint listed in
	// DestinationEndpointKey to the scheduler's score for it. Emitted only when the EPP is started with
	// --emit-endpoint-scores and the picker surfaced scores.
	DestinationEndpointScoresKey = "x-gateway-destination-endpoint-scores"
	// DestinationEndpointServedKey is the metadata key used by Envoy to specify the endpoint that served the request.
	DestinationEndpointServedKey = "x-gateway-destination-endpoint-served"
	// FlowFairnessIDKey is the header key used to pass the fairness ID to be used in Flow Control.
	FlowFairnessIDKey = "x-llm-d-inference-fairness-id"
	// OldFlowFairnessIDKey is the deprecated alias for FlowFairnessIDKey.
	OldFlowFairnessIDKey = "x-gateway-inference-fairness-id"
	// ObjectiveKey is the header key used to specify the objective of an incoming request.
	ObjectiveKey = "x-llm-d-inference-objective"
	// OldObjectiveKey is the deprecated alias for ObjectiveKey.
	OldObjectiveKey = "x-gateway-inference-objective"
	// ModelNameRewriteKey is the header key used to specify the model name to be used when the request is forwarded to the model server.
	ModelNameRewriteKey = "x-llm-d-model-name-rewrite"
	// OldModelNameRewriteKey is the deprecated alias for ModelNameRewriteKey.
	OldModelNameRewriteKey = "x-gateway-model-name-rewrite"
	// TTFTSLOHeaderKey is the header key used to specify the time-to-first-token SLO in milliseconds.
	TTFTSLOHeaderKey = "x-llm-d-slo-ttft-ms"
	// OldTTFTSLOHeaderKey is the deprecated alias for TTFTSLOHeaderKey.
	OldTTFTSLOHeaderKey = "x-slo-ttft-ms"
	// TPOTSLOHeaderKey is the header key used to specify the time-per-output-token SLO in milliseconds.
	TPOTSLOHeaderKey = "x-llm-d-slo-tpot-ms"
	// OldTPOTSLOHeaderKey is the deprecated alias for TPOTSLOHeaderKey.
	OldTPOTSLOHeaderKey = "x-slo-tpot-ms"
	// VideoFPSHeaderKey is the header key used to specify a request's source video frame rate (frames per second).
	VideoFPSHeaderKey = "x-llm-d-video-fps"
	// VideoDurationHeaderKey is the header key used to specify a request's video length in seconds.
	VideoDurationHeaderKey = "x-llm-d-video-duration-seconds"
	// VideoResolutionHeaderKey is the header key used to specify a request's video frame resolution as "WIDTHxHEIGHT".
	VideoResolutionHeaderKey = "x-llm-d-video-resolution"

	// DefaultFairnessID is the default fairness ID used when no ID is provided in the request.
	// This ensures that requests without explicit fairness identifiers are still grouped and managed by the Flow Control
	// system.
	DefaultFairnessID = "default-flow"
)

// All headerAliases keys and values must be lower case.
var headerAliases = map[string][]string{
	FlowFairnessIDKey:   {FlowFairnessIDKey, OldFlowFairnessIDKey},
	ObjectiveKey:        {ObjectiveKey, OldObjectiveKey},
	ModelNameRewriteKey: {ModelNameRewriteKey, OldModelNameRewriteKey},
	TTFTSLOHeaderKey:    {TTFTSLOHeaderKey, OldTTFTSLOHeaderKey},
	TPOTSLOHeaderKey:    {TPOTSLOHeaderKey, OldTPOTSLOHeaderKey},
}

// HeaderNames returns the current header name followed by deprecated aliases.
func HeaderNames(key string) []string {
	key = strings.ToLower(key)
	if names, ok := headerAliases[key]; ok {
		return names
	}
	return []string{key}
}

// GetLowerCaseHeaderValue returns a header value from a lower-case header map using the current name first, then deprecated aliases.
func GetLowerCaseHeaderValue(headers map[string]string, key string) (string, bool) {
	key = strings.ToLower(key)
	if aliases, ok := headerAliases[key]; ok {
		for _, alias := range aliases {
			if value, ok := headers[alias]; ok {
				return value, true
			}
		}
		return "", false
	}
	value, ok := headers[key]
	return value, ok
}
