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
	SubsetFilterKey = "x-llm-d-router-destination-endpoint-subset"
	// OldSubsetFilterKey is the deprecated alias for SubsetFilterKey.
	OldSubsetFilterKey = "x-gateway-destination-endpoint-subset"
	// DestinationEndpointNamespace is the key for the outer namespace struct in the metadata field of the extproc response that is used to wrap the target endpoint.
	DestinationEndpointNamespace = "envoy.lb"
	// DestinationEndpointKey is the header and response metadata key used by Envoy to route to the appropriate pod.
	DestinationEndpointKey = "x-llm-d-router-destination-endpoint"
	// OldDestinationEndpointKey is the deprecated alias for DestinationEndpointKey.
	OldDestinationEndpointKey = "x-gateway-destination-endpoint"
	// DestinationEndpointServedKey is the metadata key used by Envoy to specify the endpoint that served the request.
	DestinationEndpointServedKey = "x-llm-d-router-destination-endpoint-served"
	// OldDestinationEndpointServedKey is the deprecated alias for DestinationEndpointServedKey.
	OldDestinationEndpointServedKey = "x-gateway-destination-endpoint-served"
	// FlowFairnessIDKey is the header key used to pass the fairness ID to be used in Flow Control.
	FlowFairnessIDKey = "x-llm-d-router-inference-fairness-id"
	// OldFlowFairnessIDKey is the deprecated alias for FlowFairnessIDKey.
	OldFlowFairnessIDKey = "x-gateway-inference-fairness-id"
	// ObjectiveKey is the header key used to specify the objective of an incoming request.
	ObjectiveKey = "x-llm-d-router-inference-objective"
	// OldObjectiveKey is the deprecated alias for ObjectiveKey.
	OldObjectiveKey = "x-gateway-inference-objective"
	// ModelNameRewriteKey is the header key used to specify the model name to be used when the request is forwarded to the model server.
	ModelNameRewriteKey = "x-llm-d-router-model-name-rewrite"
	// OldModelNameRewriteKey is the deprecated alias for ModelNameRewriteKey.
	OldModelNameRewriteKey = "x-gateway-model-name-rewrite"
	// TTFTSLOHeaderKey is the header key used to specify the time-to-first-token SLO in milliseconds.
	TTFTSLOHeaderKey = "x-llm-d-router-slo-ttft-ms"
	// OldTTFTSLOHeaderKey is the deprecated alias for TTFTSLOHeaderKey.
	OldTTFTSLOHeaderKey = "x-slo-ttft-ms"
	// TPOTSLOHeaderKey is the header key used to specify the time-per-output-token SLO in milliseconds.
	TPOTSLOHeaderKey = "x-llm-d-router-slo-tpot-ms"
	// OldTPOTSLOHeaderKey is the deprecated alias for TPOTSLOHeaderKey.
	OldTPOTSLOHeaderKey = "x-slo-tpot-ms"

	// DefaultFairnessID is the default fairness ID used when no ID is provided in the request.
	// This ensures that requests without explicit fairness identifiers are still grouped and managed by the Flow Control
	// system.
	DefaultFairnessID = "default-flow"
)

var headerAliases = map[string][]string{
	SubsetFilterKey:              {OldSubsetFilterKey},
	DestinationEndpointKey:       {OldDestinationEndpointKey},
	DestinationEndpointServedKey: {OldDestinationEndpointServedKey},
	FlowFairnessIDKey:            {OldFlowFairnessIDKey},
	ObjectiveKey:                 {OldObjectiveKey},
	ModelNameRewriteKey:          {OldModelNameRewriteKey},
	TTFTSLOHeaderKey:             {OldTTFTSLOHeaderKey},
	TPOTSLOHeaderKey:             {OldTPOTSLOHeaderKey},
}

var canonicalHeaderKeys = func() map[string]string {
	keys := make(map[string]string, len(headerAliases)*2)
	for canonical, aliases := range headerAliases {
		keys[strings.ToLower(canonical)] = canonical
		for _, alias := range aliases {
			keys[strings.ToLower(alias)] = canonical
		}
	}
	return keys
}()

// CanonicalHeaderKey maps deprecated EPP-managed header names to their current name.
func CanonicalHeaderKey(key string) string {
	if canonical, ok := canonicalHeaderKeys[strings.ToLower(key)]; ok {
		return canonical
	}
	return key
}

// HeaderNames returns the current header name followed by deprecated aliases.
func HeaderNames(key string) []string {
	canonical := CanonicalHeaderKey(key)
	names := []string{canonical}
	names = append(names, headerAliases[canonical]...)
	return names
}

// IsManagedHeader reports whether key is an EPP-managed header or deprecated alias.
func IsManagedHeader(key string) bool {
	_, ok := canonicalHeaderKeys[strings.ToLower(key)]
	return ok
}

// GetHeader returns a header value using the current name first, then deprecated aliases.
func GetHeader(headers map[string]string, key string) string {
	value, _ := GetHeaderValue(headers, key)
	return value
}

// GetHeaderValue returns a header value using the current name first, then deprecated aliases.
func GetHeaderValue(headers map[string]string, key string) (string, bool) {
	for _, name := range HeaderNames(key) {
		if value, ok := getStringValue(headers, name); ok {
			return value, true
		}
	}
	return "", false
}

// GetValue returns a metadata value using the current key first, then deprecated aliases.
func GetValue(values map[string]any, key string) (any, bool) {
	for _, name := range HeaderNames(key) {
		if value, ok := getAnyValue(values, name); ok {
			return value, true
		}
	}
	return nil, false
}

func getStringValue(values map[string]string, key string) (string, bool) {
	if value, ok := values[key]; ok {
		return value, true
	}
	lower := strings.ToLower(key)
	for candidate, value := range values {
		if strings.ToLower(candidate) == lower {
			return value, true
		}
	}
	return "", false
}

func getAnyValue(values map[string]any, key string) (any, bool) {
	if value, ok := values[key]; ok {
		return value, true
	}
	lower := strings.ToLower(key)
	for candidate, value := range values {
		if strings.ToLower(candidate) == lower {
			return value, true
		}
	}
	return nil, false
}
