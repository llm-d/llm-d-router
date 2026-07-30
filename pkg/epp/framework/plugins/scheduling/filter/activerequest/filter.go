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

package activerequest

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const (
	// ActiveRequestFilterType is the type of the ActiveRequestFilter.
	ActiveRequestFilterType = "active-request-filter"
)

// parameters defines the parameters for ActiveRequestFilter.
type parameters struct {
	// MaxRequests is the maximum number of in-flight requests an endpoint may
	// have and still pass the filter. Endpoints with a request count greater
	// than MaxRequests are dropped.
	MaxRequests int64 `json:"maxRequests"`

	// FallbackOnEmpty returns the unfiltered candidates when every endpoint
	// was filtered out, so the request can still be routed somewhere.
	FallbackOnEmpty bool `json:"fallbackOnEmpty"`

	InFlightLoadProducerName string `json:"inFlightLoadProducerName,omitempty"`
}

// compile-time type assertion
var _ scheduling.Filter = &ActiveRequestFilter{}
var _ plugin.ConsumerPlugin = &ActiveRequestFilter{}

// Factory defines the factory function for ActiveRequestFilter.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", ActiveRequestFilterType, err)
		}
	}

	return NewActiveRequestFilter(name, params)
}

// NewActiveRequestFilter validates the given parameters and returns a new
// ActiveRequestFilter with the given name.
func NewActiveRequestFilter(name string, params parameters) (*ActiveRequestFilter, error) {
	if name == "" {
		name = ActiveRequestFilterType
	}
	if params.MaxRequests <= 0 {
		return nil, fmt.Errorf("active request filter requires a positive maxRequests, got %d", params.MaxRequests)
	}

	return &ActiveRequestFilter{
		typedName:           plugin.TypedName{Type: ActiveRequestFilterType, Name: name},
		maxRequests:         params.MaxRequests,
		fallbackOnEmpty:     params.FallbackOnEmpty,
		inFlightLoadDataKey: attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(params.InFlightLoadProducerName),
	}, nil
}

// ActiveRequestFilter filters candidate endpoints by the in-flight request
// count produced by the inflight-load-producer data producer. Endpoints whose
// count exceeds the configured maximum are dropped. Endpoints without the
// attribute are treated as having zero in-flight requests and kept.
type ActiveRequestFilter struct {
	typedName           plugin.TypedName
	inFlightLoadDataKey plugin.DataKey

	// maxRequests is the maximum in-flight request count an endpoint may have
	// and still be kept.
	maxRequests int64
	// fallbackOnEmpty returns the unfiltered candidates when every endpoint
	// was filtered out.
	fallbackOnEmpty bool
}

// TypedName returns the typed name of the plugin.
func (f *ActiveRequestFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// Consumes returns the in-flight load attribute required for filtering.
func (f *ActiveRequestFilter) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{f.inFlightLoadDataKey: attrconcurrency.InFlightLoad{}},
	}
}

// Filter keeps the endpoints whose in-flight request count is at most
// maxRequests. When all endpoints are filtered out and fallbackOnEmpty is set,
// the original candidates are returned.
func (f *ActiveRequestFilter) Filter(ctx context.Context, _ *scheduling.InferenceRequest,
	endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	filtered := make([]scheduling.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if f.requestCount(ctx, endpoint) <= f.maxRequests {
			filtered = append(filtered, endpoint)
		}
	}

	log.FromContext(ctx).V(logutil.TRACE).Info("Filtered endpoints by active request count",
		"maxRequests", f.maxRequests, "candidates", len(endpoints), "kept", len(filtered))

	if len(filtered) == 0 && f.fallbackOnEmpty {
		return endpoints
	}
	return filtered
}

func (f *ActiveRequestFilter) requestCount(ctx context.Context, endpoint scheduling.Endpoint) int64 {
	val, ok := endpoint.Get(f.inFlightLoadDataKey.String())
	if !ok {
		return 0
	}

	switch load := val.(type) {
	case *attrconcurrency.InFlightLoad:
		if load == nil {
			log.FromContext(ctx).V(logutil.TRACE).Info("Ignoring nil in-flight load attribute",
				"endpoint", endpoint.GetMetadata().NamespacedName.String())
			return 0
		}
		return load.Requests
	default:
		log.FromContext(ctx).V(logutil.TRACE).Info("Ignoring in-flight load attribute with unexpected type",
			"endpoint", endpoint.GetMetadata().NamespacedName.String(),
			"attributeType", fmt.Sprintf("%T", val))
		return 0
	}
}
