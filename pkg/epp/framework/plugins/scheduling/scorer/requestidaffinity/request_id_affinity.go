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

package requestidaffinity

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// ScorerType is the type identifier for the RequestIDAffinityScorer.
	ScorerType = "request-id-affinity-scorer"

	// defaultHeader is the header read when no headerName is configured.
	defaultHeader = reqcommon.RequestIDHeaderKey
)

// parameters configures the RequestIDAffinityScorer.
type parameters struct {
	// HeaderName is the request header whose numeric suffix is used for bucket
	// assignment. Defaults to x-request-id when empty.
	HeaderName string `json:"headerName"`
	// Modulus fixes the divisor used in bucket = numericSuffix % modulus.
	// When 0 (the default), len(endpoints) is used so that every candidate
	// endpoint gets exactly one bucket.
	Modulus int `json:"modulus"`
}

// compile-time type assertion
var _ scheduling.Scorer = &RequestIDAffinityScorer{}

// Factory is the plugin factory function for RequestIDAffinityScorer.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", ScorerType, err)
		}
	}

	if params.Modulus < 0 {
		return nil, fmt.Errorf("'%s' scorer: modulus must be >= 0, got %d", ScorerType, params.Modulus)
	}

	return New(name, params.HeaderName, params.Modulus), nil
}

// New returns a RequestIDAffinityScorer. When headerName is empty the default
// x-request-id header is used. When modulus is 0, len(endpoints) is used at
// score time so every candidate endpoint gets exactly one bucket.
func New(name, headerName string, modulus int) *RequestIDAffinityScorer {
	h := strings.ToLower(strings.TrimSpace(headerName))
	if h == "" {
		h = defaultHeader
	}
	return &RequestIDAffinityScorer{
		typedName:  plugin.TypedName{Type: ScorerType, Name: name},
		headerName: h,
		modulus:    modulus,
	}
}

// RequestIDAffinityScorer routes requests to a deterministic bucket of
// endpoints derived from the numeric suffix of a request header value.
// Endpoints are sorted by name for stable bucket assignment, then
// bucket = numericSuffix % modulus is selected; that endpoint receives
// score 1.0 and all others receive 0.0.
type RequestIDAffinityScorer struct {
	typedName  plugin.TypedName
	headerName string
	// modulus is the divisor; 0 means use len(endpoints).
	modulus int
}

// TypedName returns the typed name of the plugin.
func (r *RequestIDAffinityScorer) TypedName() plugin.TypedName {
	return r.typedName
}

// Category returns Affinity: the scorer prefers a specific endpoint per request.
func (r *RequestIDAffinityScorer) Category() scheduling.ScorerCategory {
	return scheduling.Affinity
}

// Score gives score 1.0 to the endpoint in the bucket derived from the
// numeric suffix of the configured header, and 0.0 to all others.
// When the header is absent, non-numeric, or there are no endpoints, all
// endpoints receive score 0.0.
func (r *RequestIDAffinityScorer) Score(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	logger := log.FromContext(ctx)
	scores := make(map[scheduling.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		scores[ep] = 0.0
	}

	if len(endpoints) == 0 {
		return scores
	}

	headerVal := request.Headers[r.headerName]
	num, ok := parseNumericSuffix(headerVal)
	if !ok {
		logger.V(logutil.DEBUG).Info("request-id-affinity-scorer: no numeric suffix found, scoring all endpoints 0", "header", r.headerName, "value", headerVal)
		return scores
	}

	mod := r.modulus
	if mod == 0 {
		mod = len(endpoints)
	}

	bucket := num % mod

	// Sort endpoints by their namespaced name for a stable bucket assignment.
	sorted := make([]scheduling.Endpoint, len(endpoints))
	copy(sorted, endpoints)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].GetMetadata().NamespacedName.String() < sorted[j].GetMetadata().NamespacedName.String()
	})

	// Map bucket to an endpoint index; wrap around if bucket >= len(endpoints).
	idx := bucket % len(sorted)
	scores[sorted[idx]] = 1.0

	return scores
}

// parseNumericSuffix extracts the trailing integer from s.
// Examples: "req-17" -> 17, "abc123" -> 123, "42" -> 42, "abc" -> false.
func parseNumericSuffix(s string) (int, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	// Walk backwards to collect digit characters.
	end := len(s)
	start := end
	for start > 0 && s[start-1] >= '0' && s[start-1] <= '9' {
		start--
	}
	if start == end {
		return 0, false
	}
	n, err := strconv.Atoi(s[start:end])
	if err != nil {
		return 0, false
	}
	return n, true
}
