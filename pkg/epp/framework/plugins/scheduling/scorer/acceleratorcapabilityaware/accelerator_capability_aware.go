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

package acceleratorcapabilityaware

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

const (
	// Type is the type of the accelerator capability aware plugin.
	Type = "accelerator-capability-aware"

	// DefaultLabel is the default label name used to identify accelerator
	// capability ranges on pods.
	DefaultLabel = "llm-d.ai/accelerator-capability-range"
)

type parameters struct {
	// Label is the pod label name to check for accelerator capability range.
	// Format expected: "min-max" (e.g., "0-1024" or "1024-4096"), where min
	// and max are estimated request tokens.
	Label string `json:"label"`

	// EnableFiltering determines whether the plugin also filters pods that do
	// not match a request's estimated size. If false, the plugin only scores pods.
	EnableFiltering bool `json:"enableFiltering"`
}

type capabilityRange struct {
	min int
	max int
}

var _ scheduling.Filter = &Plugin{}
var _ scheduling.Scorer = &Plugin{}

// Factory defines the factory function for the accelerator capability aware plugin.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	parameters := &parameters{
		Label:           DefaultLabel,
		EnableFiltering: false,
	}

	if rawParameters != nil {
		if err := rawParameters.Decode(parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", Type, err)
		}
	}

	if parameters.Label == "" {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: 'label' must be specified", Type)
	}

	return New(name, parameters), nil
}

// New creates and returns an instance of the accelerator capability aware plugin.
func New(name string, params *parameters) *Plugin {
	return &Plugin{
		typedName:       plugin.TypedName{Type: Type, Name: name},
		labelName:       params.Label,
		enableFiltering: params.EnableFiltering,
	}
}

// Plugin filters and scores endpoints by matching an estimated request size
// against static accelerator capability ranges declared on endpoint labels.
//
// The request size is the token count from InferenceRequestBody.TokenizedPrompt,
// populated by the token-producer. For multimodal requests, the estimate backend
// can include image placeholder tokens, making this useful for heterogeneous MIG
// or accelerator-profile pools without runtime DCGM metrics.
type Plugin struct {
	typedName       plugin.TypedName
	labelName       string
	enableFiltering bool
}

// TypedName returns the typed name of the plugin.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Consumes declares the TokenizedPrompt dependency so token-producer runs before
// this plugin and can be auto-created when no explicit producer is configured.
func (p *Plugin) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{tokenproducer.TokenizedPromptDataKey: scheduling.TokenizedPrompt{}},
	}
}

// Filter filters out endpoints whose declared accelerator capability range does
// not contain the request size. Unlabeled endpoints pass through so deployments
// can adopt labels incrementally.
func (p *Plugin) Filter(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	if !p.enableFiltering {
		return endpoints
	}

	logger := log.FromContext(ctx).V(logging.DEBUG).WithName("AcceleratorCapabilityAware.Filter")
	requestSize := getRequestSize(request)
	logger.V(logging.TRACE).Info("Filtering endpoints by accelerator capability", "requestSize", requestSize)

	filteredEndpoints := []scheduling.Endpoint{}
	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			filteredEndpoints = append(filteredEndpoints, endpoint)
			continue
		}

		rangeStr, hasLabel := metadata.Labels[p.labelName]
		if !hasLabel {
			filteredEndpoints = append(filteredEndpoints, endpoint)
			continue
		}

		r, err := parseCapabilityRange(rangeStr)
		if err != nil {
			logger.Error(err, "Failed to parse accelerator capability range label", "endpoint", metadata.NamespacedName, "rangeStr", rangeStr)
			continue
		}

		if requestSize >= r.min && requestSize <= r.max {
			filteredEndpoints = append(filteredEndpoints, endpoint)
		}
	}

	logger.V(logging.TRACE).Info("Filtered endpoints", "originalCount", len(endpoints), "filteredCount", len(filteredEndpoints))
	return filteredEndpoints
}

// Score scores endpoints based on how well their accelerator capability range
// fits the estimated request size. Tighter matching ranges with remaining
// headroom score highest; unlabeled endpoints receive a neutral score.
func (p *Plugin) Score(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	logger := log.FromContext(ctx).V(logging.DEBUG).WithName("AcceleratorCapabilityAware.Score")
	requestSize := getRequestSize(request)
	logger.V(logging.TRACE).Info("Scoring endpoints by accelerator capability", "requestSize", requestSize)

	scoredEndpoints := make(map[scheduling.Endpoint]float64)
	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			scoredEndpoints[endpoint] = 0.5
			continue
		}

		rangeStr, hasLabel := metadata.Labels[p.labelName]
		if !hasLabel {
			scoredEndpoints[endpoint] = 0.5
			continue
		}

		r, err := parseCapabilityRange(rangeStr)
		if err != nil {
			logger.Error(err, "Failed to parse accelerator capability range label", "endpoint", metadata.NamespacedName, "rangeStr", rangeStr)
			scoredEndpoints[endpoint] = 0.0
			continue
		}

		scoredEndpoints[endpoint] = calculateRangeScore(requestSize, r)
	}

	logger.V(logging.TRACE).Info("Scored endpoints", "scores", scoredEndpoints)
	return scoredEndpoints
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (p *Plugin) Category() scheduling.ScorerCategory {
	return scheduling.Balance
}

func getRequestSize(request *scheduling.InferenceRequest) int {
	if request == nil || request.Body == nil || request.Body.TokenizedPrompt == nil {
		return 0
	}
	return request.Body.TokenizedPrompt.TokenCount()
}

func parseCapabilityRange(rangeStr string) (capabilityRange, error) {
	if rangeStr == "" {
		return capabilityRange{}, errors.New("empty range string")
	}

	bounds := strings.Split(rangeStr, "-")
	if len(bounds) != 2 {
		return capabilityRange{}, fmt.Errorf("invalid range format: %s (expected 'min-max')", rangeStr)
	}

	minVal, err := strconv.Atoi(strings.TrimSpace(bounds[0]))
	if err != nil {
		return capabilityRange{}, fmt.Errorf("invalid min value: %s", bounds[0])
	}

	maxVal, err := strconv.Atoi(strings.TrimSpace(bounds[1]))
	if err != nil {
		return capabilityRange{}, fmt.Errorf("invalid max value: %s", bounds[1])
	}

	if minVal > maxVal {
		return capabilityRange{}, fmt.Errorf("min (%d) cannot be greater than max (%d)", minVal, maxVal)
	}

	return capabilityRange{min: minVal, max: maxVal}, nil
}

func calculateRangeScore(requestSize int, r capabilityRange) float64 {
	const maxFallbackScore = 0.3

	if requestSize >= r.min && requestSize <= r.max {
		rangeWidth := r.max - r.min
		if rangeWidth == 0 {
			return 1.0
		}

		widthScore := 1.0 / (1.0 + float64(rangeWidth)/10000.0)
		headroom := float64(r.max - requestSize)
		positionScore := headroom / float64(rangeWidth)
		rawScore := 0.7*widthScore + 0.3*positionScore
		return maxFallbackScore + rawScore*(1.0-maxFallbackScore)
	}

	var distance int
	if requestSize > r.max {
		distance = requestSize - r.max
	} else {
		distance = r.min - requestSize
	}
	return maxFallbackScore / (1.0 + float64(distance)/1000.0)
}
