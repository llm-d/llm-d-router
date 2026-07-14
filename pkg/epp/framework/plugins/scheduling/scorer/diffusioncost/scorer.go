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

// Package diffusioncost scores endpoints by the outstanding declared cost of
// their in-flight image generation requests, produced by the
// diffusion-load-producer. Where active-request scoring treats every request
// as equal work, this scorer weighs requests by declared diffusion cost
// (inference steps x resolution x image count), so two queued low-step
// thumbnails do not count the same as one queued high-step full-size render.
package diffusioncost

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrdiffusion "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/diffusion"
)

const (
	// DiffusionCostScorerType is the type of the DiffusionCost scorer.
	DiffusionCostScorerType = "diffusion-cost-scorer"
)

// Parameters defines the parameters for the DiffusionCost scorer.
type Parameters struct {
	// DiffusionLoadProducerName selects which diffusion-load-producer
	// instance's DiffusionLoad attribute to read. Empty defaults to the
	// producer registered under its type name.
	DiffusionLoadProducerName string `json:"diffusionLoadProducerName,omitempty"`
}

// compile-time type assertions
var (
	_ scheduling.Scorer     = &DiffusionCost{}
	_ plugin.ConsumerPlugin = &DiffusionCost{}
)

// Factory defines the factory function for the DiffusionCost scorer.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	parameters := Parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", DiffusionCostScorerType, err)
		}
	}

	return NewDiffusionCost(&parameters).WithName(name), nil
}

// NewDiffusionCost creates a new DiffusionCost scorer.
func NewDiffusionCost(params *Parameters) *DiffusionCost {
	var producerName string
	if params != nil {
		producerName = params.DiffusionLoadProducerName
	}

	return &DiffusionCost{
		typedName:            plugin.TypedName{Type: DiffusionCostScorerType},
		diffusionLoadDataKey: attrdiffusion.DiffusionLoadDataKey.WithNonEmptyProducerName(producerName),
	}
}

// DiffusionCost scores endpoints based on the outstanding declared diffusion
// cost produced by the diffusion-load-producer.
type DiffusionCost struct {
	typedName            plugin.TypedName
	diffusionLoadDataKey plugin.DataKey
}

// TypedName returns the typed name of the plugin.
func (s *DiffusionCost) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *DiffusionCost) WithName(name string) *DiffusionCost {
	s.typedName.Name = name
	return s
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *DiffusionCost) Category() scheduling.ScorerCategory {
	return scheduling.Distribution
}

// Consumes returns the diffusion load attribute required for scoring.
func (s *DiffusionCost) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{s.diffusionLoadDataKey: attrdiffusion.DiffusionLoad{}},
	}
}

// Score scores the given endpoints by outstanding declared cost, normalized
// to 0-1. Endpoints with no outstanding cost get the maximum score; the most
// loaded endpoint gets 0.
func (s *DiffusionCost) Score(ctx context.Context, _ *scheduling.InferenceRequest,
	endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	costs := make(map[scheduling.Endpoint]int64, len(endpoints))
	logCosts := make(map[string]int64, len(endpoints))
	maxCost := int64(0)

	for _, endpoint := range endpoints {
		cost := s.costUnits(ctx, endpoint)
		costs[endpoint] = cost
		logCosts[endpoint.GetMetadata().NamespacedName.String()] = cost
		if cost > maxCost {
			maxCost = cost
		}
	}

	// TODO: change the log level before submission
	log.FromContext(ctx).V(logutil.DEFAULT).Info("Diffusion cost units", "endpointCosts", logCosts, "maxCost", maxCost)

	scoredEndpointsMap := make(map[scheduling.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		cost := costs[endpoint]
		if cost == 0 {
			scoredEndpointsMap[endpoint] = 1.0
			continue
		}
		scoredEndpointsMap[endpoint] = float64(maxCost-cost) / float64(maxCost)
	}

	return scoredEndpointsMap
}

func (s *DiffusionCost) costUnits(ctx context.Context, endpoint scheduling.Endpoint) int64 {
	val, ok := endpoint.Get(s.diffusionLoadDataKey.String())
	if !ok {
		return 0
	}

	load, ok := val.(*attrdiffusion.DiffusionLoad)
	if !ok || load == nil {
		log.FromContext(ctx).V(logutil.TRACE).Info("Ignoring diffusion load attribute with unexpected type or nil value",
			"endpoint", endpoint.GetMetadata().NamespacedName.String(),
			"attributeType", fmt.Sprintf("%T", val))
		return 0
	}
	return load.CostUnits
}
