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

// Package mmcacheaffinity scores endpoints from multimodal encoder-cache match
// info produced by the request-control multimodal data producer.
package mmcacheaffinity

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrmm "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

const (
	// Type is the type name used to register the multimodal encoder-cache scorer.
	Type = "mm-cache-affinity-scorer"
)

var (
	_ scheduling.Scorer     = &Scorer{}
	_ plugin.ConsumerPlugin = &Scorer{}
)

// Factory creates a multimodal encoder-cache affinity scorer.
func Factory(name string, rawParameters json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	if len(rawParameters) > 0 {
		var parameters map[string]any
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", Type, err)
		}
	}

	return New().WithName(name), nil
}

// Scorer computes normalized endpoint affinity from produced multimodal match data.
type Scorer struct {
	typedName plugin.TypedName
}

// New creates a Scorer.
func New() *Scorer {
	return &Scorer{typedName: plugin.TypedName{Type: Type}}
}

// TypedName returns the plugin type/name.
func (s *Scorer) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the plugin instance name.
func (s *Scorer) WithName(name string) *Scorer {
	s.typedName.Name = name
	return s
}

// Category returns the scorer category.
func (s *Scorer) Category() scheduling.ScorerCategory {
	return scheduling.Affinity
}

// Consumes returns the endpoint data consumed by this scorer.
func (s *Scorer) Consumes() map[string]any {
	return map[string]any{attrmm.EncoderCacheMatchInfoKey: attrmm.EncoderCacheMatchInfo{}}
}

// Score scores endpoints by matched multimodal encoder-cache item size divided
// by total multimodal request item size.
func (s *Scorer) Score(ctx context.Context, _ *scheduling.CycleState, req *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	logger := log.FromContext(ctx).V(logging.DEBUG)
	requestID := ""
	if req != nil {
		requestID = req.RequestID
	}
	scores := make(map[scheduling.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scores[endpoint] = 0
		pod := ""
		if meta := endpoint.GetMetadata(); meta != nil {
			pod = meta.PodName
		}
		info, ok := endpoint.Get(attrmm.EncoderCacheMatchInfoKey)
		if !ok {
			logger.Info("mm-cache-affinity: no match info, score 0", "requestID", requestID, "pod", pod, "scorer", s.typedName)
			continue
		}
		matchInfo, ok := info.(*attrmm.EncoderCacheMatchInfo)
		if !ok {
			logger.Info("mm-cache-affinity: invalid match info, score 0", "requestID", requestID, "pod", pod, "scorer", s.typedName)
			continue
		}
		totalWeight := itemWeight(matchInfo.RequestItems())
		if totalWeight <= 0 {
			logger.Info("mm-cache-affinity: invalid match info, score 0", "requestID", requestID, "pod", pod, "scorer", s.typedName)
			continue
		}
		matchedWeight := itemWeight(matchInfo.MatchedItems())
		score := float64(matchedWeight) / float64(totalWeight)
		scores[endpoint] = score
		logger.Info("mm-cache-affinity: pod score",
			"requestID", requestID,
			"pod", pod,
			"matchedWeight", matchedWeight,
			"totalWeight", totalWeight,
			"affinityScore", score,
			"scorer", s.typedName)
	}
	return scores
}

func itemWeight(items []attrmm.MatchItem) int {
	weight := 0
	for _, item := range items {
		weight += item.Size
	}
	return weight
}
