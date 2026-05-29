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

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

const (
	// Type is the type name used to register the multimodal encoder-cache scorer.
	Type = "mm-embeddings-cache-scorer"
)

var (
	_ scheduling.Scorer     = &Scorer{}
	_ plugin.ConsumerPlugin = &Scorer{}
)

// Config holds optional configuration for the scorer.
type Config struct {
	// ProducerName scopes the data key to a specific named producer instance.
	// Leave empty to consume from the default (unnamed) producer.
	ProducerName string `json:"producerName,omitempty"`
}

// Factory creates a multimodal encoder-cache affinity scorer.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var cfg Config
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for %q scorer: %w", Type, err)
		}
	}
	return New(name, cfg.ProducerName), nil
}

// Scorer computes normalized endpoint affinity from produced multimodal match data.
type Scorer struct {
	typedName      plugin.TypedName
	mmMatchDataKey plugin.DataKey
}

// New creates a Scorer. producerName scopes the data key to a specific producer
// instance; pass an empty string to use the default producer's key.
func New(name, producerName string) *Scorer {
	return &Scorer{
		typedName:      plugin.TypedName{Type: Type, Name: name},
		mmMatchDataKey: attrmm.EncoderCacheMatchInfoKey.WithNonEmptyProducerName(producerName),
	}
}

// TypedName returns the plugin type/name.
func (s *Scorer) TypedName() plugin.TypedName {
	return s.typedName
}

// Category returns the scorer category.
func (s *Scorer) Category() scheduling.ScorerCategory {
	return scheduling.Affinity
}

// Consumes returns the endpoint data consumed by this scorer.
func (s *Scorer) Consumes() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{s.mmMatchDataKey: attrmm.EncoderCacheMatchInfo{}}
}

// OptionalConsumes returns nil as this plugin has no optional data dependencies.
func (s *Scorer) OptionalConsumes() map[plugin.DataKey]any { return nil }

// Score scores endpoints by matched multimodal encoder-cache item size divided
// by total multimodal request item size.
func (s *Scorer) Score(ctx context.Context, req *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	traceLogger := log.FromContext(ctx).V(logging.TRACE)
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
		info, ok := endpoint.Get(s.mmMatchDataKey.String())
		if !ok {
			traceLogger.Info("mm-embeddings-cache: no match info, score 0", "requestID", requestID, "pod", pod, "scorer", s.typedName)
			continue
		}
		matchInfo, ok := info.(*attrmm.EncoderCacheMatchInfo)
		if !ok {
			traceLogger.Info("mm-embeddings-cache: invalid match info, score 0", "requestID", requestID, "pod", pod, "scorer", s.typedName)
			continue
		}
		totalWeight := itemWeight(matchInfo.RequestItems())
		if totalWeight <= 0 {
			traceLogger.Info("mm-embeddings-cache: invalid match info, score 0", "requestID", requestID, "pod", pod, "scorer", s.typedName)
			continue
		}
		matchedWeight := itemWeight(matchInfo.MatchedItems())
		score := float64(matchedWeight) / float64(totalWeight)
		scores[endpoint] = score
		traceLogger.Info("mm-embeddings-cache: pod score",
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
