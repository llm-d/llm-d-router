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

package tokenload

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const (
	TokenLoadScorerType        = "token-load-scorer"
	tokenQueueThresholdDefault = 4194304 // 128 requests @ 32K per request
)

// Config holds the configuration for the TokenLoadScorer.
type Config struct {
	// QueueThresholdTokens defines the maximum number of in-flight tokens used for scoring normalization.
	// Defaults to 4194304 if unset.
	QueueThresholdTokens     int64  `json:"queueThresholdTokens"`
	InFlightLoadProducerName string `json:"inFlightLoadProducerName,omitempty"`
}

// compile-time type assertion
var _ fwksched.Scorer = &TokenLoadScorer{}

type TokenLoadScorer struct {
	typedName                    fwkplugin.TypedName
	queueThresholdTokens         float64
	inFlightLoadDataKey          fwkplugin.DataKey
	uncachedRequestTokensDataKey fwkplugin.DataKey
}

func TokenLoadScorerFactory(name string, params *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := Config{
		QueueThresholdTokens: tokenQueueThresholdDefault,
	}
	if params != nil {
		if err := params.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal token load scorer config: %w", err)
		}
	}
	if cfg.QueueThresholdTokens <= 0 {
		cfg.QueueThresholdTokens = tokenQueueThresholdDefault
	}

	return &TokenLoadScorer{
		typedName:                    fwkplugin.TypedName{Type: TokenLoadScorerType, Name: name},
		queueThresholdTokens:         float64(cfg.QueueThresholdTokens),
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(cfg.InFlightLoadProducerName),
		uncachedRequestTokensDataKey: attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(cfg.InFlightLoadProducerName),
	}, nil
}

func (s *TokenLoadScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *TokenLoadScorer) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

func (s *TokenLoadScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			s.inFlightLoadDataKey:          attrconcurrency.InFlightLoad{},
			s.uncachedRequestTokensDataKey: attrconcurrency.UncachedRequestTokens{},
		},
	}
}

func (s *TokenLoadScorer) Score(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	logger := log.FromContext(ctx)

	for _, endpoint := range endpoints {
		endpointID := endpoint.GetMetadata().NamespacedName.String()
		tokenLoad := 0.0

		// Read both accumulated in-flight load and the projected impact of the
		// request being scored, which are now carried on separate attributes.
		var tokens int64
		if val, ok := endpoint.Get(s.inFlightLoadDataKey.String()); ok {
			if load, ok := val.(*attrconcurrency.InFlightLoad); ok && load != nil {
				tokens += load.Tokens
			}
		}
		if val, ok := endpoint.Get(s.uncachedRequestTokensDataKey.String()); ok {
			if uncached, ok := val.(*attrconcurrency.UncachedRequestTokens); ok && uncached != nil {
				tokens += uncached.Tokens
			}
		}
		tokenLoad = float64(tokens)

		score := 0.0
		if tokenLoad <= 0 {
			score = 1.0
		} else {
			if tokenLoad > s.queueThresholdTokens {
				tokenLoad = s.queueThresholdTokens
			}
			score = 1.0 - (tokenLoad / s.queueThresholdTokens)
		}
		scores[endpoint] = score
		logger.V(logutil.DEBUG).Info("TokenLoadScorer scoring", "endpoint", endpointID, "tokenLoad", tokenLoad, "score", score)
	}

	return scores
}
