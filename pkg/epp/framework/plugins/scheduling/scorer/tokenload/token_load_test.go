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
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

func TestTokenLoadScorer(t *testing.T) {
	threshold := 1000.0

	scorer := &TokenLoadScorer{
		typedName:                           fwkplugin.TypedName{Type: TokenLoadScorerType, Name: TokenLoadScorerType},
		queueThresholdTokens:                threshold,
		inFlightLoadDataKey:                 attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(""),
		currentRequestEndpointImpactDataKey: attrconcurrency.CurrentRequestEndpointImpactDataKey.WithNonEmptyProducerName(""),
	}

	t.Run("in-flight load only", func(t *testing.T) {
		pod1NN := types.NamespacedName{Namespace: "default", Name: "pod1"}
		pod2NN := types.NamespacedName{Namespace: "default", Name: "pod2"}
		pod3NN := types.NamespacedName{Namespace: "default", Name: "pod3"}

		endpoints := []fwksched.Endpoint{
			fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod1NN}, &fwkdl.Metrics{}, nil),
			fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod2NN}, &fwkdl.Metrics{}, nil),
			fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod3NN}, &fwkdl.Metrics{}, nil),
		}

		// pod1: 0 in-flight. Score = 1 - 0/1000 = 1.0
		// pod2: 500 in-flight. Score = 1 - 500/1000 = 0.5
		endpoints[1].Put(scorer.inFlightLoadDataKey.String(), &attrconcurrency.InFlightLoad{Tokens: 500})
		// pod3: 1000 in-flight. Score = 1 - 1000/1000 = 0.0
		endpoints[2].Put(scorer.inFlightLoadDataKey.String(), &attrconcurrency.InFlightLoad{Tokens: 1000})

		scores := scorer.Score(context.Background(), fwksched.NewCycleState(), &fwksched.InferenceRequest{}, endpoints)

		assert.InDelta(t, 1.0, scores[endpoints[0]], 0.0001, "Pod1 (0 tokens) should have score 1.0")
		assert.InDelta(t, 0.5, scores[endpoints[1]], 0.0001, "Pod2 (500 tokens) should have score 0.5")
		assert.InDelta(t, 0.0, scores[endpoints[2]], 0.0001, "Pod3 (1000 tokens) should have score 0.0")
	})

	t.Run("in-flight load plus current request impact", func(t *testing.T) {
		pod1NN := types.NamespacedName{Namespace: "default", Name: "pod1"}
		pod2NN := types.NamespacedName{Namespace: "default", Name: "pod2"}
		pod3NN := types.NamespacedName{Namespace: "default", Name: "pod3"}

		endpoints := []fwksched.Endpoint{
			fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod1NN}, &fwkdl.Metrics{}, nil),
			fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod2NN}, &fwkdl.Metrics{}, nil),
			fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod3NN}, &fwkdl.Metrics{}, nil),
		}

		// pod1: 0 in-flight + 250 current = 250. Score = 1 - 250/1000 = 0.75
		endpoints[0].Put(scorer.currentRequestEndpointImpactDataKey.String(), &attrconcurrency.InFlightLoad{Tokens: 250})
		// pod2: 250 in-flight + 250 current = 500. Score = 1 - 500/1000 = 0.5
		endpoints[1].Put(scorer.inFlightLoadDataKey.String(), &attrconcurrency.InFlightLoad{Tokens: 250})
		endpoints[1].Put(scorer.currentRequestEndpointImpactDataKey.String(), &attrconcurrency.InFlightLoad{Tokens: 250})
		// pod3: 750 in-flight + 250 current = 1000. Score = 1 - 1000/1000 = 0.0
		endpoints[2].Put(scorer.inFlightLoadDataKey.String(), &attrconcurrency.InFlightLoad{Tokens: 750})
		endpoints[2].Put(scorer.currentRequestEndpointImpactDataKey.String(), &attrconcurrency.InFlightLoad{Tokens: 250})

		scores := scorer.Score(context.Background(), &fwksched.InferenceRequest{}, endpoints)

		assert.InDelta(t, 0.75, scores[endpoints[0]], 0.0001, "Pod1 (0 + 250 tokens) should have score 0.75")
		assert.InDelta(t, 0.5, scores[endpoints[1]], 0.0001, "Pod2 (250 + 250 tokens) should have score 0.5")
		assert.InDelta(t, 0.0, scores[endpoints[2]], 0.0001, "Pod3 (750 + 250 tokens) should have score 0.0")
	})

	t.Run("missing data keys degrades gracefully", func(t *testing.T) {
		podNN := types.NamespacedName{Namespace: "default", Name: "pod-no-data"}
		endpoints := []fwksched.Endpoint{
			fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: podNN}, &fwkdl.Metrics{}, nil),
		}

		// No keys set; should score as if 0 token load
		scores := scorer.Score(context.Background(), &fwksched.InferenceRequest{}, endpoints)
		assert.Equal(t, 1.0, scores[endpoints[0]], "Endpoint with missing keys should have score 1.0 (0 load)")
	})

	t.Run("typed nil attributes handle gracefully", func(t *testing.T) {
		podNN := types.NamespacedName{Namespace: "default", Name: "pod-typed-nil"}
		endpoints := []fwksched.Endpoint{
			fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: podNN}, &fwkdl.Metrics{}, nil),
		}

		var nilLoad *attrconcurrency.InFlightLoad
		endpoints[0].Put(scorer.inFlightLoadDataKey.String(), nilLoad)
		endpoints[0].Put(scorer.currentRequestEndpointImpactDataKey.String(), nilLoad)

		// Typed nils; should score as if 0 token load (no panic)
		scores := scorer.Score(context.Background(), fwksched.NewCycleState(), &fwksched.InferenceRequest{}, endpoints)
		assert.Equal(t, 1.0, scores[endpoints[0]], "Endpoint with typed nil attributes should have score 1.0 (0 load)")
	})
}
