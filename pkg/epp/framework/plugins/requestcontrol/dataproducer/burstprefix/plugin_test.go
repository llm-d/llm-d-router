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

package burstprefix

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func tokenizedRequest(tokens []uint32) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{tokens}},
		},
	}
}

// assignedReplica returns the name of the endpoint this producer steered the
// request to (the one scored with a non-zero match), or "" if none.
func (p *dataProducer) assignedReplica(t *testing.T, endpoints []fwksched.Endpoint) string {
	t.Helper()
	for _, ep := range endpoints {
		v, ok := ep.Get(p.dk.String())
		require.True(t, ok, "PrefixCacheMatchInfo must be attached to every endpoint")
		info, ok := v.(*attrprefix.PrefixCacheMatchInfo)
		require.True(t, ok)
		if info.MatchBlocks() > 0 {
			return ep.GetMetadata().NamespacedName.Name
		}
	}
	return ""
}

func TestNew_RejectsInvalidConfig(t *testing.T) {
	_, err := newDataProducer(context.Background(), "burst", config{WindowDurationMs: 0, MaxPerReplica: -1, BlockSizeTokens: 64})
	assert.Error(t, err, "windowDurationMs must be > 0")

	_, err = newDataProducer(context.Background(), "burst", config{WindowDurationMs: 100, MaxPerReplica: 0, BlockSizeTokens: 64})
	assert.Error(t, err, "maxPerReplica 0 is invalid")

	_, err = newDataProducer(context.Background(), "burst", config{WindowDurationMs: 100, MaxPerReplica: -1, BlockSizeTokens: 0})
	assert.Error(t, err, "blockSizeTokens must be > 0")
}

func TestProduce_ColocatesIdenticalPromptBurst(t *testing.T) {
	p, err := newDataProducer(context.Background(), "burst", config{WindowDurationMs: 100, MaxPerReplica: unlimitedPerReplica, BlockSizeTokens: 4, MaxBatchSize: unlimitedBatchSize})
	require.NoError(t, err)

	const samples = 8
	names := make([]string, samples)
	var wg sync.WaitGroup
	for i := 0; i < samples; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Each request sees its own snapshot of the same four replicas.
			endpoints := []fwksched.Endpoint{
				testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3"), testEndpoint("pod4"),
			}
			err := p.Produce(context.Background(), tokenizedRequest([]uint32{1, 2, 3, 4, 5, 6, 7, 8}), endpoints)
			assert.NoError(t, err)
			names[i] = p.assignedReplica(t, endpoints)
		}(i)
	}
	wg.Wait()

	for i := 0; i < samples; i++ {
		assert.NotEmpty(t, names[i], "every sample of a duplicated prompt must be assigned a replica")
		assert.Equal(t, names[0], names[i], "all samples of one prompt must co-locate on the same replica")
	}
}

func TestProduce_SingletonHasNoAffinity(t *testing.T) {
	p, err := newDataProducer(context.Background(), "burst", config{WindowDurationMs: 50, MaxPerReplica: unlimitedPerReplica, BlockSizeTokens: 4, MaxBatchSize: unlimitedBatchSize})
	require.NoError(t, err)

	endpoints := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	err = p.Produce(context.Background(), tokenizedRequest([]uint32{1, 2, 3, 4}), endpoints)
	require.NoError(t, err)

	assert.Empty(t, p.assignedReplica(t, endpoints),
		"a lone request in a window shares no prompt and must not be steered")
}
