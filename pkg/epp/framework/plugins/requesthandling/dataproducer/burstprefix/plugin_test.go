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

package burstprefix

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func tokenizedRequest(tokens []uint32) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: tokens}}},
		},
	}
}

// assignedReplica returns the name of the endpoint this producer steered the
// request to (the one scored with a non-zero match), or "" if none.
func (p *dataProducer) assignedReplica(t *testing.T, endpoints []fwksched.Endpoint) string {
	t.Helper()
	for _, ep := range endpoints {
		v, ok := ep.Get(p.dk)
		require.True(t, ok, "PrefixCacheMatchInfo must be attached to every endpoint")
		info, ok := v.(*attrprefix.PrefixCacheMatchInfo)
		require.True(t, ok)
		if info.MatchBlocks() > 0 {
			return ep.GetMetadata().ID.Name
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

	_, err = newDataProducer(context.Background(), "burst", config{WindowDurationMs: 100, MaxPerReplica: -1, BlockSizeTokens: 64, BalanceBy: "bogus"})
	assert.Error(t, err, "balanceBy must be requests or tokens")
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

func TestProduce_CancelledContextBeforeSeal(t *testing.T) {
	// A long window guarantees the batch cannot seal on its own before the
	// already-cancelled context is observed.
	p, err := newDataProducer(context.Background(), "burst", config{WindowDurationMs: maxWindowDurationMs, MaxPerReplica: unlimitedPerReplica, BlockSizeTokens: 4, MaxBatchSize: unlimitedBatchSize})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	endpoints := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2")}
	err = p.Produce(ctx, tokenizedRequest([]uint32{1, 2, 3, 4}), endpoints)
	require.Error(t, err, "a context cancelled before seal must return an error")

	for _, ep := range endpoints {
		v, ok := ep.Get(p.dk)
		if !ok {
			continue // no affinity attached is expected when Produce returns early
		}
		info, ok := v.(*attrprefix.PrefixCacheMatchInfo)
		require.True(t, ok)
		assert.Zero(t, info.MatchBlocks(), "a cancelled request must not produce a non-zero match")
	}
}

func TestProduce_SecondWindowAfterSeal(t *testing.T) {
	p, err := newDataProducer(context.Background(), "burst", config{WindowDurationMs: 30, MaxPerReplica: unlimitedPerReplica, BlockSizeTokens: 4, MaxBatchSize: unlimitedBatchSize})
	require.NoError(t, err)

	runWindow := func(prompt []uint32) []string {
		const samples = 4
		names := make([]string, samples)
		var wg sync.WaitGroup
		for i := 0; i < samples; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				endpoints := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3")}
				assert.NoError(t, p.Produce(context.Background(), tokenizedRequest(prompt), endpoints))
				names[i] = p.assignedReplica(t, endpoints)
			}(i)
		}
		wg.Wait()
		return names
	}

	first := runWindow([]uint32{1, 2, 3, 4, 5, 6, 7, 8})

	// seal must detach the batch so the next request opens a fresh window.
	p.mu.Lock()
	reset := p.batch == nil
	p.mu.Unlock()
	assert.True(t, reset, "seal must reset p.batch to nil after the first window")

	second := runWindow([]uint32{9, 10, 11, 12, 13, 14, 15, 16})

	for i := range first {
		assert.NotEmpty(t, first[i], "every first-window sample must be assigned")
		assert.Equal(t, first[0], first[i], "first-window samples must co-locate")
	}
	for i := range second {
		assert.NotEmpty(t, second[i], "every second-window sample must be assigned")
		assert.Equal(t, second[0], second[i], "second-window samples must co-locate")
	}
}

// mmTokenizedRequest builds a request whose prompt is text, a media placeholder
// run, and text, carrying the media's feature: the shape vLLM renders for a
// multimodal prompt, where the placeholder tokens are the same for every media
// of a size and only the feature hash tells two apart.
func mmTokenizedRequest(imageHash string) *fwksched.InferenceRequest {
	const placeholder = 9
	return &fwksched.InferenceRequest{
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{
				TokenIDs: []uint32{1, 2, 3, 4, placeholder, placeholder, placeholder, placeholder, placeholder, placeholder, 5, 6},
				MultiModalFeatures: []fwkrequest.MultiModalFeature{
					{Modality: reqcommon.ModalityImage, Hash: imageHash, Offset: 4, Length: 6},
				},
			}}},
		},
	}
}

// A burst of requests carrying the same image shares one prefix group and must
// co-locate; the same placeholder tokens with a different image are a
// different prefix to the hasher, so each request stands alone and no replica
// is steered.
func TestProduce_MediaContentSeparatesBursts(t *testing.T) {
	for _, tc := range []struct {
		name       string
		firstHash  string
		secondHash string
		wantShared bool
	}{
		{name: "same image co-locates", firstHash: "image-a", secondHash: "image-a", wantShared: true},
		{name: "different image stands alone", firstHash: "image-a", secondHash: "image-b", wantShared: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := newDataProducer(context.Background(), "burst", config{WindowDurationMs: 100, MaxPerReplica: unlimitedPerReplica, BlockSizeTokens: 4, MaxBatchSize: unlimitedBatchSize})
			require.NoError(t, err)

			endpoints := []fwksched.Endpoint{testEndpoint("pod1"), testEndpoint("pod2"), testEndpoint("pod3")}
			var first, second string
			var wg sync.WaitGroup
			wg.Add(2)
			go func() {
				defer wg.Done()
				err := p.Produce(context.Background(), mmTokenizedRequest(tc.firstHash), endpoints)
				assert.NoError(t, err)
				first = p.assignedReplica(t, endpoints)
			}()
			go func() {
				defer wg.Done()
				err := p.Produce(context.Background(), mmTokenizedRequest(tc.secondHash), endpoints)
				assert.NoError(t, err)
				second = p.assignedReplica(t, endpoints)
			}()
			wg.Wait()

			if tc.wantShared {
				assert.NotEmpty(t, first, "a shared image must group the burst")
				assert.Equal(t, first, second, "requests with the same image must co-locate on one replica")
			} else {
				assert.Empty(t, first, "a lone request in its group must not be steered")
				assert.Empty(t, second, "a lone request in its group must not be steered")
			}
		})
	}
}
