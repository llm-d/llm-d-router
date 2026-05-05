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

package multimodal

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrmm "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

func TestFactory(t *testing.T) {
	raw, err := json.Marshal(map[string]any{"cacheSize": 4})
	require.NoError(t, err)

	created, err := Factory("mm-producer", raw, &testHandle{ctx: context.Background()})
	require.NoError(t, err)
	require.NotNil(t, created)
	assert.Equal(t, "mm-producer", created.TypedName().Name)

	_, err = Factory("bad", json.RawMessage(`{"cacheSize":"bad"}`), &testHandle{ctx: context.Background()})
	require.Error(t, err)
}

func TestExtractMMHashesWithWeightsFromTokenizedPrompt(t *testing.T) {
	hashes := ExtractMMHashesWithWeights(&scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				MultiModalFeatures: []fwkrh.MultiModalFeature{
					{Hash: "image-a", Length: 576},
					{Hash: "image-b", Length: 0},
					{Hash: "image-a", Length: 144},
				},
			},
		},
	})

	assert.Equal(t, map[string]int{"image-a": 1, "image-b": 1}, hashes)
}

func TestExtractMMHashesWithWeightsFromStructuredChat(t *testing.T) {
	request := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{
					Role: "user",
					Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
						{Type: "text", Text: "describe"},
						{Type: "image_url", ImageURL: fwkrh.ImageBlock{Url: "https://example.com/cat.png"}},
						{Type: "video_url", VideoURL: fwkrh.VideoBlock{Url: "https://example.com/cat.mp4"}},
					}},
				}},
			},
		},
	}

	hashes := ExtractMMHashesWithWeights(request)
	require.Len(t, hashes, 2)
	assert.Contains(t, hashes, contentHash("image_url", "https://example.com/cat.png"))
	assert.Contains(t, hashes, contentHash("video_url", "https://example.com/cat.mp4"))
}

func TestPrepareDataMatchesMultiplePodsAndPreRequestUpdatesPlacement(t *testing.T) {
	producer := newTestProducer(t, nil, nil)
	podA := k8stypes.NamespacedName{Namespace: "default", Name: "pod-a"}
	podB := k8stypes.NamespacedName{Namespace: "default", Name: "pod-b"}
	podC := k8stypes.NamespacedName{Namespace: "default", Name: "pod-c"}
	producer.putCacheEntry("hash-a", podA, podB)

	endpointA := newEndpoint(podA)
	endpointB := newEndpoint(podB)
	endpointC := newEndpoint(podC)
	request := requestWithHashes("req-1", map[string]int{"hash-a": 80, "hash-c": 20})

	require.NoError(t, producer.PrepareRequestData(context.Background(), request, []scheduling.Endpoint{endpointA, endpointB, endpointC}))

	assertMatchInfo(t, endpointA, 1, 2, map[string]int{"hash-a": 1})
	assertMatchInfo(t, endpointB, 1, 2, map[string]int{"hash-a": 1})
	assertMatchInfo(t, endpointC, 0, 2, map[string]int{})

	producer.PreRequest(context.Background(), request, schedulingResult(endpointC))

	cache := producer.cacheSnapshot()
	assert.Contains(t, cache["hash-a"], podA.String())
	assert.Contains(t, cache["hash-a"], podB.String())
	assert.Contains(t, cache["hash-a"], podC.String())
	assert.Contains(t, cache["hash-c"], podC.String())
}

func TestLRUEviction(t *testing.T) {
	producer := newTestProducer(t, &Parameters{CacheSize: 2}, nil)
	endpoint := newEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "pod-a"})

	for _, hash := range []string{"hash-1", "hash-2", "hash-3"} {
		request := requestWithHashes(hash, map[string]int{hash: 1})
		require.NoError(t, producer.PrepareRequestData(context.Background(), request, []scheduling.Endpoint{endpoint}))
		producer.PreRequest(context.Background(), request, schedulingResult(endpoint))
	}

	cache := producer.cacheSnapshot()
	assert.NotContains(t, cache, "hash-1")
	assert.Contains(t, cache, "hash-2")
	assert.Contains(t, cache, "hash-3")
}

func TestStalePodCleanup(t *testing.T) {
	podA := k8stypes.NamespacedName{Namespace: "default", Name: "pod-a"}
	podB := k8stypes.NamespacedName{Namespace: "default", Name: "pod-b"}
	producer := newTestProducer(t, nil, func() []k8stypes.NamespacedName { return []k8stypes.NamespacedName{podA} })
	producer.putCacheEntry("hash-a", podA, podB)

	endpointA := newEndpoint(podA)
	endpointB := newEndpoint(podB)
	require.NoError(t, producer.PrepareRequestData(context.Background(), requestWithHashes("req", map[string]int{"hash-a": 1}), []scheduling.Endpoint{endpointA, endpointB}))

	assertMatchInfo(t, endpointA, 1, 1, map[string]int{"hash-a": 1})
	assertMatchInfo(t, endpointB, 0, 1, map[string]int{})
	assert.NotContains(t, producer.cacheSnapshot()["hash-a"], podB.String())
}

type testHandle struct {
	ctx     context.Context
	podList func() []k8stypes.NamespacedName
}

func (h *testHandle) Context() context.Context {
	return h.ctx
}

func (h *testHandle) Plugin(string) plugin.Plugin {
	return nil
}

func (h *testHandle) AddPlugin(string, plugin.Plugin) {}

func (h *testHandle) GetAllPlugins() []plugin.Plugin {
	return nil
}

func (h *testHandle) GetAllPluginsWithNames() map[string]plugin.Plugin {
	return nil
}

func (h *testHandle) PodList() []k8stypes.NamespacedName {
	if h.podList == nil {
		return nil
	}
	return h.podList()
}

func newTestProducer(t *testing.T, params *Parameters, podList func() []k8stypes.NamespacedName) *Producer {
	t.Helper()
	producer, err := New(context.Background(), params, podList)
	require.NoError(t, err)
	return producer
}

func newEndpoint(name k8stypes.NamespacedName) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: name},
		&fwkdl.Metrics{},
		nil,
	)
}

func requestWithHashes(requestID string, hashToWeight map[string]int) *scheduling.InferenceRequest {
	features := make([]fwkrh.MultiModalFeature, 0, len(hashToWeight))
	for hash, weight := range hashToWeight {
		features = append(features, fwkrh.MultiModalFeature{Hash: hash, Length: weight})
	}
	return &scheduling.InferenceRequest{
		RequestId: requestID,
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{MultiModalFeatures: features},
		},
	}
}

func schedulingResult(target scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{target}},
		},
	}
}

func assertMatchInfo(t *testing.T, endpoint scheduling.Endpoint, matched, total int, hashes map[string]int) {
	t.Helper()
	raw, ok := endpoint.Get(attrmm.EncoderCacheMatchInfoKey)
	require.True(t, ok)
	info, ok := raw.(*attrmm.EncoderCacheMatchInfo)
	require.True(t, ok)
	assert.Equal(t, matched, info.MatchedWeight())
	assert.Equal(t, total, info.TotalWeight())
	assert.Equal(t, hashes, info.MatchedHashes())
}
