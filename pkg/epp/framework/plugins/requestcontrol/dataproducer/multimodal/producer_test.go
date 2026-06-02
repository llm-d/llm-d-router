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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

func TestFactory(t *testing.T) {
	raw, err := json.Marshal(map[string]any{"cacheSize": 4})
	require.NoError(t, err)

	created, err := Factory("mm-producer", plugin.StrictDecoder(raw), &testHandle{ctx: context.Background()})
	require.NoError(t, err)
	require.NotNil(t, created)
	assert.Equal(t, "mm-producer", created.TypedName().Name)
	assert.Equal(t, ProducerType, created.TypedName().Type)

	_, err = Factory("bad", plugin.StrictDecoder(json.RawMessage(`{"cacheSize":"bad"}`)), &testHandle{ctx: context.Background()})
	require.Error(t, err)
}

func TestExtractMMItemsFromTokenizedPrompt(t *testing.T) {
	items := ExtractMMItems(&scheduling.InferenceRequest{
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

	assert.ElementsMatch(t, []attrmm.MatchItem{{Hash: "image-a", Size: 1}, {Hash: "image-b", Size: 1}}, items)
}

func TestExtractMMItemsNilTokenizedPromptReturnsNil(t *testing.T) {
	items := ExtractMMItems(&scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{},
	})
	assert.Nil(t, items)
}

func TestExtractMMItemsEmptyMultiModalFeaturesReturnsNil(t *testing.T) {
	items := ExtractMMItems(&scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{},
		},
	})
	assert.Nil(t, items)
}

func TestExtractMMItemsIgnoresProtocolStructs(t *testing.T) {
	// Protocol structs carry multimodal content but are never read; only the
	// tokenized prompt's features count.
	items := ExtractMMItems(&scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{
					Role: "user",
					Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
						{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/cat.png"}},
					}},
				}},
			},
		},
	})

	assert.Nil(t, items)
}

func TestProduceMatchesMultiplePodsAndPreRequestUpdatesPlacement(t *testing.T) {
	producer := newTestProducer(t, nil, nil)
	podA := k8stypes.NamespacedName{Namespace: "default", Name: "pod-a"}
	podB := k8stypes.NamespacedName{Namespace: "default", Name: "pod-b"}
	podC := k8stypes.NamespacedName{Namespace: "default", Name: "pod-c"}
	producer.putCacheEntry("hash-a", podA, podB)

	endpointA := newEndpoint(podA)
	endpointB := newEndpoint(podB)
	endpointC := newEndpoint(podC)
	request := requestWithHashes("req-1", map[string]int{"hash-a": 80, "hash-c": 20})

	require.NoError(t, producer.Produce(context.Background(), request, []scheduling.Endpoint{endpointA, endpointB, endpointC}))

	assertMatchInfo(t, producer, endpointA,
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1}},
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1}, {Hash: "hash-c", Size: 1}})
	assertMatchInfo(t, producer, endpointB,
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1}},
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1}, {Hash: "hash-c", Size: 1}})
	assertMatchInfo(t, producer, endpointC,
		nil,
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1}, {Hash: "hash-c", Size: 1}})

	producer.PreRequest(context.Background(), request, schedulingResult(endpointC))
	producer.wg.Wait()

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
		require.NoError(t, producer.Produce(context.Background(), request, []scheduling.Endpoint{endpoint}))
		producer.PreRequest(context.Background(), request, schedulingResult(endpoint))
		producer.wg.Wait()
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

	// Simulate the periodic cleanup loop firing.
	producer.removeStalePods()

	assert.NotContains(t, producer.cacheSnapshot()["hash-a"], podB.String())
	assert.Contains(t, producer.cacheSnapshot()["hash-a"], podA.String())

	endpointA := newEndpoint(podA)
	endpointB := newEndpoint(podB)
	require.NoError(t, producer.Produce(context.Background(), requestWithHashes("req", map[string]int{"hash-a": 1}), []scheduling.Endpoint{endpointA, endpointB}))

	assertMatchInfo(t, producer, endpointA,
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1}},
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1}})
	assertMatchInfo(t, producer, endpointB,
		nil,
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1}})
}

func TestProducerEndpointExtractorInterfaceContract(t *testing.T) {
	producer := newTestProducer(t, nil, nil)
	var _ fwkdl.EndpointExtractor = producer
	assert.True(t, reflect.TypeOf(producer).Implements(reflect.TypeFor[fwkdl.EndpointExtractor]()))
}

func TestExtractEndpointRemovesDeletedPod(t *testing.T) {
	podA := k8stypes.NamespacedName{Namespace: "default", Name: "pod-a"}
	podB := k8stypes.NamespacedName{Namespace: "default", Name: "pod-b"}
	producer := newTestProducer(t, nil, nil)
	producer.putCacheEntry("hash-a", podA, podB)

	err := producer.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: podB}, nil),
	})

	require.NoError(t, err)
	cache := producer.cacheSnapshot()
	assert.Contains(t, cache["hash-a"], podA.String())
	assert.NotContains(t, cache["hash-a"], podB.String())
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

func (h *testHandle) Metrics() plugin.MetricsRecorder {
	return nil
}

func (h *testHandle) PodList() []k8stypes.NamespacedName {
	if h.podList == nil {
		return nil
	}
	return h.podList()
}

const testName = "test-mm-embeddings-cache-producer"

func newTestProducer(t *testing.T, params *Parameters, podList func() []k8stypes.NamespacedName) *Producer {
	t.Helper()
	producer, err := New(context.Background(), testName, params, podList)
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
		RequestID: requestID,
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

func assertMatchInfo(t *testing.T, p *Producer, endpoint scheduling.Endpoint, matchedItems, requestItems []attrmm.MatchItem) {
	t.Helper()
	raw, ok := endpoint.Get(p.dk.String())
	require.True(t, ok)
	info, ok := raw.(*attrmm.EncoderCacheMatchInfo)
	require.True(t, ok)
	assert.ElementsMatch(t, matchedItems, info.MatchedItems())
	assert.ElementsMatch(t, requestItems, info.RequestItems())
}
