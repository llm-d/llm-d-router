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

package steps

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const (
	testImageURL = "data:image/png;base64,aW1n"
	testAudioB64 = "YXVk"
)

func mmKwargsChatBody() map[string]any {
	return map[string]any{
		"model": "m",
		"messages": []any{map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "describe"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": testImageURL}},
			map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": testAudioB64, "format": "wav"}},
		}}},
	}
}

// renderServer answers a chat render with two items, including kwargs_data only
// when the request did not set return_mm_kwargs to false.
func renderServer(t *testing.T, gotReturnMMKwargs *any) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var parsed map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&parsed))
		v, present := parsed["return_mm_kwargs"]
		*gotReturnMMKwargs = v
		features := map[string]any{
			"mm_hashes": map[string]any{"image": []string{"h-img"}, "audio": []string{"h-aud"}},
			"mm_placeholders": map[string]any{
				"image": []any{map[string]any{"offset": 1, "length": 3}},
				"audio": []any{map[string]any{"offset": 5, "length": 2}},
			},
		}
		if !present {
			features["kwargs_data"] = map[string]any{"image": []string{"aW1n"}, "audio": []string{"YXVk"}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 9, 9, 9, 7, 8, 8, 2},
			"features":  features,
		})
	}))
}

func runRender(t *testing.T, params map[string]any, gotReturnMMKwargs *any) *pipeline.RequestContext {
	server := renderServer(t, gotReturnMMKwargs)
	t.Cleanup(server.Close)
	step, err := NewRenderStep(nil, params)
	require.NoError(t, err)
	step.(*RenderStep).SetServiceAddress(server.URL)
	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
		Model:        "m",
		Body:         mmKwargsChatBody(),
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage}, {Modality: reqcommon.ModalityAudio},
		},
	}
	require.NoError(t, step.Execute(context.Background(), reqCtx))
	return reqCtx
}

func TestRenderStep_ReturnMMKwargs(t *testing.T) {
	var got any
	reqCtx := runRender(t, map[string]any{}, &got)
	require.Equal(t, false, got, "default render request must ask vLLM to omit tensors")
	require.True(t, reqCtx.MMKwargsOmitted)
	require.Empty(t, reqCtx.MultimodalEntries[0].KwargsData)
	require.Equal(t, "h-img", reqCtx.MultimodalEntries[0].Hash)
	_, inBody := reqCtx.Body["return_mm_kwargs"]
	require.False(t, inBody, "the field must not leak into the client body forwarded downstream")

	got = "unset"
	reqCtx = runRender(t, map[string]any{"return_mm_kwargs": true}, &got)
	require.Nil(t, got, "return_mm_kwargs: true sends the render request without the field")
	require.False(t, reqCtx.MMKwargsOmitted)
	require.Equal(t, "aW1n", reqCtx.MultimodalEntries[0].KwargsData)

	_, err := NewRenderStep(nil, map[string]any{"return_mm_kwargs": "no"})
	require.Error(t, err)
}

func TestEncodeAndPrefill_GenerateContentPartsWhenMMKwargsOmitted(t *testing.T) {
	var got any
	reqCtx := runRender(t, map[string]any{}, &got)

	var mu sync.Mutex
	var encodeBodies []map[string]any
	var prefillBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		require.NoError(t, json.Unmarshal(raw, &body))
		mu.Lock()
		defer mu.Unlock()
		if r.Header.Get(reqcommon.EPPProfileHeaderKey) == gateway.PhaseEncode {
			encodeBodies = append(encodeBodies, body)
			parts := body["content_parts"].([]any)
			hash := parts[0].(map[string]any)["uuid"].(string)
			_ = json.NewEncoder(w).Encode(map[string]any{"ec_transfer_params": map[string]any{hash: map[string]any{"peer_port": 1}}})
			return
		}
		prefillBody = body
		_ = json.NewEncoder(w).Encode(map[string]any{"kv_transfer_params": map[string]any{}})
	}))
	defer server.Close()
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})

	encode, err := NewEncodeStep(gwClient, map[string]any{"use_openai_format": false, ParamECConnector: ec.NIXL})
	require.NoError(t, err)
	require.NoError(t, encode.Execute(context.Background(), reqCtx))
	prefill, err := NewPrefillStep(gwClient, map[string]any{"use_openai_format": false, ParamECConnector: ec.NIXL})
	require.NoError(t, err)
	require.NoError(t, prefill.Execute(context.Background(), reqCtx))

	imagePart := map[string]any{"type": "image_url", "url": testImageURL, "uuid": "h-img"}
	audioPart := map[string]any{"type": "audio_url", "url": "data:audio/wav;base64," + testAudioB64, "uuid": "h-aud"}

	require.Len(t, encodeBodies, 2)
	byHash := map[string]map[string]any{}
	for _, b := range encodeBodies {
		require.NotContains(t, b, "features")
		byHash[b["content_parts"].([]any)[0].(map[string]any)["uuid"].(string)] = b
	}
	require.Equal(t, []any{imagePart}, byHash["h-img"]["content_parts"])
	require.Equal(t, []any{float64(1), float64(9)}, byHash["h-img"]["token_ids"])
	require.Equal(t, []any{audioPart}, byHash["h-aud"]["content_parts"])
	require.Equal(t, []any{float64(1), float64(8)}, byHash["h-aud"]["token_ids"])

	require.NotContains(t, prefillBody, "features")
	require.Equal(t, []any{imagePart, audioPart}, prefillBody["content_parts"])
	require.Equal(t, []any{float64(1), float64(9), float64(7), float64(8), float64(2)}, prefillBody["token_ids"])
	require.Contains(t, prefillBody, reqcommon.FieldECTransferParams)
}

func TestCollapsePlaceholders(t *testing.T) {
	entries := []pipeline.MultimodalEntry{
		{Placeholder: pipeline.PlaceholderRange{Offset: 5, Length: 2}},
		{Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
	}
	got, err := collapsePlaceholders([]int{1, 9, 9, 9, 7, 8, 8, 2}, entries)
	require.NoError(t, err)
	require.Equal(t, []int{1, 9, 7, 8, 2}, got)

	_, err = collapsePlaceholders([]int{1, 9, 9}, []pipeline.MultimodalEntry{{Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 5}}})
	require.Error(t, err)
	_, err = collapsePlaceholders([]int{1, 9, 9, 9}, []pipeline.MultimodalEntry{
		{Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 2}},
		{Placeholder: pipeline.PlaceholderRange{Offset: 2, Length: 2}},
	})
	require.Error(t, err)
}

// A renderer without the return_mm_kwargs field returns the tensors anyway, and
// the request keeps the features path.
func TestRenderStep_RendererIgnoringReturnMMKwargsKeepsFeatures(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 9, 9, 2},
			"features": map[string]any{
				"mm_hashes":       map[string]any{"image": []string{"h-img"}},
				"mm_placeholders": map[string]any{"image": []any{map[string]any{"offset": 1, "length": 2}}},
				"kwargs_data":     map[string]any{"image": []string{"aW1n"}},
			},
		})
	}))
	defer server.Close()
	step, err := NewRenderStep(nil, map[string]any{})
	require.NoError(t, err)
	step.(*RenderStep).SetServiceAddress(server.URL)
	reqCtx := &pipeline.RequestContext{
		OriginalPath:      reqcommon.PathChatCompletions,
		Body:              map[string]any{"model": "m", "messages": []any{}},
		MultimodalEntries: []pipeline.MultimodalEntry{{Modality: reqcommon.ModalityImage}},
	}
	require.NoError(t, step.Execute(context.Background(), reqCtx))
	require.False(t, reqCtx.MMKwargsOmitted)
	require.Equal(t, "aW1n", reqCtx.MultimodalEntries[0].KwargsData)
}

// A text-only request has no entries, so there is nothing to omit.
func TestRenderStep_TextOnlyLeavesMMKwargsOmittedUnset(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"token_ids": []int{1, 2}})
	}))
	defer server.Close()
	step, err := NewRenderStep(nil, map[string]any{})
	require.NoError(t, err)
	step.(*RenderStep).SetServiceAddress(server.URL)
	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
		Body:         map[string]any{"model": "m", "messages": []any{}},
	}
	require.NoError(t, step.Execute(context.Background(), reqCtx))
	require.False(t, reqCtx.MMKwargsOmitted)
}

// The media URL is read from wherever each API keeps it.
func TestGenerateContentParts_PerAPIShape(t *testing.T) {
	entry := pipeline.MultimodalEntry{Modality: reqcommon.ModalityImage, Hash: "h"}
	want := []map[string]any{{"type": "image_url", "url": testImageURL, "uuid": "h"}}
	for _, tc := range []struct {
		name string
		path string
		body map[string]any
	}{
		{
			name: "responses input_image",
			path: reqcommon.PathResponses,
			body: map[string]any{"input": []any{map[string]any{"role": "user", "content": []any{
				map[string]any{"type": reqcommon.PartTypeInputImage, "image_url": testImageURL},
			}}}},
		},
		{
			name: "messages base64 image block",
			path: reqcommon.PathMessages,
			body: map[string]any{"messages": []any{map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": "aW1n"}},
			}}}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reqCtx := &pipeline.RequestContext{OriginalPath: tc.path, Body: tc.body, MultimodalEntries: []pipeline.MultimodalEntry{entry}}
			got, err := generateContentParts(reqCtx)
			require.NoError(t, err)
			require.Equal(t, want, got)
		})
	}
}

// Every failure means entries and parts got out of line inside the
// coordinator, so the request fails instead of sending the worker a wrong body.
func TestGenerateContentParts_Errors(t *testing.T) {
	chat := func(parts ...any) map[string]any {
		return map[string]any{"messages": []any{map[string]any{"role": "user", "content": parts}}}
	}
	for _, tc := range []struct {
		name  string
		body  map[string]any
		entry pipeline.MultimodalEntry
	}{
		{
			name:  "no part for the entry",
			body:  chat(map[string]any{"type": "text", "text": "hi"}),
			entry: pipeline.MultimodalEntry{Modality: reqcommon.ModalityImage},
		},
		{
			name:  "part without a url",
			body:  chat(map[string]any{"type": "image_url", "image_url": map[string]any{}}),
			entry: pipeline.MultimodalEntry{Modality: reqcommon.ModalityImage},
		},
		{
			name:  "unsupported input_audio format",
			body:  chat(map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": testAudioB64, "format": "xyz"}}),
			entry: pipeline.MultimodalEntry{Modality: reqcommon.ModalityAudio},
		},
		{
			name:  "modality with no content_parts type",
			body:  chat(map[string]any{"type": "image_url", "image_url": map[string]any{"url": testImageURL}}),
			entry: pipeline.MultimodalEntry{Modality: "point_cloud"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reqCtx := &pipeline.RequestContext{
				OriginalPath:      reqcommon.PathChatCompletions,
				Body:              tc.body,
				MultimodalEntries: []pipeline.MultimodalEntry{tc.entry},
			}
			_, err := generateContentParts(reqCtx)
			require.Error(t, err)
		})
	}
}

// With OpenAI passthrough on, encode and prefill forward the client body and
// never build content_parts, so the flag changes nothing on that path.
func TestPrefillStep_OpenAIFormatIgnoresMMKwargsOmitted(t *testing.T) {
	var prefillBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&prefillBody))
		_ = json.NewEncoder(w).Encode(map[string]any{"kv_transfer_params": map[string]any{}})
	}))
	defer server.Close()
	step, err := NewPrefillStep(gateway.New(config.GatewayConfig{Address: server.URL}), map[string]any{})
	require.NoError(t, err)
	reqCtx := &pipeline.RequestContext{
		OriginalPath:      reqcommon.PathChatCompletions,
		Model:             "m",
		Body:              mmKwargsChatBody(),
		TokenIDs:          []int{1, 9, 9, 9, 2},
		MultimodalEntries: []pipeline.MultimodalEntry{{Modality: reqcommon.ModalityImage, Hash: "h", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}}},
		MMKwargsOmitted:   true,
	}
	require.NoError(t, step.Execute(context.Background(), reqCtx))
	require.NotContains(t, prefillBody, "content_parts")
	require.Contains(t, prefillBody, "messages")
}
