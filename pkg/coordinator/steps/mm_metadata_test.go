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

// vllmGridThwMetadata is the mm_metadata vLLM render returned for a 1920x1080
// image on Qwen3-VL-32B: image_grid_thw [1, 68, 120].
const vllmGridThwMetadata = "ga5pbWFnZV9ncmlkX3Rod4KkZGF0YZOlaW50NjSRA8cYAwEAAAAAAAAARAAAAAAAAAB4AAAAAAAAAKVmaWVsZJKnYmF0Y2hlZIGra2VlcF9vbl9jcHXD"

func gridThw(t, h, w float64) map[string]any {
	return map[string]any{"image_grid_thw": []any{t, h, w}}
}

func TestMMMetadataFromEC_MatchesVLLM(t *testing.T) {
	got, ok := mmMetadataFromEC(gridThw(1, 68, 120))
	require.True(t, ok)
	require.Equal(t, vllmGridThwMetadata, got)
}

func TestMMMetadataFromEC_RejectsUnsupportedValues(t *testing.T) {
	for name, metadata := range map[string]map[string]any{
		"empty":          {},
		"nil":            nil,
		"not a list":     {"image_grid_thw": float64(3)},
		"empty list":     {"image_grid_thw": []any{}},
		"fraction":       {"image_grid_thw": []any{1.5, float64(2)}},
		"string element": {"image_grid_thw": []any{"1"}},
		"nested list":    {"timestamps": []any{[]any{0.5}}},
	} {
		t.Run(name, func(t *testing.T) {
			_, ok := mmMetadataFromEC(metadata)
			require.False(t, ok)
		})
	}
}

// ecServer answers encode requests with a descriptor carrying metadata, and
// prefill requests with empty kv params, capturing each body by profile.
type ecServer struct {
	mu       sync.Mutex
	encode   []map[string]any
	prefill  map[string]any
	metadata func(hash string) map[string]any
}

func (e *ecServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	e.mu.Lock()
	defer e.mu.Unlock()
	if r.Header.Get(reqcommon.EPPProfileHeaderKey) == gateway.PhaseEncode {
		e.encode = append(e.encode, body)
		hash := body["content_parts"].([]any)[0].(map[string]any)["uuid"].(string)
		desc := map[string]any{"peer_port": 1}
		if md := e.metadata(hash); md != nil {
			desc["metadata"] = md
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"ec_transfer_params": map[string]any{hash: desc}})
		return
	}
	e.prefill = body
	_ = json.NewEncoder(w).Encode(map[string]any{"kv_transfer_params": map[string]any{}})
}

func runEncodePrefill(t *testing.T, srv *ecServer) {
	var got any
	reqCtx := runRender(t, map[string]any{"mm_payload": pipeline.MMPayloadNone}, &got)
	require.Equal(t, pipeline.MMPayloadNone, got)
	encodePrefill(t, srv, reqCtx)
}

func encodePrefill(t *testing.T, srv *ecServer, reqCtx *pipeline.RequestContext) {
	server := httptest.NewServer(srv)
	t.Cleanup(server.Close)
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	params := map[string]any{"use_openai_format": false, ParamECConnector: ec.NIXL}
	encode, err := NewEncodeStep(gwClient, params)
	require.NoError(t, err)
	require.NoError(t, encode.Execute(context.Background(), reqCtx))
	prefill, err := NewPrefillStep(gwClient, params)
	require.NoError(t, err)
	require.NoError(t, prefill.Execute(context.Background(), reqCtx))
}

// With the tensors omitted and every entry covered, encode carries the media
// and prefill gets mm_metadata rebuilt from the encode response instead of the
// media.
func TestEncodeAndPrefill_MMPayloadNoneRebuildsMetadata(t *testing.T) {
	srv := &ecServer{metadata: func(string) map[string]any { return gridThw(1, 68, 120) }}
	runEncodePrefill(t, srv)

	require.Len(t, srv.encode, 2)
	for _, b := range srv.encode {
		require.NotContains(t, b, "features")
		require.Len(t, b["content_parts"], 1)
	}

	body := srv.prefill
	require.NotContains(t, body, "content_parts")
	require.Equal(t, []any{float64(1), float64(9), float64(9), float64(9), float64(7), float64(8), float64(8), float64(2)}, body["token_ids"],
		"prefill keeps the expanded prompt when it sends features")
	features := body["features"].(map[string]any)
	require.Equal(t, map[string]any{"image": []any{"h-img"}, "audio": []any{"h-aud"}}, features["mm_hashes"])
	require.Equal(t, map[string]any{"image": []any{vllmGridThwMetadata}, "audio": []any{vllmGridThwMetadata}}, features["mm_metadata"])
	require.Equal(t, map[string]any{"image": []any{nil}, "audio": []any{nil}}, features["kwargs_data"])
	require.Contains(t, body, reqcommon.FieldECTransferParams)
}

// One entry without usable metadata sends the whole prefill request as
// content_parts: vLLM cannot mix features and content_parts.
func TestEncodeAndPrefill_MMPayloadNoneFallsBackToContentParts(t *testing.T) {
	srv := &ecServer{metadata: func(hash string) map[string]any {
		if hash == "h-aud" {
			return nil
		}
		return gridThw(1, 68, 120)
	}}
	runEncodePrefill(t, srv)

	body := srv.prefill
	require.NotContains(t, body, "features")
	require.Len(t, body["content_parts"], 2)
	require.Equal(t, []any{float64(1), float64(9), float64(7), float64(8), float64(2)}, body["token_ids"])
	require.Contains(t, body, reqcommon.FieldECTransferParams)
}

// mm_metadata the render response carried is sent as-is: the EC descriptor
// needs no metadata for an entry render already described.
func TestEncodeAndPrefill_RenderMetadataNeedsNoECMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 9, 9, 9, 2},
			"features": map[string]any{
				"mm_hashes":       map[string]any{"image": []string{"h-img"}},
				"mm_placeholders": map[string]any{"image": []any{map[string]any{"offset": 1, "length": 3}}},
				"mm_metadata":     map[string]any{"image": []string{vllmGridThwMetadata}},
			},
		})
	}))
	defer server.Close()
	step, err := NewRenderStep(nil, map[string]any{"mm_payload": pipeline.MMPayloadNone})
	require.NoError(t, err)
	step.(*RenderStep).SetServiceAddress(server.URL)
	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
		Model:        "m",
		Body: map[string]any{"model": "m", "messages": []any{map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": testImageURL}},
		}}}},
		MultimodalEntries: []pipeline.MultimodalEntry{{Modality: reqcommon.ModalityImage}},
	}
	require.NoError(t, step.Execute(context.Background(), reqCtx))
	require.Equal(t, vllmGridThwMetadata, reqCtx.MultimodalEntries[0].MMMetadata)

	srv := &ecServer{metadata: func(string) map[string]any { return nil }}
	encodePrefill(t, srv, reqCtx)

	body := srv.prefill
	require.NotContains(t, body, "content_parts")
	features := body["features"].(map[string]any)
	require.Equal(t, map[string]any{"image": []any{vllmGridThwMetadata}}, features["mm_metadata"])
	require.Equal(t, map[string]any{"image": []any{nil}}, features["kwargs_data"])
}
