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

package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

func TestECMooncakeConsumerZMQ(t *testing.T) {
	tests := []struct {
		name    string
		podIP   string
		prefill string
		want    string
		wantErr bool
	}{
		{name: "prefiller is the consumer", podIP: "10.0.0.9", prefill: "10.0.0.1:8000", want: "tcp://10.0.0.1:14579"},
		{name: "prefiller with scheme", prefill: "http://10.0.0.1:8000", want: "tcp://10.0.0.1:14579"},
		{name: "ipv6 prefiller", prefill: "[fd00::1]:8000", want: "tcp://[fd00::1]:14579"},
		{name: "local decoder is the consumer", podIP: "10.0.0.9", want: "tcp://10.0.0.9:14579"},
		{name: "no consumer address", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := NewProxy(Config{Port: "0", ECMooncakePort: 14579, ECMooncakePodIP: tt.podIP})
			got, err := srv.ecMooncakeConsumerZMQ(tt.prefill)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

// encoderCall is what one encoder request carried.
type encoderCall struct {
	part        map[string]any
	consumerZMQ any
	transferIDs []any
}

// ecMooncakeHarness runs handleECMooncake against a fake encoder and prefiller.
// respond returns the ec_transfer_params JSON for an encoder request's part.
type ecMooncakeHarness struct {
	mu          sync.Mutex
	calls       []encoderCall
	prefillBody map[string]any
}

func runECMooncake(t *testing.T, path string, apiType reqcommon.APIType, body string, respond func(part map[string]any) string) (*ecMooncakeHarness, *httptest.ResponseRecorder) {
	t.Helper()
	h := &ecMooncakeHarness{}

	encoder := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		turns, _ := req[reqcommon.FieldMessages].([]any)
		if turns == nil {
			turns, _ = req[reqcommon.FieldInput].([]any)
		}
		require.Len(t, turns, 1)
		content := turns[0].(map[string]any)[reqcommon.FieldContent].([]any)
		require.Len(t, content, 1)
		part := content[0].(map[string]any)
		ec, _ := req[reqcommon.FieldECTransferParams].(map[string]any)
		require.NotNil(t, ec, "encoder request must carry ec_transfer_params")
		ecItems := ec[ecMooncakeFieldItems].([]any)
		ids := make([]any, 0, len(ecItems))
		for _, item := range ecItems {
			ids = append(ids, item.(map[string]any)[ecMooncakeFieldTransferID])
		}
		h.mu.Lock()
		h.calls = append(h.calls, encoderCall{part: part, consumerZMQ: ec[ecMooncakeFieldConsumerZMQ], transferIDs: ids})
		h.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}],"ec_transfer_params":` + respond(part) + `}`))
	}))
	t.Cleanup(encoder.Close)

	prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		h.mu.Lock()
		h.prefillBody = req
		h.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"kv_transfer_params":{}}`))
	}))
	t.Cleanup(prefill.Close)

	decodeURL, err := url.Parse("http://decoder:8000")
	require.NoError(t, err)
	srv := NewProxy(Config{
		Port: "0", DecoderURL: decodeURL,
		KVConnector: constants.KVConnectorNIXLV2, ECConnector: constants.ECConnectorMooncake,
		ECMooncakePort: 19019,
	})
	srv.logger = log.Log
	srv.allowlistValidator = &AllowlistValidator{}
	srv.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	})

	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set(routing.PrefillEndpointHeader, strings.TrimPrefix(prefill.URL, "http://"))
	req.Header.Set(routing.EncoderEndpointsHeader, strings.TrimPrefix(encoder.URL, "http://"))
	recorder := httptest.NewRecorder()
	srv.disaggregatedPrefillHandler(apiType)(recorder, req)
	return h, recorder
}

// callFor returns the encoder call whose part references url.
func (h *ecMooncakeHarness) callFor(t *testing.T, url string) encoderCall {
	t.Helper()
	for _, c := range h.calls {
		if reqcommon.MediaPartURL(c.part) == url {
			return c
		}
	}
	t.Fatalf("no encoder call for %s", url)
	return encoderCall{}
}

func prefillContent(t *testing.T, body map[string]any) []any {
	t.Helper()
	messages, ok := body[reqcommon.FieldMessages].([]any)
	require.True(t, ok)
	return messages[0].(map[string]any)[reqcommon.FieldContent].([]any)
}

// TestHandleECMooncakeRewritesChat pins the vLLM ECMooncakeConnector contract
// for a chat request: each encoder request names the consumer and one
// transfer_id, and the prefill request references each pushed embedding by a
// metadata-only embeds part plus an ec_items entry carrying the same
// transfer_id under the encoder-reported hash.
func TestHandleECMooncakeRewritesChat(t *testing.T) {
	const img1, img2 = "https://example.com/a.jpg", "https://example.com/b.jpg"
	body := `{"model":"m","messages":[{"role":"user","content":[` +
		`{"type":"text","text":"compare"},` +
		`{"type":"image_url","image_url":{"url":"` + img1 + `"}},` +
		`{"type":"image_url","image_url":{"url":"` + img2 + `"},"uuid":"client-uuid"}]}]}`

	h, rec := runECMooncake(t, reqcommon.PathChatCompletions, reqcommon.APITypeChatCompletions, body, func(part map[string]any) string {
		// The engine folds processing options into the hash, so it reports a
		// key other than the uuid it was sent.
		return `{"engine-` + part[partFieldUUID].(string) + `":{"metadata":{"image_grid_thw":[[1,4,6]]},"item_indices":[0]}}`
	})
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
	require.Len(t, h.calls, 2)

	call1, call2 := h.callFor(t, img1), h.callFor(t, img2)
	uuid1 := ecMooncakeContentUUID(imageURLItem(img1))
	assert.Equal(t, uuid1, call1.part[partFieldUUID], "a part without a uuid is keyed by its content")
	assert.Equal(t, "client-uuid", call2.part[partFieldUUID], "a client uuid is kept")
	for _, c := range []encoderCall{call1, call2} {
		assert.Equal(t, "tcp://127.0.0.1:19019", c.consumerZMQ, "the prefiller is the EC consumer")
		require.Len(t, c.transferIDs, 1)
	}
	assert.NotEqual(t, call1.transferIDs[0], call2.transferIDs[0])

	content := prefillContent(t, h.prefillBody)
	require.Len(t, content, 3)
	assert.Equal(t, map[string]any{"type": "text", "text": "compare"}, content[0])
	assert.Equal(t, map[string]any{
		"type":         "image_embeds",
		"image_embeds": map[string]any{"image_grid_thw": []any{float64(1), float64(4), float64(6)}},
		"uuid":         uuid1,
	}, content[1])
	assert.Equal(t, "client-uuid", content[2].(map[string]any)[partFieldUUID])

	ec := h.prefillBody[reqcommon.FieldECTransferParams].(map[string]any)
	assert.Contains(t, ec, "engine-"+uuid1)
	assert.Contains(t, ec, "engine-client-uuid")
	assert.Equal(t, []any{
		map[string]any{"mm_hash": "engine-" + uuid1, "transfer_id": call1.transferIDs[0]},
		map[string]any{"mm_hash": "engine-client-uuid", "transfer_id": call2.transferIDs[0]},
	}, ec[ecMooncakeFieldItems], "ec_items follow the media parts' order")
}

// TestHandleECMooncakeNoMetadataKeepsMedia pins that a part the encoder
// reported no metadata for keeps its media and gets no ec_items entry.
func TestHandleECMooncakeNoMetadataKeepsMedia(t *testing.T) {
	const img = "https://example.com/a.jpg"
	body := `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"` + img + `"}}]}]}`
	h, rec := runECMooncake(t, reqcommon.PathChatCompletions, reqcommon.APITypeChatCompletions, body, func(_ map[string]any) string {
		return `{"h":{"metadata":{},"item_indices":[0]}}`
	})
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())

	assert.Equal(t, []any{imageURLItem(img)}, prefillContent(t, h.prefillBody))
	ec := h.prefillBody[reqcommon.FieldECTransferParams].(map[string]any)
	assert.NotContains(t, ec, ecMooncakeFieldItems)
}

// TestHandleECMooncakeAudioStaysRaw pins that when an audio part is present
// and any part lacks metadata, every part keeps its media, tagged with its
// uuid, and still gets an ec_items entry.
func TestHandleECMooncakeAudioStaysRaw(t *testing.T) {
	const img, audio = "https://example.com/a.jpg", "https://example.com/a.wav"
	body := `{"model":"m","messages":[{"role":"user","content":[` +
		`{"type":"image_url","image_url":{"url":"` + img + `"}},` +
		`{"type":"audio_url","audio_url":{"url":"` + audio + `"}}]}]}`
	h, rec := runECMooncake(t, reqcommon.PathChatCompletions, reqcommon.APITypeChatCompletions, body, func(part map[string]any) string {
		if part["type"] == "audio_url" {
			return `{"h-audio":{"metadata":{},"item_indices":[0]}}`
		}
		return `{"h-img":{"metadata":{"image_grid_thw":[[1,2,2]]},"item_indices":[0]}}`
	})
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())

	content := prefillContent(t, h.prefillBody)
	require.Len(t, content, 2)
	assert.Equal(t, "image_url", content[0].(map[string]any)["type"])
	assert.Equal(t, ecMooncakeContentUUID(imageURLItem(img)), content[0].(map[string]any)[partFieldUUID])
	assert.Equal(t, "audio_url", content[1].(map[string]any)["type"])
	assert.Equal(t, ecMooncakeContentUUID(audioURLItem(audio)), content[1].(map[string]any)[partFieldUUID])

	ec := h.prefillBody[reqcommon.FieldECTransferParams].(map[string]any)
	assert.Equal(t, []any{
		map[string]any{"mm_hash": "h-img", "transfer_id": h.callFor(t, img).transferIDs[0]},
		map[string]any{"mm_hash": "h-audio", "transfer_id": h.callFor(t, audio).transferIDs[0]},
	}, ec[ecMooncakeFieldItems])
}

// TestHandleECMooncakeResponsesNoRewrite pins that a Responses request is
// forwarded with its media, matched by the engine-reported hash: no uuid is
// sent to the encoder, so both sides hash the content the same way.
func TestHandleECMooncakeResponsesNoRewrite(t *testing.T) {
	const img = "https://example.com/a.jpg"
	body := `{"model":"m","input":[{"role":"user","content":[{"type":"input_image","image_url":"` + img + `"}]}]}`
	h, rec := runECMooncake(t, reqcommon.PathResponses, reqcommon.APITypeResponses, body, func(part map[string]any) string {
		assert.NotContains(t, part, partFieldUUID)
		return `{"content-hash":{"metadata":{"image_grid_thw":[[1,2,2]]},"item_indices":[0]}}`
	})
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())

	input := h.prefillBody[reqcommon.FieldInput].([]any)
	assert.Equal(t, []any{inputImageItem(img)}, input[0].(map[string]any)[reqcommon.FieldContent])
	ec := h.prefillBody[reqcommon.FieldECTransferParams].(map[string]any)
	assert.Equal(t, []any{
		map[string]any{"mm_hash": "content-hash", "transfer_id": h.calls[0].transferIDs[0]},
	}, ec[ecMooncakeFieldItems])
}

// TestHandleECMooncakeUnmatchedMetadata pins that an encoder response whose
// entries cannot be tied to the item fails the request.
func TestHandleECMooncakeUnmatchedMetadata(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.jpg"}}]}]}`
	h, rec := runECMooncake(t, reqcommon.PathChatCompletions, reqcommon.APITypeChatCompletions, body, func(_ map[string]any) string {
		return `{"x":{"metadata":{}},"y":{"metadata":{}}}`
	})
	assert.Equal(t, http.StatusBadGateway, rec.Code)
	assert.Nil(t, h.prefillBody, "nothing is dispatched after a failed fanout")
}

func TestFlattenECMetadata(t *testing.T) {
	got := flattenECMetadata(map[string]any{
		"image_grid_thw": []any{[]any{1, 4, 6}},
		"timestamps":     []any{0.5, 1.0},
		"scalar":         7,
	})
	assert.Equal(t, map[string]any{
		"image_grid_thw": []any{1, 4, 6},
		"timestamps":     []any{0.5, 1.0},
		"scalar":         7,
	}, got)
}
