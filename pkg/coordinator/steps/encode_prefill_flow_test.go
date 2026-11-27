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
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// testECParams is the ec_transfer_params payload a fake encoder reports for one
// media item.
var testECParams = map[string]any{"peer_port": 5501, "size_bytes": 1228800, "nixl_agent_metadata_b64": "bml4..."}

// encodeSubRequestHash returns the modality and hash one generate-format encode
// fanout sub-request carries. A sub-request covers exactly one entry, so
// features.mm_hashes holds one modality key with one hash; reading whichever key
// is present rather than assuming image is what lets a fake encoder serve audio
// and video. The shape is asserted so an encode-side regression surfaces as a
// named failure instead of an index panic. Failures use Errorf, not Fatalf:
// this runs on the server's goroutine, and FailNow must be called from the
// test's own.
func encodeSubRequestHash(t *testing.T, body []byte) (modality, hash string, ok bool) {
	t.Helper()
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Errorf("encode sub-request is not JSON: %v", err)
		return "", "", false
	}
	features, _ := parsed["features"].(map[string]any)
	mmHashes, _ := features["mm_hashes"].(map[string]any)
	if len(mmHashes) != 1 {
		t.Errorf("encode sub-request must carry exactly one modality, got %v", mmHashes)
		return "", "", false
	}
	for mod, raw := range mmHashes {
		hashes, _ := raw.([]any)
		if len(hashes) != 1 {
			t.Errorf("encode sub-request for %s must carry exactly one hash, got %v", mod, raw)
			return "", "", false
		}
		h, isString := hashes[0].(string)
		if !isString {
			t.Errorf("encode sub-request hash for %s must be a string, got %T", mod, hashes[0])
			return "", "", false
		}
		return mod, h, true
	}
	return "", "", false
}

// encodeSubRequestMediaURL returns the media URL one chat-completions-format
// encode fanout sub-request carries. Unlike the generate format, a chat
// sub-request carries the client's own media part (NewEncoderPrimingBody)
// rather than a coordinator-assigned hash, so the fake encoder below
// correlates sub-requests to entries by the URL instead, which the test gives
// a distinct value per logical image.
func encodeSubRequestMediaURL(t *testing.T, body []byte) (url string, ok bool) {
	t.Helper()
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Errorf("encode sub-request is not JSON: %v", err)
		return "", false
	}
	messages, _ := parsed["messages"].([]any)
	if len(messages) != 1 {
		t.Errorf("chat encode sub-request must carry exactly one message, got %v", messages)
		return "", false
	}
	turn, _ := messages[0].(map[string]any)
	content, _ := turn["content"].([]any)
	if len(content) != 1 {
		t.Errorf("chat encode sub-request must carry exactly one content part, got %v", content)
		return "", false
	}
	part, _ := content[0].(map[string]any)
	url = reqcommon.MediaPartURL(part)
	if url == "" {
		t.Errorf("chat encode sub-request content part carries no media url: %v", part)
		return "", false
	}
	return url, true
}

// newEncodePrefillGateway serves the encode and prefill phases for the flow
// tests below. ecFor decides what one encode sub-request gets back, keyed by
// the modality and hash it carried; nil sends an empty response, standing in
// for an encoder reporting no EC params. The prefill body lands in prefillBody.
func newEncodePrefillGateway(t *testing.T, prefillBody *map[string]any, kvParams map[string]any, ecFor func(modality, hash string) map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch phase := r.Header.Get(reqcommon.EPPProfileHeaderKey); phase {
		case gateway.PhaseEncode:
			body, _ := io.ReadAll(r.Body)
			modality, hash, ok := encodeSubRequestHash(t, body)
			if !ok {
				http.Error(w, "malformed encode sub-request", http.StatusInternalServerError)
				return
			}
			resp := map[string]any{}
			if params := ecFor(modality, hash); params != nil {
				resp["ec_transfer_params"] = map[string]any{hash: params}
			}
			_ = json.NewEncoder(w).Encode(resp)

		case gateway.PhasePrefill:
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, prefillBody)
			_ = json.NewEncoder(w).Encode(map[string]any{"kv_transfer_params": kvParams})

		default:
			http.Error(w, "unexpected phase: "+phase, http.StatusNotFound)
		}
	}))
}

// newChatEncodePrefillGateway is newEncodePrefillGateway's chat-completions
// counterpart: a chat sub-request carries the client's media URL rather than a
// coordinator hash (see encodeSubRequestMediaURL), so ecFor and the response it
// builds are keyed by that URL instead. The response key the test cares about,
// mapping the fake descriptor back onto an mm_hash for the prefill assertions,
// stays the caller's job: ecFor returns both the descriptor and the mm_hash to
// publish it under.
func newChatEncodePrefillGateway(t *testing.T, prefillBody *map[string]any, kvParams map[string]any, ecFor func(url string) (mmHash string, params map[string]any)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch phase := r.Header.Get(reqcommon.EPPProfileHeaderKey); phase {
		case gateway.PhaseEncode:
			body, _ := io.ReadAll(r.Body)
			url, ok := encodeSubRequestMediaURL(t, body)
			if !ok {
				http.Error(w, "malformed encode sub-request", http.StatusInternalServerError)
				return
			}
			resp := map[string]any{}
			if mmHash, params := ecFor(url); params != nil {
				resp["ec_transfer_params"] = map[string]any{mmHash: params}
			}
			_ = json.NewEncoder(w).Encode(resp)

		case gateway.PhasePrefill:
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, prefillBody)
			_ = json.NewEncoder(w).Encode(map[string]any{"kv_transfer_params": kvParams})

		default:
			http.Error(w, "unexpected phase: "+phase, http.StatusNotFound)
		}
	}))
}

func TestEncodeToPrefill_ECTransferParamsFlow(t *testing.T) {
	var prefillBody map[string]any

	gwServer := newEncodePrefillGateway(t, &prefillBody,
		map[string]any{"block_id": "b1", "peer_host": "10.0.0.2", "peer_port": 5502},
		func(_, _ string) map[string]any { return testECParams })
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	reqCtx := &pipeline.RequestContext{
		RequestID: "encode-prefill-flow",
		Model:     "llama-3",
		TokenIDs:  []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage, Hash: "img-hash-1", KwargsData: "dDE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Modality: reqcommon.ModalityImage, Hash: "img-hash-2", KwargsData: "dDI=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
		},
		KVTransferParams: make(map[string]any),
	}

	// Run encode step
	encodeStep, _ := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})

	err := encodeStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	// Verify encode populated ECTransferParams (ordered list)
	if len(reqCtx.ECTransferParams) != 2 {
		t.Fatalf("expected 2 ec_transfer_params, got %d", len(reqCtx.ECTransferParams))
	}

	// Run prefill step
	prefillStep, _ := NewPrefillStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})

	err = prefillStep.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("prefill failed: %v", err)
	}

	// Validate prefill request body
	if prefillBody == nil {
		t.Fatal("prefill was not called")
	}

	// Verify features has kwargs_data with per-image base64 tensors
	features, _ := prefillBody["features"].(map[string]any)
	kwargsData, ok := features["kwargs_data"].(map[string]any)
	if !ok {
		t.Fatalf("expected kwargs_data map in prefill, got %T", features["kwargs_data"])
	}
	imageKwargs, _ := kwargsData[string(reqcommon.ModalityImage)].([]any)
	if len(imageKwargs) != 2 || imageKwargs[0] != "dDE=" || imageKwargs[1] != "dDI=" {
		t.Fatalf("expected kwargs_data.image=[dDE=,dDI=], got %v", imageKwargs)
	}

	// Verify mm_hashes in features
	mmHashes, _ := features["mm_hashes"].(map[string]any)
	imageHashes, _ := mmHashes[string(reqcommon.ModalityImage)].([]any)
	if len(imageHashes) != 2 {
		t.Fatalf("expected 2 mm_hashes in prefill features, got %d", len(imageHashes))
	}

	// Verify sampling_params carries the capped generation fields.
	samplingParams, _ := prefillBody["sampling_params"].(map[string]any)
	if samplingParams["max_tokens"] != float64(1) {
		t.Fatalf("expected sampling_params.max_tokens=1, got %v", samplingParams["max_tokens"])
	}
	kvParams, ok := prefillBody["kv_transfer_params"].(map[string]any)
	if !ok {
		t.Fatal("expected top-level kv_transfer_params in generate format")
	}
	if kvParams["do_remote_decode"] != true {
		t.Fatalf("expected do_remote_decode=true, got %v", kvParams["do_remote_decode"])
	}

	// Verify ec_transfer_params is a top-level flat map keyed by mm_hash.
	ecParams, ok := prefillBody["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("expected top-level ec_transfer_params, got %T", prefillBody["ec_transfer_params"])
	}
	if len(ecParams) != 2 {
		t.Fatalf("expected 2 ec_transfer_params entries, got %d: %v", len(ecParams), ecParams)
	}
	for _, want := range []string{"img-hash-1", "img-hash-2"} {
		entry, ok := ecParams[want].(map[string]any)
		if !ok || len(entry) == 0 {
			t.Errorf("ec_transfer_params[%q] missing or empty: %v", want, ecParams[want])
		}
	}

	// Verify response populated KVTransferParams
	if reqCtx.KVTransferParams["block_id"] != "b1" {
		t.Fatalf("expected KVTransferParams.block_id=b1, got %v", reqCtx.KVTransferParams["block_id"])
	}
}

// TestEncodeToPrefill_DeduplicatesRepeatedHashes exercises the gap between the
// request's original entry order and the compacted index the encode fanout
// uses once same-hash duplicates drop out: entries after the duplicate sit at
// a different position in the compacted result slice than in
// reqCtx.MultimodalEntries, so a fix that indexed results by their slot in
// that slice (entryIndex) rather than their original entry position would misroute
// a later entry's response. The distinct-hash entry right after the duplicate
// exercises that gap with a result a mixup would actually be able to swap;
// trailing-empty keeps the no-descriptor path covered too.
func TestEncodeToPrefill_DeduplicatesRepeatedHashes(t *testing.T) {
	var (
		prefillBody map[string]any
		mu          sync.Mutex
		requests    = make(map[string]int)
	)

	gwServer := newEncodePrefillGateway(t, &prefillBody,
		map[string]any{"block_id": "b1"},
		func(_, hash string) map[string]any {
			mu.Lock()
			requests[hash]++
			count := requests[hash]
			mu.Unlock()
			if hash == "" {
				return nil
			}
			// peer_port is keyed off the hash, not a shared counter, so a
			// response that lands under the wrong hash reads as the wrong port
			// instead of merely a duplicate count.
			port := map[string]int{"same-hash": 5501, "distinct-hash": 5502}[hash]
			return map[string]any{"peer_host": "10.0.0.1", "peer_port": port + count}
		})
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})
	reqCtx := &pipeline.RequestContext{
		RequestID: "encode-prefill-duplicate-hash",
		Model:     "llama-3",
		TokenIDs:  []int{1, 32000, 32000, 32000, 32000},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage, Hash: "same-hash", KwargsData: "dDE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 1}},
			{Modality: reqcommon.ModalityImage, Hash: "same-hash", KwargsData: "dDE=", Placeholder: pipeline.PlaceholderRange{Offset: 2, Length: 1}},
			{Modality: reqcommon.ModalityImage, Hash: "distinct-hash", KwargsData: "dDI=", Placeholder: pipeline.PlaceholderRange{Offset: 3, Length: 1}},
			{Modality: reqcommon.ModalityImage, KwargsData: "dDM=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 1}},
		},
		KVTransferParams: make(map[string]any),
	}

	encodeStep, err := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := encodeStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	prefillStep, err := NewPrefillStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := prefillStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("prefill failed: %v", err)
	}
	if prefillBody == nil {
		t.Fatal("prefill was not called")
	}

	mu.Lock()
	defer mu.Unlock()
	if requests["same-hash"] != 1 {
		t.Errorf("same hash encode requests = %d, want 1", requests["same-hash"])
	}
	if requests["distinct-hash"] != 1 {
		t.Errorf("distinct hash encode requests = %d, want 1", requests["distinct-hash"])
	}
	if requests[""] != 1 {
		t.Errorf("empty hash encode requests = %d, want 1", requests[""])
	}
	if reqCtx.EncodeFanout != 3 {
		t.Errorf("encode fanout = %d, want 3", reqCtx.EncodeFanout)
	}
	if len(reqCtx.ECTransferParams) != 2 {
		t.Errorf("EC transfer responses = %d, want 2", len(reqCtx.ECTransferParams))
	}
	ecParams, ok := prefillBody["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("prefill ec_transfer_params = %T, want map", prefillBody["ec_transfer_params"])
	}
	if len(ecParams) != 2 {
		t.Errorf("prefill EC transfer params = %d entries, want 2: %v", len(ecParams), ecParams)
	}
	sameHash, ok := ecParams["same-hash"].(map[string]any)
	if !ok {
		t.Fatalf("prefill EC transfer params missing same-hash: %v", ecParams)
	}
	if sameHash["peer_port"] != float64(5502) {
		t.Errorf("same-hash peer_port = %v, want 5502", sameHash["peer_port"])
	}
	distinctHash, ok := ecParams["distinct-hash"].(map[string]any)
	if !ok {
		t.Fatalf("prefill EC transfer params missing distinct-hash: %v", ecParams)
	}
	if distinctHash["peer_port"] != float64(5503) {
		// A mixup that indexed the compacted result slice by resultIndex
		// rather than entryIndex would route same-hash's response here
		// instead, reporting 5502.
		t.Errorf("distinct-hash peer_port = %v, want 5503 (not same-hash's response)", distinctHash["peer_port"])
	}
	t.Logf("encode fanout=%d, same-hash requests=%d, distinct-hash requests=%d, empty-hash requests=%d, prefill ec_transfer_params=%v",
		reqCtx.EncodeFanout, requests["same-hash"], requests["distinct-hash"], requests[""], ecParams)
}

// TestEncodeToPrefill_DeduplicatesRepeatedHashes_OpenAIFormat is
// TestEncodeToPrefill_DeduplicatesRepeatedHashes's chat-completions
// counterpart: the compacted-index gap it exercises comes from the dedup
// check in Execute, which runs before buildEncodeBody picks a format, so the
// same misrouting risk applies to the chat body path and localIdx lookup,
// not only the generate path.
func TestEncodeToPrefill_DeduplicatesRepeatedHashes_OpenAIFormat(t *testing.T) {
	const (
		sameHashURL     = "data:image/png;base64,c2FtZQ=="
		distinctHashURL = "data:image/png;base64,ZGlzdGluY3Q="
		emptyHashURL    = "data:image/png;base64,ZW1wdHk="
	)
	var (
		prefillBody map[string]any
		mu          sync.Mutex
		requests    = make(map[string]int)
	)

	gwServer := newChatEncodePrefillGateway(t, &prefillBody,
		map[string]any{"block_id": "b1"},
		func(url string) (string, map[string]any) {
			mu.Lock()
			requests[url]++
			count := requests[url]
			mu.Unlock()
			switch url {
			case sameHashURL:
				return "same-hash", map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501 + count}
			case distinctHashURL:
				return "distinct-hash", map[string]any{"peer_host": "10.0.0.1", "peer_port": 5502 + count}
			default:
				// Mirrors the generate-format test's trailing empty-hash
				// entry: this image's encoder reports no descriptor.
				return "", nil
			}
		})
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})
	reqCtx := &pipeline.RequestContext{
		RequestID:    "encode-prefill-duplicate-hash-openai",
		Model:        "llama-3",
		OriginalPath: reqcommon.PathChatCompletions,
		TokenIDs:     []int{1, 32000, 32000, 32000, 32000},
		Body: map[string]any{
			"messages": []any{
				map[string]any{"role": reqcommon.RoleUser, "content": []any{
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": sameHashURL}},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": sameHashURL}},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": distinctHashURL}},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": emptyHashURL}},
				}},
			},
		},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage, Hash: "same-hash", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 1}},
			{Modality: reqcommon.ModalityImage, Hash: "same-hash", Placeholder: pipeline.PlaceholderRange{Offset: 2, Length: 1}},
			{Modality: reqcommon.ModalityImage, Hash: "distinct-hash", Placeholder: pipeline.PlaceholderRange{Offset: 3, Length: 1}},
			{Modality: reqcommon.ModalityImage, Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 1}},
		},
		KVTransferParams: make(map[string]any),
	}

	encodeStep, err := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": true,
		ParamECConnector:    ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := encodeStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	prefillStep, err := NewPrefillStep(gwClient, map[string]any{
		"use_openai_format": true,
		ParamECConnector:    ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := prefillStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("prefill failed: %v", err)
	}
	if prefillBody == nil {
		t.Fatal("prefill was not called")
	}

	mu.Lock()
	defer mu.Unlock()
	if requests[sameHashURL] != 1 {
		t.Errorf("same-hash image encode requests = %d, want 1", requests[sameHashURL])
	}
	if requests[distinctHashURL] != 1 {
		t.Errorf("distinct-hash image encode requests = %d, want 1", requests[distinctHashURL])
	}
	if requests[emptyHashURL] != 1 {
		t.Errorf("empty-hash image encode requests = %d, want 1", requests[emptyHashURL])
	}
	if reqCtx.EncodeFanout != 3 {
		t.Errorf("encode fanout = %d, want 3", reqCtx.EncodeFanout)
	}
	ecParams, ok := prefillBody["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("prefill ec_transfer_params = %T, want map", prefillBody["ec_transfer_params"])
	}
	if len(ecParams) != 2 {
		t.Errorf("prefill EC transfer params = %d entries, want 2: %v", len(ecParams), ecParams)
	}
	sameHash, ok := ecParams["same-hash"].(map[string]any)
	if !ok {
		t.Fatalf("prefill EC transfer params missing same-hash: %v", ecParams)
	}
	if sameHash["peer_port"] != float64(5502) {
		t.Errorf("same-hash peer_port = %v, want 5502", sameHash["peer_port"])
	}
	distinctHash, ok := ecParams["distinct-hash"].(map[string]any)
	if !ok {
		t.Fatalf("prefill EC transfer params missing distinct-hash: %v", ecParams)
	}
	if distinctHash["peer_port"] != float64(5503) {
		// Under the same mixup TestEncodeToPrefill_DeduplicatesRepeatedHashes
		// guards against, this would carry same-hash's response (5502) instead.
		t.Errorf("distinct-hash peer_port = %v, want 5503 (not same-hash's response)", distinctHash["peer_port"])
	}
	t.Logf("encode fanout=%d, same-hash requests=%d, distinct-hash requests=%d, empty-hash requests=%d, prefill ec_transfer_params=%v",
		reqCtx.EncodeFanout, requests[sameHashURL], requests[distinctHashURL], requests[emptyHashURL], ecParams)
}

// TestEncodeToPrefill_PartialECResponse verifies that when only some encoders
// return ec_transfer_params (others return 200 OK with no EC field), the
// flow does not panic, prefill is still sent, and ec_transfer_params on the
// prefill request contains only the hashes that were reported.
func TestEncodeToPrefill_PartialECResponse(t *testing.T) {
	var prefillBody map[string]any

	// Only image 1 gets EC params; image 2 reports none.
	gwServer := newEncodePrefillGateway(t, &prefillBody,
		map[string]any{"block_id": "b1"},
		func(_, hash string) map[string]any {
			if hash == "img-hash-1" {
				return testECParams
			}
			return nil
		})
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	reqCtx := &pipeline.RequestContext{
		RequestID: "partial-ec-flow",
		Model:     "llama-3",
		TokenIDs:  []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage, Hash: "img-hash-1", KwargsData: "dDE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Modality: reqcommon.ModalityImage, Hash: "img-hash-2", KwargsData: "dDI=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
		},
		KVTransferParams: make(map[string]any),
	}

	encodeStep, _ := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err := encodeStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	// Only image 1 contributed to ECTransferParams; image 2's empty response was skipped.
	if len(reqCtx.ECTransferParams) != 1 {
		t.Fatalf("expected 1 ECTransferParams entry after partial response, got %d: %v",
			len(reqCtx.ECTransferParams), reqCtx.ECTransferParams)
	}

	prefillStep, _ := NewPrefillStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err := prefillStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("prefill failed: %v", err)
	}

	if prefillBody == nil {
		t.Fatal("prefill was not called")
	}

	ecParams, ok := prefillBody["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("expected top-level ec_transfer_params (with reported hash only), got %T", prefillBody["ec_transfer_params"])
	}
	if len(ecParams) != 1 {
		t.Fatalf("expected 1 ec_transfer_params entry, got %d: %v", len(ecParams), ecParams)
	}
	if _, ok := ecParams["img-hash-1"]; !ok {
		t.Errorf("expected img-hash-1 in ec_transfer_params, got %v", ecParams)
	}
	if _, ok := ecParams["img-hash-2"]; ok {
		t.Errorf("unexpected img-hash-2 in ec_transfer_params (encoder did not report it): %v", ecParams)
	}
}

// The encode-to-prefill flow with one entry per modality. The prefill body is
// where the two steps meet: encode reports EC params keyed by hash, which
// carries no modality, while prefill groups features by modality. The per-step
// tests cover each side alone, so nothing else checks that they agree once a
// request holds more than one modality.
func TestEncodeToPrefill_MixedModalityECFlow(t *testing.T) {
	var prefillBody map[string]any

	// Records what each sub-request asked for, to check the fanout filed every
	// entry under its own modality rather than defaulting to image. The fanout
	// is concurrent, so the map needs a lock.
	var mu sync.Mutex
	seen := map[string]string{}
	gwServer := newEncodePrefillGateway(t, &prefillBody,
		map[string]any{"block_id": "b1"},
		func(modality, hash string) map[string]any {
			mu.Lock()
			defer mu.Unlock()
			seen[modality] = hash
			return testECParams
		})
	defer gwServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gwServer.URL})

	reqCtx := &pipeline.RequestContext{
		RequestID: "mixed-modality-flow",
		Model:     "llama-3",
		TokenIDs:  []int{1, 32000, 32000, 32000, 51000, 51000, 71000, 71000, 71000, 2345},
		MultimodalEntries: []pipeline.MultimodalEntry{
			{Modality: reqcommon.ModalityImage, Hash: "img-hash", KwargsData: "aW1n", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Modality: reqcommon.ModalityAudio, Hash: "aud-hash", KwargsData: "YXVk", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 2}},
			{Modality: reqcommon.ModalityVideo, Hash: "vid-hash", KwargsData: "dmlk", Placeholder: pipeline.PlaceholderRange{Offset: 6, Length: 3}},
		},
		KVTransferParams: make(map[string]any),
	}

	encodeStep, err := NewEncodeStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := encodeStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	wantSeen := map[string]string{
		string(reqcommon.ModalityImage): "img-hash",
		string(reqcommon.ModalityAudio): "aud-hash",
		string(reqcommon.ModalityVideo): "vid-hash",
	}
	mu.Lock()
	gotSeen := maps.Clone(seen)
	mu.Unlock()
	if !reflect.DeepEqual(gotSeen, wantSeen) {
		t.Errorf("encode fanout sent %v, want %v", gotSeen, wantSeen)
	}
	if len(reqCtx.ECTransferParams) != 3 {
		t.Fatalf("expected 3 ec_transfer_params, got %d: %v", len(reqCtx.ECTransferParams), reqCtx.ECTransferParams)
	}

	prefillStep, err := NewPrefillStep(gwClient, map[string]any{
		"use_openai_format": false,
		ParamECConnector:    ec.NIXL,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := prefillStep.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("prefill failed: %v", err)
	}
	if prefillBody == nil {
		t.Fatal("prefill was not called")
	}

	// Each modality keeps its own slot in both feature maps.
	features, _ := prefillBody["features"].(map[string]any)
	for _, f := range []struct {
		field string
		want  map[reqcommon.Modality]string
	}{
		{"mm_hashes", map[reqcommon.Modality]string{reqcommon.ModalityImage: "img-hash", reqcommon.ModalityAudio: "aud-hash", reqcommon.ModalityVideo: "vid-hash"}},
		{"kwargs_data", map[reqcommon.Modality]string{reqcommon.ModalityImage: "aW1n", reqcommon.ModalityAudio: "YXVk", reqcommon.ModalityVideo: "dmlk"}},
	} {
		byMod, ok := features[f.field].(map[string]any)
		if !ok {
			t.Errorf("expected %s map in prefill features, got %T", f.field, features[f.field])
			continue
		}
		if len(byMod) != len(f.want) {
			t.Errorf("%s has %d modalities, want %d: %v", f.field, len(byMod), len(f.want), byMod)
		}
		for mod, want := range f.want {
			items, _ := byMod[string(mod)].([]any)
			if len(items) != 1 || items[0] != want {
				t.Errorf("%s[%s] = %v, want [%s]", f.field, mod, byMod[string(mod)], want)
			}
		}
	}

	// ec_transfer_params stays one flat hash-keyed map across all modalities,
	// at the top level of the generate body.
	ecParams, ok := prefillBody["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("expected top-level ec_transfer_params, got %T", prefillBody["ec_transfer_params"])
	}
	if len(ecParams) != 3 {
		t.Fatalf("expected 3 ec_transfer_params entries, got %d: %v", len(ecParams), ecParams)
	}
	for _, hash := range []string{"img-hash", "aud-hash", "vid-hash"} {
		if entry, ok := ecParams[hash].(map[string]any); !ok || len(entry) == 0 {
			t.Errorf("ec_transfer_params[%q] missing or empty: %v", hash, ecParams[hash])
		}
	}
}
