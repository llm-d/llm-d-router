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

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// testECParams is the ec_transfer_params payload a fake encoder reports for one
// media item.
var testECParams = map[string]any{"peer_port": 5501, "size_bytes": 1228800, "nixl_agent_metadata_b64": "bml4..."}

// encodeSubRequestHash returns the modality and hash one encode fanout
// sub-request carries.
//
// A sub-request covers exactly one entry, so features.mm_hashes holds a single
// modality key with a single hash under it (see singleEntryKwargs). Reading
// whichever key is present, rather than assuming image, is what lets a fake
// encoder serve audio and video sub-requests. The shape is asserted so an
// encode-side regression surfaces as a named failure instead of an index panic
// inside the handler.
//
// Failures are reported with Errorf, not Fatalf: this runs on the server's
// goroutine, and FailNow must be called from the goroutine running the test.
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

// newEncodePrefillGateway serves the encode and prefill phases for the flow
// tests below. ecFor decides what one encode sub-request gets back, keyed by the
// modality and hash it carried; returning nil sends an empty response, standing
// in for an encoder that reports no EC params. The prefill request body is
// decoded into prefillBody.
func newEncodePrefillGateway(t *testing.T, prefillBody *map[string]any, kvParams map[string]any, ecFor func(modality, hash string) map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch phase := r.Header.Get(gateway.EPPProfileHeader); phase {
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
			{Modality: ModalityImage, Hash: "img-hash-1", KwargsData: "dDE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Modality: ModalityImage, Hash: "img-hash-2", KwargsData: "dDI=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
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
	imageKwargs, _ := kwargsData[ModalityImage].([]any)
	if len(imageKwargs) != 2 || imageKwargs[0] != "dDE=" || imageKwargs[1] != "dDI=" {
		t.Fatalf("expected kwargs_data.image=[dDE=,dDI=], got %v", imageKwargs)
	}

	// Verify mm_hashes in features
	mmHashes, _ := features["mm_hashes"].(map[string]any)
	imageHashes, _ := mmHashes[ModalityImage].([]any)
	if len(imageHashes) != 2 {
		t.Fatalf("expected 2 mm_hashes in prefill features, got %d", len(imageHashes))
	}

	// Verify sampling_params with extra_args workaround
	samplingParams, _ := prefillBody["sampling_params"].(map[string]any)
	if samplingParams["max_tokens"] != float64(1) {
		t.Fatalf("expected sampling_params.max_tokens=1, got %v", samplingParams["max_tokens"])
	}
	extraArgs, ok := samplingParams["extra_args"].(map[string]any)
	if !ok {
		t.Fatal("expected extra_args in sampling_params for generate format")
	}
	kvParams, ok := extraArgs["kv_transfer_params"].(map[string]any)
	if !ok {
		t.Fatal("expected kv_transfer_params in extra_args")
	}
	if kvParams["do_remote_decode"] != true {
		t.Fatalf("expected do_remote_decode=true, got %v", kvParams["do_remote_decode"])
	}

	// Verify ec_transfer_params is a flat map keyed by mm_hash, nested in extra_args
	ecParams, ok := extraArgs["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("expected ec_transfer_params in extra_args, got %T", extraArgs["ec_transfer_params"])
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
			{Modality: ModalityImage, Hash: "img-hash-1", KwargsData: "dDE=", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Modality: ModalityImage, Hash: "img-hash-2", KwargsData: "dDI=", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 3}},
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

	samplingParams, _ := prefillBody["sampling_params"].(map[string]any)
	extraArgs, _ := samplingParams["extra_args"].(map[string]any)
	ecParams, ok := extraArgs["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("expected ec_transfer_params in extra_args (with reported hash only), got %T", extraArgs["ec_transfer_params"])
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

// TestEncodeToPrefill_MixedModalityECFlow runs the encode-to-prefill flow with
// one entry per modality.
//
// The two steps describe the same media items in different terms, and the
// prefill body is where the two meet. Encode reports EC params keyed by hash,
// which carries no modality, while prefill groups the features by modality. The
// per-step tests cover each side on its own; nothing else checks that the two
// agree once a request holds more than one modality.
func TestEncodeToPrefill_MixedModalityECFlow(t *testing.T) {
	var prefillBody map[string]any

	// Record what each sub-request asked for, so the fanout is checked to have
	// filed every entry under its own modality rather than defaulting to image.
	// The fanout is concurrent, so each sub-request lands on its own server
	// goroutine and the map needs a lock.
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
			{Modality: ModalityImage, Hash: "img-hash", KwargsData: "aW1n", Placeholder: pipeline.PlaceholderRange{Offset: 1, Length: 3}},
			{Modality: ModalityAudio, Hash: "aud-hash", KwargsData: "YXVk", Placeholder: pipeline.PlaceholderRange{Offset: 4, Length: 2}},
			{Modality: ModalityVideo, Hash: "vid-hash", KwargsData: "dmlk", Placeholder: pipeline.PlaceholderRange{Offset: 6, Length: 3}},
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
		ModalityImage: "img-hash",
		ModalityAudio: "aud-hash",
		ModalityVideo: "vid-hash",
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
		want  map[string]string
	}{
		{"mm_hashes", map[string]string{ModalityImage: "img-hash", ModalityAudio: "aud-hash", ModalityVideo: "vid-hash"}},
		{"kwargs_data", map[string]string{ModalityImage: "aW1n", ModalityAudio: "YXVk", ModalityVideo: "dmlk"}},
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
			items, _ := byMod[mod].([]any)
			if len(items) != 1 || items[0] != want {
				t.Errorf("%s[%s] = %v, want [%s]", f.field, mod, byMod[mod], want)
			}
		}
	}

	// ec_transfer_params stays one flat hash-keyed map across all modalities.
	samplingParams, _ := prefillBody["sampling_params"].(map[string]any)
	extraArgs, _ := samplingParams["extra_args"].(map[string]any)
	ecParams, ok := extraArgs["ec_transfer_params"].(map[string]any)
	if !ok {
		t.Fatalf("expected ec_transfer_params in extra_args, got %T", extraArgs["ec_transfer_params"])
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
