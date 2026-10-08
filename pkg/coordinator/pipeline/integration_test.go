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

package pipeline_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
)

func TestFullPipeline_AllConnectorCombinations(t *testing.T) {
	cases := []struct {
		kvConnector     string
		ecConnector     string
		wantECInPrefill bool // ec_transfer_params should be present in prefill body
	}{
		{kv.NIXL, ec.NIXL, true},
		{kv.NIXL, ec.SharedStorage, false},
		{kv.SharedStorage, ec.NIXL, true},
		{kv.SharedStorage, ec.SharedStorage, false},
	}

	for _, tc := range cases {
		t.Run(tc.kvConnector+"+"+tc.ecConnector, func(t *testing.T) {
			renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"token_ids": []int{1, 32000, 32000, 32000, 2345, 6789},
					"features": map[string]any{
						"mm_hashes":       map[reqcommon.Modality][]string{reqcommon.ModalityImage: {"vllm-hash-img0"}},
						"mm_placeholders": map[reqcommon.Modality][]any{reqcommon.ModalityImage: {map[string]any{"offset": 1, "length": 3}}},
						"kwargs_data":     map[reqcommon.Modality][]string{reqcommon.ModalityImage: {"dGVuc29yLWRhdGE="}},
					},
				})
			}))
			defer renderServer.Close()

			var mu sync.Mutex
			var capturedPrefillBody map[string]any

			gatewayServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				phase := r.Header.Get(reqcommon.EPPProfileHeaderKey)
				switch phase {
				case gateway.PhaseEncode:
					body, _ := io.ReadAll(r.Body)
					var parsed map[string]any
					_ = json.Unmarshal(body, &parsed)
					// Generate format: features at top level
					features, _ := parsed["features"].(map[string]any)
					mmHashes, _ := features["mm_hashes"].(map[string]any)
					imageHashes, _ := mmHashes[string(reqcommon.ModalityImage)].([]any)
					hash, _ := imageHashes[0].(string)
					_ = json.NewEncoder(w).Encode(map[string]any{
						"ec_transfer_params": map[string]any{
							hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
						},
					})
				case gateway.PhasePrefill:
					body, _ := io.ReadAll(r.Body)
					var parsed map[string]any
					_ = json.Unmarshal(body, &parsed)
					mu.Lock()
					capturedPrefillBody = parsed
					mu.Unlock()
					_ = json.NewEncoder(w).Encode(map[string]any{
						"kv_transfer_params": map[string]any{
							"block_id":  "abc123",
							"peer_host": "10.0.0.2",
							"peer_port": 5502,
						},
					})
				case gateway.PhaseDecode:
					_ = json.NewEncoder(w).Encode(map[string]any{
						"choices": []map[string]any{
							{"message": map[string]any{"role": "assistant", "content": "Hello!"}},
						},
					})
				default:
					http.Error(w, "not found", 404)
				}
			}))
			defer gatewayServer.Close()

			gwClient := gateway.New(config.GatewayConfig{Address: gatewayServer.URL, MaxIdleConnsPerHost: 10})

			stepConfigs := []config.StepConfig{
				{Type: "replace-media-urls", Params: map[string]any{"download_timeout": "5s"}},
				{Type: "render", Params: map[string]any{"endpoint": reqcommon.PathChatCompletions + "/render"}},
				{Type: "encode", Params: map[string]any{"use_openai_format": false, steps.ParamECConnector: tc.ecConnector}},
				{Type: "prefill", Params: map[string]any{"use_openai_format": false, steps.ParamKVConnector: tc.kvConnector, steps.ParamECConnector: tc.ecConnector}},
				{Type: "decode", Params: map[string]any{steps.ParamKVConnector: tc.kvConnector}},
			}

			pipelineSteps := make([]pipeline.Step, 0, len(stepConfigs))
			for _, sc := range stepConfigs {
				step, err := pipeline.Build(sc.Type, gwClient, sc.Params)
				if err != nil {
					t.Fatalf("building step %s: %v", sc.Type, err)
				}
				if ra, ok := step.(renderAware); ok {
					ra.SetServiceAddress(renderServer.URL)
				}
				pipelineSteps = append(pipelineSteps, step)
			}

			requestBody := `{
				"model": "test-model", "stream": false,
				"messages": [{"role": "user", "content": [
					{"type": "text", "text": "What is in this image?"},
					{"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZS1pbWFnZS1kYXRh"}}
				]}]
			}`

			recorder := httptest.NewRecorder()
			reqCtx := &pipeline.RequestContext{
				RequestID:        "test-" + tc.kvConnector + "+" + tc.ecConnector,
				OriginalPath:     reqcommon.PathChatCompletions,
				OriginalBody:     []byte(requestBody),
				Model:            "test-model",
				KVTransferParams: make(map[string]any),
				ResponseWriter:   recorder,
			}
			_ = json.Unmarshal([]byte(requestBody), &reqCtx.Body)

			if err := pipeline.New(pipelineSteps).Execute(t.Context(), reqCtx); err != nil {
				t.Fatalf("pipeline failed: %v", err)
			}

			respBody, _ := io.ReadAll(recorder.Result().Body)
			if !strings.Contains(string(respBody), "Hello!") {
				t.Fatalf("expected 'Hello!' in response, got: %s", respBody)
			}

			if tc.wantECInPrefill {
				if len(reqCtx.ECTransferParams) == 0 {
					t.Error("expected ECTransferParams to be populated")
				}
			} else {
				if len(reqCtx.ECTransferParams) != 0 {
					t.Errorf("expected ECTransferParams to be empty, got %d entries", len(reqCtx.ECTransferParams))
				}
			}
			if len(reqCtx.KVTransferParams) == 0 {
				t.Error("expected KVTransferParams to be populated")
			}

			mu.Lock()
			captured := capturedPrefillBody
			mu.Unlock()
			if captured == nil {
				t.Fatal("prefill was not called")
			}
			// Generate format carries transfer params at the top level of the body.
			_, hasEC := captured["ec_transfer_params"]
			if tc.wantECInPrefill && !hasEC {
				t.Error("expected top-level ec_transfer_params in prefill body")
			}
			if !tc.wantECInPrefill && hasEC {
				t.Error("unexpected top-level ec_transfer_params in prefill body")
			}
		})
	}
}

func TestFullPipeline_Integration(t *testing.T) {
	renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 32000, 32000, 32000, 2345, 6789},
			"features": map[string]any{
				"mm_hashes":       map[reqcommon.Modality][]string{reqcommon.ModalityImage: {"vllm-hash-img0"}},
				"mm_placeholders": map[reqcommon.Modality][]any{reqcommon.ModalityImage: {map[string]any{"offset": 1, "length": 3}}},
				"kwargs_data":     map[reqcommon.Modality][]string{reqcommon.ModalityImage: {"dGVuc29yLWRhdGE="}},
			},
		})
	}))
	defer renderServer.Close()

	var mu sync.Mutex
	var capturedDecodeBody map[string]any

	gatewayServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		phase := r.Header.Get(reqcommon.EPPProfileHeaderKey)
		switch phase {
		case gateway.PhaseEncode:
			body, _ := io.ReadAll(r.Body)
			var parsed map[string]any
			_ = json.Unmarshal(body, &parsed)
			features, _ := parsed["features"].(map[string]any)
			mmHashes, _ := features["mm_hashes"].(map[string]any)
			imageHashes, _ := mmHashes[string(reqcommon.ModalityImage)].([]any)
			hash, _ := imageHashes[0].(string)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ec_transfer_params": map[string]any{
					hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
				},
			})
		case gateway.PhasePrefill:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{
					"block_id":  "abc123",
					"peer_host": "10.0.0.2",
					"peer_port": 5502,
				},
			})
		case gateway.PhaseDecode:
			body, _ := io.ReadAll(r.Body)
			var parsed map[string]any
			_ = json.Unmarshal(body, &parsed)
			mu.Lock()
			capturedDecodeBody = parsed
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{
					{"message": map[string]any{"role": "assistant", "content": "Hello!"}},
				},
			})
		default:
			http.Error(w, "not found", 404)
		}
	}))
	defer gatewayServer.Close()

	gwClient := gateway.New(config.GatewayConfig{
		Address:             gatewayServer.URL,
		MaxIdleConnsPerHost: 10,
	})

	stepConfigs := []config.StepConfig{
		{Type: "replace-media-urls", Params: map[string]any{"download_timeout": "5s"}},
		{Type: "render", Params: map[string]any{"endpoint": reqcommon.PathChatCompletions + "/render"}},
		{Type: "encode", Params: map[string]any{"use_openai_format": false, steps.ParamECConnector: ec.NIXL}},
		{Type: "prefill", Params: map[string]any{"use_openai_format": false, steps.ParamECConnector: ec.NIXL}},
		{Type: "decode"},
	}

	pipelineSteps := make([]pipeline.Step, 0, len(stepConfigs))
	for _, sc := range stepConfigs {
		step, err := pipeline.Build(sc.Type, gwClient, sc.Params)
		if err != nil {
			t.Fatalf("building step %s: %v", sc.Type, err)
		}

		if ra, ok := step.(renderAware); ok {
			ra.SetServiceAddress(renderServer.URL)
		}

		pipelineSteps = append(pipelineSteps, step)
	}

	p := pipeline.New(pipelineSteps)

	requestBody := `{
		"model": "test-model",
		"stream": false,
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "What is in this image?"},
					{"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZS1pbWFnZS1kYXRh"}}
				]
			}
		]
	}`

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "test-123",
		OriginalPath:     reqcommon.PathChatCompletions,
		OriginalBody:     []byte(requestBody),
		Stream:           false,
		Model:            "test-model",
		KVTransferParams: make(map[string]any),
		ResponseWriter:   recorder,
	}

	_ = json.Unmarshal([]byte(requestBody), &reqCtx.Body)

	err := p.Execute(t.Context(), reqCtx)
	if err != nil {
		t.Fatalf("pipeline execution failed: %v", err)
	}

	result := recorder.Result()
	if result.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", result.StatusCode)
	}

	respBody, _ := io.ReadAll(result.Body)
	if !strings.Contains(string(respBody), "Hello!") {
		t.Fatalf("expected response to contain 'Hello!', got: %s", string(respBody))
	}

	if len(reqCtx.ECTransferParams) == 0 {
		t.Fatal("expected ECTransferParams to be populated")
	}
	if len(reqCtx.KVTransferParams) == 0 {
		t.Fatal("expected KVTransferParams to be populated")
	}

	// The decode step's body format must track the request's original path
	// (/v1/chat/completions) rather than the pipeline's use_openai_format
	// setting, which decode ignores. A regression here would nest
	// kv_transfer_params under sampling_params.extra_args instead, the
	// generate-shaped body a chat-completions endpoint does not read.
	mu.Lock()
	decodeBody := capturedDecodeBody
	mu.Unlock()
	if decodeBody == nil {
		t.Fatal("decode was not called")
	}
	if _, ok := decodeBody["kv_transfer_params"]; !ok {
		t.Error("expected top-level kv_transfer_params in decode body for /v1/chat/completions")
	}
	if sp, ok := decodeBody["sampling_params"].(map[string]any); ok {
		if ea, ok := sp["extra_args"].(map[string]any); ok {
			if _, ok := ea["kv_transfer_params"]; ok {
				t.Error("kv_transfer_params must not be nested under sampling_params.extra_args for chat completions")
			}
		}
	}
}

// TestFullPipeline_ResponsesFormat runs a multimodal /v1/responses request
// through the real step chain with stub upstreams. The render service is a stub
// here, which is what makes this runnable: no renderer available to the cluster
// e2e suite serves /v1/responses/render (llm-d/llm-d-inference-sim#708), so the
// chained Responses path has no coverage there.
//
// The request carries two images in the two places a Responses body can hold
// them: one under an input item's content, one under a function_call_output's
// output. That pins the positional contract across the whole chain, since
// replace-media-urls, encode and decode each walk the body independently and
// agree only by visiting the same parts in the same order (content before
// output, per reqcommon.ItemPartArrays).
func TestFullPipeline_ResponsesFormat(t *testing.T) {
	const (
		contentHash = "vllm-hash-content"
		outputHash  = "vllm-hash-output"
		contentData = "data:image/png;base64,Y29udGVudC1pbWFnZQ=="
		outputData  = "data:image/png;base64,b3V0cHV0LWltYWdl"
	)

	var renderPath string
	renderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		renderPath = r.URL.Path
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_ids": []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 2345},
			"features": map[string]any{
				"mm_hashes": map[reqcommon.Modality][]string{reqcommon.ModalityImage: {contentHash, outputHash}},
				"mm_placeholders": map[reqcommon.Modality][]any{reqcommon.ModalityImage: {
					map[string]any{"offset": 1, "length": 3},
					map[string]any{"offset": 4, "length": 3},
				}},
				"kwargs_data": map[reqcommon.Modality][]string{reqcommon.ModalityImage: {"dGVuc29yLWE=", "dGVuc29yLWI="}},
			},
		})
	}))
	defer renderServer.Close()

	var mu sync.Mutex
	var encodeBodies []map[string]any
	var encodePaths []string
	var prefillBody, decodeBody map[string]any

	gatewayServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)

		switch phase := r.Header.Get(reqcommon.EPPProfileHeaderKey); phase {
		case gateway.PhaseEncode:
			// The encode leg speaks the Responses format here, so the image it
			// primes is in the body rather than in a top-level features map. Key
			// the ec params off the hash the fan-out is priming, found by matching
			// the part's URL, so a swap between the two legs fails the assertions
			// below rather than silently matching.
			mu.Lock()
			encodeBodies = append(encodeBodies, parsed)
			encodePaths = append(encodePaths, r.URL.Path)
			mu.Unlock()

			hash := contentHash
			if strings.Contains(string(body), outputData) {
				hash = outputHash
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ec_transfer_params": map[string]any{
					hash: map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501},
				},
			})

		case gateway.PhasePrefill:
			mu.Lock()
			prefillBody = parsed
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{
				"kv_transfer_params": map[string]any{"block_id": "abc123", "peer_host": "10.0.0.2", "peer_port": 5502},
			})

		case gateway.PhaseDecode:
			mu.Lock()
			decodeBody = parsed
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{
				"output": []map[string]any{
					{"type": "message", "role": "assistant", "content": []map[string]any{
						{"type": "output_text", "text": "Two pictures."},
					}},
				},
			})

		default:
			http.Error(w, "unexpected phase: "+phase, http.StatusNotFound)
		}
	}))
	defer gatewayServer.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gatewayServer.URL, MaxIdleConnsPerHost: 10})

	stepConfigs := []config.StepConfig{
		{Type: "replace-media-urls", Params: map[string]any{"download_timeout": "5s"}},
		{Type: "render", Params: map[string]any{}},
		{Type: "encode", Params: map[string]any{"use_openai_format": true, steps.ParamECConnector: ec.NIXL}},
		{Type: "prefill", Params: map[string]any{"use_openai_format": true, steps.ParamKVConnector: kv.NIXL, steps.ParamECConnector: ec.NIXL}},
		{Type: "decode", Params: map[string]any{steps.ParamKVConnector: kv.NIXL}},
	}

	pipelineSteps := make([]pipeline.Step, 0, len(stepConfigs))
	for _, sc := range stepConfigs {
		step, err := pipeline.Build(sc.Type, gwClient, sc.Params)
		if err != nil {
			t.Fatalf("building step %s: %v", sc.Type, err)
		}
		if ra, ok := step.(renderAware); ok {
			ra.SetServiceAddress(renderServer.URL)
		}
		pipelineSteps = append(pipelineSteps, step)
	}

	requestBody := `{
		"model": "test-model", "stream": false, "max_output_tokens": 40,
		"input": [
			{"role": "user", "content": [
				{"type": "input_text", "text": "What is in these?"},
				{"type": "input_image", "image_url": "` + contentData + `", "detail": "auto"}
			]},
			{"type": "function_call_output", "call_id": "call-1", "output": [
				{"type": "input_image", "image_url": "` + outputData + `"}
			]}
		]
	}`

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "responses-chain",
		OriginalPath:     reqcommon.PathResponses,
		OriginalBody:     []byte(requestBody),
		Model:            "test-model",
		KVTransferParams: make(map[string]any),
		ResponseWriter:   recorder,
	}
	if err := json.Unmarshal([]byte(requestBody), &reqCtx.Body); err != nil {
		t.Fatalf("unmarshalling request body: %v", err)
	}

	if err := pipeline.New(pipelineSteps).Execute(t.Context(), reqCtx); err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	respBody, _ := io.ReadAll(recorder.Result().Body)
	if !strings.Contains(string(respBody), "Two pictures.") {
		t.Fatalf("expected the decode response to reach the client, got: %s", respBody)
	}

	if want := reqcommon.PathResponses + "/render"; renderPath != want {
		t.Errorf("render posted to %s, want %s", renderPath, want)
	}

	// render's hashes land on the entries in walk order.
	if len(reqCtx.MultimodalEntries) != 2 {
		t.Fatalf("expected 2 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}
	for i, want := range []string{contentHash, outputHash} {
		if got := reqCtx.MultimodalEntries[i].Hash; got != want {
			t.Errorf("entry %d hash = %q, want %q", i, got, want)
		}
	}

	mu.Lock()
	defer mu.Unlock()

	// One encode sub-request per image, each a Responses body carrying that one
	// image part, capped and with store pinned off.
	if len(encodeBodies) != 2 {
		t.Fatalf("expected 2 encode sub-requests, got %d", len(encodeBodies))
	}
	// The fan-out is concurrent, so the sub-requests are keyed by the image they
	// primed rather than by arrival order. Which hash belongs to which part is
	// pinned by the decode assertions below, not here.
	primed := map[string]map[string]any{}
	for i, body := range encodeBodies {
		if encodePaths[i] != reqcommon.PathResponses {
			t.Errorf("encode sub-request %d posted to %s, want %s", i, encodePaths[i], reqcommon.PathResponses)
		}
		if body["max_output_tokens"] != float64(1) {
			t.Errorf("encode sub-request %d max_output_tokens = %v, want 1", i, body["max_output_tokens"])
		}
		if body["store"] != false {
			t.Errorf("encode sub-request %d store = %v, want false", i, body["store"])
		}
		if _, ok := body["messages"]; ok {
			t.Errorf("encode sub-request %d must not carry a messages array", i)
		}
		part := encodeImagePart(t, body)
		url, ok := part[reqcommon.FieldImageURL].(string)
		if !ok {
			t.Fatalf("encode sub-request %d part carries no string image_url: %v", i, part)
		}
		if _, dup := primed[url]; dup {
			t.Errorf("image %q was primed more than once", url)
		}
		primed[url] = part
	}

	// Both images are primed, exactly once each.
	for _, want := range []string{contentData, outputData} {
		if _, ok := primed[want]; !ok {
			t.Errorf("no encode sub-request primed %q", want)
		}
	}
	// The part goes out unreshaped, so the client's detail sibling comes along.
	if part, ok := primed[contentData]; ok && part["detail"] != "auto" {
		t.Errorf("encode dropped the client's detail sibling, got %v", part["detail"])
	}

	// Prefill clones the client's Responses body, caps it, and carries both
	// transfer params.
	if prefillBody == nil {
		t.Fatal("prefill was not called")
	}
	if prefillBody["max_output_tokens"] != float64(1) {
		t.Errorf("prefill max_output_tokens = %v, want 1", prefillBody["max_output_tokens"])
	}
	if prefillBody["store"] != false {
		t.Errorf("prefill store = %v, want false", prefillBody["store"])
	}
	if _, ok := prefillBody["input"].([]any); !ok {
		t.Error("prefill body must carry the client's input array")
	}
	for _, field := range []string{"kv_transfer_params", "ec_transfer_params"} {
		if _, ok := prefillBody[field]; !ok {
			t.Errorf("prefill body missing %s", field)
		}
	}
	if len(reqCtx.ECTransferParams) != 2 {
		t.Errorf("expected ec params merged for both images, got %d", len(reqCtx.ECTransferParams))
	}

	// Decode forwards the client's own body, uncapped, with each image part
	// stamped with its entry's hash. The output-array image getting outputHash is
	// what proves the three walks agreed on the order.
	if decodeBody == nil {
		t.Fatal("decode was not called")
	}
	if decodeBody["max_output_tokens"] != float64(40) {
		t.Errorf("decode max_output_tokens = %v, want the client's 40", decodeBody["max_output_tokens"])
	}
	input := decodeBody["input"].([]any)
	contentPart := input[0].(map[string]any)["content"].([]any)[1].(map[string]any)
	if contentPart["uuid"] != contentHash {
		t.Errorf("content image uuid = %v, want %q", contentPart["uuid"], contentHash)
	}
	outputPart := input[1].(map[string]any)["output"].([]any)[0].(map[string]any)
	if outputPart["uuid"] != outputHash {
		t.Errorf("output-array image uuid = %v, want %q", outputPart["uuid"], outputHash)
	}
}

// encodeImagePart returns the single image part an encode priming body carries.
func encodeImagePart(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	input, ok := body[reqcommon.FieldInput].([]any)
	if !ok || len(input) != 1 {
		t.Fatalf("encode body has no single-item input array: %v", body[reqcommon.FieldInput])
	}
	content, ok := input[0].(map[string]any)[reqcommon.FieldContent].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("encode body input item has no single content part: %v", input[0])
	}
	part, ok := content[0].(map[string]any)
	if !ok {
		t.Fatalf("encode body content part is not an object: %v", content[0])
	}
	return part
}
