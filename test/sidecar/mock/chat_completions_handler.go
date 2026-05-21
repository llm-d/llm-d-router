/*
Copyright 2025 The llm-d Authors.

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

// Package mock offers a mock CompletionHandler for tests
package mock

import (
	"encoding/json"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
)

// Role of the mocked handler
type Role string

const (
	// RoleDecode indicates the handler is a decoder
	RoleDecode Role = "decode"

	// RolePrefill indicates the handler is a prefiller
	RolePrefill Role = "prefill"

	contentTypeEventStream = "text/event-stream"
)

// ChatCompletionHandler is a simple chat completion mock handler
type ChatCompletionHandler struct {
	Connector           string
	Role                Role
	RawResponse         string
	RawResponseType     string
	RequestCount        atomic.Int32
	CompletionRequests  []map[string]any
	CompletionResponses []map[string]any
	mu                  sync.Mutex
}

// GetCompletionRequests returns a snapshot of the received requests, safe for concurrent access.
func (cc *ChatCompletionHandler) GetCompletionRequests() []map[string]any {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return append([]map[string]any(nil), cc.CompletionRequests...)
}

func (cc *ChatCompletionHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	cc.RequestCount.Add(1)

	defer r.Body.Close() //nolint:all
	b, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest) // TODO: check FastAPI error code when failing to read body
		w.Write([]byte(err.Error()))         //nolint:all
		return
	}

	var completionRequest map[string]any
	if err := json.Unmarshal(b, &completionRequest); err != nil {
		w.Write([]byte(err.Error())) //nolint:all
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	cc.mu.Lock()
	cc.CompletionRequests = append(cc.CompletionRequests, completionRequest)
	cc.mu.Unlock()

	var rawResponse string

	switch cc.Connector {
	case "nixl":
		rawResponse = `{"remote_block_ids":[1, 2, 3], "remote_engine_id": "5b5fb28f-3f30-4bdd-9a36-958d52459200"}`

	case "nixlv2":
		switch cc.Role {
		case RoleDecode:
			rawResponse = `{"id":"chatcmpl-test","object":"chat.completion","choices":[],"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65,"prompt_tokens_details":{"cached_tokens":49}}}`
		case RolePrefill:

			// 1. Verify Prefill Request
			kvTransferParams, ok := completionRequest["kv_transfer_params"]

			if !ok || kvTransferParams == nil {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected kv_transfer_params:{...}")) //nolint:all
				return
			}
			kvTransferParamsMap, ok := kvTransferParams.(map[string]any)
			if !ok {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected kv_transfer_params:{...}")) //nolint:all
				return
			}

			if v, ok := kvTransferParamsMap["do_remote_decode"]; !ok || !v.(bool) {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected do_remote_decode:true")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["do_remote_prefill"]; !ok || v.(bool) {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected do_remote_prefill:false")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["remote_engine_id"]; !ok || v != nil {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected remote_engine_id:null")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["remote_block_ids"]; !ok || v != nil {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected remote_block_ids:null")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["remote_host"]; !ok || v != nil {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected remote_host:null")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["remote_port"]; !ok || v != nil {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected remote_port:null")) //nolint:all
				return
			}

			// 2. Produce Response

			rawResponse = `{"kv_transfer_params":{"remote_block_ids":[1, 2, 3], "remote_engine_id": "5b5fb28f-3f30-4bdd-9a36-958d52459200", "remote_host":"ahost", "remote_port":4032},"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65,"prompt_tokens_details":{"cached_tokens":7}}}`

		}

	case "shared-storage":
		// Shared Storage protocol just returns empty response
		rawResponse = `{}`

	default:
		// Default case for unspecified connector (used for basic tests)
		rawResponse = `{}`
	}

	if cc.RawResponse != "" {
		rawResponse = cc.RawResponse
	}
	if cc.RawResponseType != "" {
		w.Header().Set("Content-Type", cc.RawResponseType)
	}

	var completionResponse map[string]any
	if cc.RawResponseType != contentTypeEventStream {
		if err := json.Unmarshal([]byte(rawResponse), &completionResponse); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(err.Error())) //nolint:all
			return
		}
		cc.mu.Lock()
		cc.CompletionResponses = append(cc.CompletionResponses, completionResponse)
		cc.mu.Unlock()
	}

	w.Write([]byte(rawResponse)) //nolint:all
}
