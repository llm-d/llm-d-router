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

	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
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
	Connector       string
	Role            Role
	RawResponse     string
	RawResponseType string
	// MoRIIOWriteMode loosens the NIXLv2 prefill-side validation to allow
	// non-nil remote_host / remote_notify_port / transfer_id fields that the
	// sidecar populates when --moriio-write-mode is enabled.  Standard NIXLv2
	// READ-mode validation (everything nil) still applies when this is false.
	MoRIIOWriteMode     bool
	RequestCount        atomic.Int32
	CompletionRequests  []map[string]any
	CompletionHeaders   []http.Header
	CompletionResponses []map[string]any
	mu                  sync.Mutex

	// FailForFirstN makes the handler return FailStatusCode for
	// the first N requests, then proceed normally. 0 means never fail.
	FailForFirstN  int32
	FailStatusCode int
}

// GetCompletionRequests returns a snapshot of the received requests, safe for concurrent access.
func (cc *ChatCompletionHandler) GetCompletionRequests() []map[string]any {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return append([]map[string]any(nil), cc.CompletionRequests...)
}

// GetCompletionHeaders returns a snapshot of the received request headers,
// appended in lockstep with GetCompletionRequests, safe for concurrent access.
func (cc *ChatCompletionHandler) GetCompletionHeaders() []http.Header {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return append([]http.Header(nil), cc.CompletionHeaders...)
}

func (cc *ChatCompletionHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	count := cc.RequestCount.Add(1)

	if cc.FailForFirstN > 0 && count <= cc.FailForFirstN {
		code := cc.FailStatusCode
		if code == 0 {
			code = http.StatusBadGateway
		}
		w.WriteHeader(code)
		return
	}

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
	cc.CompletionHeaders = append(cc.CompletionHeaders, r.Header.Clone())
	cc.mu.Unlock()

	var rawResponse string

	switch cc.Connector {
	case constants.KVConnectorNIXLV2:
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
			if cc.MoRIIOWriteMode {
				// WRITE-mode expectations: remote_host is a non-empty string
				// (decode pod hostname), remote_notify_port is a number
				// matching the sidecar config, and a transfer_id is present.
				v, ok := kvTransferParamsMap["remote_host"]
				if !ok {
					w.WriteHeader(http.StatusBadRequest)
					w.Write([]byte("expected remote_host:<host>")) //nolint:all
					return
				}
				if s, isStr := v.(string); !isStr || s == "" {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected remote_host to be a non-empty string in WRITE mode")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["remote_notify_port"]; !ok {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected remote_notify_port:<int> in WRITE mode")) //nolint:all
				return
			} else if _, isNum := v.(float64); !isNum {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected remote_notify_port to be a number in WRITE mode")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["transfer_id"]; !ok {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected transfer_id:<uuid> in WRITE mode")) //nolint:all
				return
			} else if s, isStr := v.(string); !isStr || s == "" {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected transfer_id to be a non-empty string in WRITE mode")) //nolint:all
				return
			}
			} else {
				if v, ok := kvTransferParamsMap["remote_host"]; !ok || v != nil {
					w.WriteHeader(http.StatusBadRequest)
					w.Write([]byte("expected remote_host:null")) //nolint:all
					return
				}
			}
			if v, ok := kvTransferParamsMap["remote_port"]; !ok || v != nil {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected remote_port:null")) //nolint:all
				return
			}

			// 2. Produce Response

			rawResponse = `{"kv_transfer_params":{"remote_block_ids":[1, 2, 3], "remote_engine_id": "5b5fb28f-3f30-4bdd-9a36-958d52459200", "remote_host":"ahost", "remote_port":4032},"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65,"prompt_tokens_details":{"cached_tokens":7}}}`

		}

	case constants.KVConnectorMooncake:
		switch cc.Role {
		case RoleDecode:
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
			if v, ok := kvTransferParamsMap["do_remote_prefill"]; !ok || !v.(bool) {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected do_remote_prefill:true")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["do_remote_decode"]; !ok || v.(bool) {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected do_remote_decode:false")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["transfer_id"]; !ok || v == nil || v == "" {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected transfer_id to be non-empty")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["remote_bootstrap_addr"]; !ok || v == nil || v == "" {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected remote_bootstrap_addr to be non-empty")) //nolint:all
				return
			}
			if v, ok := kvTransferParamsMap["remote_engine_id"]; !ok || v == nil || v == "" {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected remote_engine_id to be non-empty")) //nolint:all
				return
			}
			rawResponse = `{"id":"chatcmpl-test","object":"chat.completion","choices":[],"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65}}`
		case RolePrefill:
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
			if v, ok := kvTransferParamsMap["transfer_id"]; !ok || v == nil || v == "" {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("expected transfer_id to be non-empty")) //nolint:all
				return
			}
			rawResponse = `{}`
		}

	case constants.KVConnectorSharedStorage:
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
