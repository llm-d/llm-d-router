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
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// TestPrefillStep_ConnectorShapesPrefillBody verifies that the connector
// selected via params controls the kv_transfer_params shape on the prefill
// request. In generate format, kv_transfer_params lives inside
// sampling_params.extra_args due to the vLLM workaround.
func TestPrefillStep_ConnectorShapesPrefillBody(t *testing.T) {
	cases := []struct {
		connector  string
		wantFields map[string]any
		denyFields []string
	}{
		{
			connector: kv.NIXL,
			wantFields: map[string]any{
				"do_remote_decode":  true,
				"do_remote_prefill": false,
				"remote_engine_id":  nil,
				"remote_block_ids":  nil,
				"remote_host":       nil,
				"remote_port":       nil,
			},
		},
		{
			connector:  kv.SharedStorage,
			wantFields: map[string]any{"do_remote_decode": true, "do_remote_prefill": false},
			denyFields: []string{"remote_engine_id", "remote_host", "remote_block_ids", "remote_port"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.connector, func(t *testing.T) {
			var captured map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				var parsed map[string]any
				_ = json.Unmarshal(body, &parsed)
				// In generate format, kv_transfer_params is in sampling_params.extra_args
				samplingParams, _ := parsed["sampling_params"].(map[string]any)
				extraArgs, _ := samplingParams["extra_args"].(map[string]any)
				captured, _ = extraArgs["kv_transfer_params"].(map[string]any)
				_ = json.NewEncoder(w).Encode(map[string]any{"kv_transfer_params": map[string]any{}})
			}))
			defer srv.Close()

			gwClient := gateway.New(config.GatewayConfig{Address: srv.URL})
			step, err := NewPrefillStep(gwClient, map[string]any{
				"use_openai_format": false,
				ParamKVConnector:    tc.connector,
			})
			if err != nil {
				t.Fatalf("NewPrefillStep: %v", err)
			}

			reqCtx := &pipeline.RequestContext{
				RequestID:        "req",
				Model:            "m",
				TokenIDs:         []int{1, 2},
				KVTransferParams: make(map[string]any),
			}
			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("Execute: %v", err)
			}

			if captured == nil {
				t.Fatal("kv_transfer_params not found in sampling_params.extra_args")
			}
			for f, want := range tc.wantFields {
				got, ok := captured[f]
				if !ok {
					t.Errorf("missing field %q; got %v", f, captured)
					continue
				}
				if got != want {
					t.Errorf("field %q = %v, want %v", f, got, want)
				}
			}
			for _, f := range tc.denyFields {
				if _, ok := captured[f]; ok {
					t.Errorf("unexpected field %q; got %v", f, captured)
				}
			}
		})
	}
}

// TestDecodeStep_ConnectorShapesDecodeBody verifies the per-connector
// kv_transfer_params shape sent on the decode request. kv-nixl forwards all
// prefill response fields and overrides do_remote_decode/do_remote_prefill;
// kv-shared-storage emits only the two flags.
func TestDecodeStep_ConnectorShapesDecodeBody(t *testing.T) {
	// nixlPrefillResponse simulates the kv_transfer_params returned by a
	// nixl prefill worker (remote addressing fields filled in).
	nixlPrefillResponse := map[string]any{
		"do_remote_decode":  true,
		"do_remote_prefill": false,
		"remote_engine_id":  "e95b1c63-2ba6-4f26-96d0-9338d40a2560",
		"remote_block_ids":  []any{[]any{float64(1)}},
		"remote_request_id": "generate-tokens-550e8400-e29b-41d4-a716-446655440000",
		"remote_host":       "10.130.5.242",
		"remote_port":       float64(5557),
		"tp_size":           float64(2),
	}

	cases := []struct {
		connector       string
		prefillResponse map[string]any
		wantFields      map[string]any
		denyFields      []string
	}{
		{
			connector:       kv.NIXL,
			prefillResponse: nixlPrefillResponse,
			wantFields: map[string]any{
				"do_remote_decode":  false,
				"do_remote_prefill": true,
				"remote_engine_id":  "e95b1c63-2ba6-4f26-96d0-9338d40a2560",
				"remote_request_id": "generate-tokens-550e8400-e29b-41d4-a716-446655440000",
				"remote_host":       "10.130.5.242",
				"remote_port":       float64(5557),
				"tp_size":           float64(2),
			},
		},
		{
			connector:       kv.SharedStorage,
			prefillResponse: map[string]any{"ignored": "field"},
			wantFields:      map[string]any{"do_remote_decode": false, "do_remote_prefill": true},
			denyFields:      []string{"remote_engine_id", "remote_host", "remote_block_ids", "remote_port", "ignored"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.connector, func(t *testing.T) {
			var captured map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				var parsed map[string]any
				_ = json.Unmarshal(body, &parsed)
				captured, _ = parsed["kv_transfer_params"].(map[string]any)
				_ = json.NewEncoder(w).Encode(map[string]any{
					"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ok"}}},
				})
			}))
			defer srv.Close()

			gwClient := gateway.New(config.GatewayConfig{Address: srv.URL})
			step, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: tc.connector})
			if err != nil {
				t.Fatalf("NewDecodeStep: %v", err)
			}

			recorder := httptest.NewRecorder()
			reqCtx := &pipeline.RequestContext{
				RequestID:        "req",
				OriginalPath:     gateway.PathChatCompletions,
				Model:            "m",
				KVTransferParams: tc.prefillResponse,
				Body:             map[string]any{"model": "m"},
				ResponseWriter:   recorder,
			}
			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("Execute: %v", err)
			}

			if captured == nil {
				t.Fatal("kv_transfer_params not sent to gateway")
			}
			for k, want := range tc.wantFields {
				if got := captured[k]; got != want {
					t.Errorf("field %q = %v, want %v", k, got, want)
				}
			}
			for _, f := range tc.denyFields {
				if _, ok := captured[f]; ok {
					t.Errorf("unexpected field %q; got %v", f, captured)
				}
			}
		})
	}
}

func TestPrefillStep_UnknownConnectorRejected(t *testing.T) {
	gwClient := gateway.New(config.GatewayConfig{})
	if _, err := NewPrefillStep(gwClient, map[string]any{ParamKVConnector: "bogus"}); err == nil {
		t.Fatal("expected error for unknown connector")
	}
}

func TestDecodeStep_UnknownConnectorRejected(t *testing.T) {
	gwClient := gateway.New(config.GatewayConfig{})
	if _, err := NewDecodeStep(gwClient, map[string]any{ParamKVConnector: "bogus"}); err == nil {
		t.Fatal("expected error for unknown connector")
	}
}
