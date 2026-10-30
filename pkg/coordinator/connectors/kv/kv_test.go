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

package kv

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func TestSGLangKV_Params(t *testing.T) {
	t.Setenv(envSGLangBootstrapPort, "")
	c, err := Build(SGLang, nil)
	if err != nil {
		t.Fatalf("Build(%q): %v", SGLang, err)
	}
	if c.Name() != SGLang {
		t.Fatalf("Name() = %q, want %q", c.Name(), SGLang)
	}

	reqCtx := &pipeline.RequestContext{
		KVTransferParams: map[string]any{
			reqcommon.FieldBootstrapHost: "10.0.0.42",
			reqcommon.FieldBootstrapPort: 8998,
			reqcommon.FieldBootstrapRoom: int64(12345),
		},
	}

	// Prefill: must have the required bootstrap fields; bootstrap_room is random so check type.
	prefill := c.PreparePrefillKVParams(context.Background(), reqCtx)
	if prefill["do_remote_decode"] != true {
		t.Errorf("prefill: do_remote_decode = %v, want true", prefill["do_remote_decode"])
	}
	if prefill["do_remote_prefill"] != false {
		t.Errorf("prefill: do_remote_prefill = %v, want false", prefill["do_remote_prefill"])
	}
	if prefill[reqcommon.FieldBootstrapPort] != defaultSGLangBootstrapPort {
		t.Errorf("prefill: %s = %v, want %d", reqcommon.FieldBootstrapPort, prefill[reqcommon.FieldBootstrapPort], defaultSGLangBootstrapPort)
	}
	room, ok := prefill[reqcommon.FieldBootstrapRoom].(string)
	if !ok || room == "" {
		t.Errorf("prefill: %s = %v (%T), want non-empty string", reqcommon.FieldBootstrapRoom, prefill[reqcommon.FieldBootstrapRoom], prefill[reqcommon.FieldBootstrapRoom])
	}

	// Decode: forwards prefill-response kv_transfer_params plus remote flags.
	wantDecode := map[string]any{
		reqcommon.FieldBootstrapHost: "10.0.0.42",
		reqcommon.FieldBootstrapPort: 8998,
		reqcommon.FieldBootstrapRoom: int64(12345),
		"do_remote_decode":           false,
		"do_remote_prefill":          true,
	}
	if got := c.PrepareDecodeKVParams(context.Background(), reqCtx); !reflect.DeepEqual(got, wantDecode) {
		t.Errorf("decode params:\n got=%v\nwant=%v", got, wantDecode)
	}
}

func TestBuild_UnknownReturnsError(t *testing.T) {
	if _, err := Build("does-not-exist", nil); err == nil {
		t.Fatal("expected error for unknown connector")
	}
}

func TestBuild_EmptyReturnsDefault(t *testing.T) {
	c, err := Build("", nil)
	if err != nil {
		t.Fatal(err)
	}
	if c.Name() != DefaultKVConnectorName {
		t.Fatalf("default = %q, want %q", c.Name(), DefaultKVConnectorName)
	}
}

func TestConnectors_KVParams(t *testing.T) {
	cases := []struct {
		name           string
		decodeIncoming map[string]any
		wantPrefill    map[string]any
		wantDecode     map[string]any
	}{
		{
			name: NIXL,
			decodeIncoming: map[string]any{
				"block_id":  "block-999",
				"peer_host": "10.0.0.42",
				"peer_port": float64(7777),
			},
			wantPrefill: map[string]any{
				"do_remote_decode":  true,
				"do_remote_prefill": false,
				"remote_engine_id":  nil,
				"remote_block_ids":  nil,
				"remote_host":       nil,
				"remote_port":       nil,
			},
			wantDecode: map[string]any{
				"do_remote_decode":  false,
				"do_remote_prefill": true,
				"block_id":          "block-999",
				"peer_host":         "10.0.0.42",
				"peer_port":         float64(7777),
			},
		},
		{
			name:           SharedStorage,
			decodeIncoming: map[string]any{"ignored": "field"},
			wantPrefill:    map[string]any{"do_remote_decode": true, "do_remote_prefill": false},
			wantDecode:     map[string]any{"do_remote_decode": false, "do_remote_prefill": true},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := Build(tc.name, nil)
			if err != nil {
				t.Fatalf("Build(%q): %v", tc.name, err)
			}
			if c.Name() != tc.name {
				t.Fatalf("Name() = %q, want %q", c.Name(), tc.name)
			}

			reqCtx := &pipeline.RequestContext{KVTransferParams: tc.decodeIncoming}

			if got := c.PreparePrefillKVParams(context.Background(), reqCtx); !reflect.DeepEqual(got, tc.wantPrefill) {
				t.Errorf("prefill params:\n got=%v\nwant=%v", got, tc.wantPrefill)
			}
			if got := c.PrepareDecodeKVParams(context.Background(), reqCtx); !reflect.DeepEqual(got, tc.wantDecode) {
				t.Errorf("decode params:\n got=%v\nwant=%v", got, tc.wantDecode)
			}
		})
	}
}

func TestBuild_SGLangBootstrapPortParam(t *testing.T) {
	const invalidPort = "kv_connector_params.bootstrap_port: invalid port"
	tests := []struct {
		name        string
		value       any
		want        int
		errContains string
	}{
		{name: "int", value: 9100, want: 9100},
		{name: "int64", value: int64(9100), want: 9100},
		{name: "integral float64", value: float64(9100), want: 9100},
		{name: "json.Number", value: json.Number("9100"), want: 9100},
		{name: "numeric string", value: "9100", want: 9100},
		{name: "min valid", value: 1, want: 1},
		{name: "max valid", value: 65535, want: 65535},
		{name: "zero", value: 0, errContains: invalidPort},
		{name: "negative", value: -1, errContains: invalidPort},
		{name: "above range", value: 65536, errContains: invalidPort},
		{name: "above range string", value: "65536", errContains: invalidPort},
		{name: "non-numeric string", value: "abc", errContains: invalidPort},
		{name: "empty string", value: "", errContains: invalidPort},
		{name: "fractional float64", value: 9100.5, errContains: invalidPort},
		{name: "bool", value: true, errContains: invalidPort},
		{name: "nil", value: nil, errContains: invalidPort},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := Build(SGLang, map[string]any{paramBootstrapPort: tt.value})
			if tt.errContains != "" {
				if err == nil || !strings.Contains(err.Error(), tt.errContains) {
					t.Fatalf("Build() = %v, %v; want error containing %q", c, err, tt.errContains)
				}
				return
			}
			if err != nil {
				t.Fatalf("Build(): %v", err)
			}
			got := c.PreparePrefillKVParams(context.Background(), &pipeline.RequestContext{})[reqcommon.FieldBootstrapPort]
			if got != tt.want {
				t.Errorf("%s = %v, want %d", reqcommon.FieldBootstrapPort, got, tt.want)
			}
		})
	}
}

func TestBuild_SGLangRejectsUnknownParam(t *testing.T) {
	_, err := Build(SGLang, map[string]any{"bootstrap_prot": 9100})
	if err == nil || !strings.Contains(err.Error(), `unknown key "bootstrap_prot"`) {
		t.Fatalf("Build() error = %v, want unknown key error", err)
	}
}

func TestBuild_ParamsRejectedForParamlessConnectors(t *testing.T) {
	for _, name := range []string{NIXL, SharedStorage, ""} {
		t.Run("connector="+name, func(t *testing.T) {
			if _, err := Build(name, map[string]any{"x": 1}); err == nil || !strings.Contains(err.Error(), "takes no parameters") {
				t.Errorf("Build() with params: error = %v, want it to contain %q", err, "takes no parameters")
			}
			if _, err := Build(name, map[string]any{}); err != nil {
				t.Errorf("Build() with empty params: %v", err)
			}
		})
	}
}
