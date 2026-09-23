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
	"reflect"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func TestSGLangKV_NoKVTransferParams(t *testing.T) {
	c, err := Build(SGLang)
	if err != nil {
		t.Fatalf("Build(%q): %v", SGLang, err)
	}
	if c.Name() != SGLang {
		t.Fatalf("Name() = %q, want %q", c.Name(), SGLang)
	}
	reqCtx := &pipeline.RequestContext{KVTransferParams: map[string]any{"ignored": "field"}}
	if got := c.PreparePrefillKVParams(context.Background(), reqCtx); got != nil {
		t.Errorf("prefill params = %v, want nil", got)
	}
	if got := c.PrepareDecodeKVParams(context.Background(), reqCtx); got != nil {
		t.Errorf("decode params = %v, want nil", got)
	}
}

func TestSGLangKV_ApplyBootstrapFields(t *testing.T) {
	c, err := Build(SGLang)
	if err != nil {
		t.Fatalf("Build(%q): %v", SGLang, err)
	}
	concurrent, ok := c.(ConcurrentConnector)
	if !ok {
		t.Fatalf("%q is not a ConcurrentConnector", SGLang)
	}

	prefill := map[string]any{"model": "m", "routed_dp_rank": 1, "data_parallel_rank": 2}
	decode := map[string]any{"model": "m", "disagg_prefill_dp_rank": 3, "stream": true}
	if err := concurrent.ApplyBootstrapFields(context.Background(), "10.0.3.7:8000", prefill, decode); err != nil {
		t.Fatalf("ApplyBootstrapFields: %v", err)
	}

	room, ok := prefill[fieldBootstrapRoom].(int64)
	if !ok || room < 0 {
		t.Fatalf("%s = %v (%T), want a non-negative int64", fieldBootstrapRoom, prefill[fieldBootstrapRoom], prefill[fieldBootstrapRoom])
	}
	for name, body := range map[string]map[string]any{"prefill": prefill, "decode": decode} {
		if body[fieldBootstrapHost] != "10.0.3.7" {
			t.Errorf("%s: %s = %v, want 10.0.3.7", name, fieldBootstrapHost, body[fieldBootstrapHost])
		}
		if body[fieldBootstrapPort] != resolveSGLangBootstrapPort(context.Background()) {
			t.Errorf("%s: %s = %v, want the resolved bootstrap port", name, fieldBootstrapPort, body[fieldBootstrapPort])
		}
		if body[fieldBootstrapRoom] != room {
			t.Errorf("%s: %s = %v, want the same room %d on both bodies", name, fieldBootstrapRoom, body[fieldBootstrapRoom], room)
		}
		for _, field := range sglangRankFields {
			if _, present := body[field]; present {
				t.Errorf("%s: client rank field %q was not removed", name, field)
			}
		}
		if _, present := body[reqcommon.FieldKVTransferParams]; present {
			t.Errorf("%s: kv_transfer_params must not be set", name)
		}
	}
	if decode["stream"] != true {
		t.Errorf("decode: unrelated field changed: stream = %v", decode["stream"])
	}
}

func TestSGLangKV_ApplyBootstrapFieldsHost(t *testing.T) {
	tests := []struct {
		hostPort string
		wantHost string
		wantErr  bool
	}{
		{hostPort: "10.0.3.7:8000", wantHost: "10.0.3.7"},
		{hostPort: "[fd00::7]:8000", wantHost: "fd00::7"},
		{hostPort: "", wantErr: true},
		{hostPort: "10.0.3.7", wantErr: true},
		{hostPort: "10.0.3.7:8000:1", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.hostPort, func(t *testing.T) {
			body := map[string]any{"model": "m"}
			err := (sglangKV{}).ApplyBootstrapFields(context.Background(), tt.hostPort, body)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				if _, present := body[fieldBootstrapHost]; present {
					t.Error("body changed on error")
				}
				return
			}
			if err != nil {
				t.Fatalf("ApplyBootstrapFields: %v", err)
			}
			if body[fieldBootstrapHost] != tt.wantHost {
				t.Errorf("%s = %v, want %s", fieldBootstrapHost, body[fieldBootstrapHost], tt.wantHost)
			}
		})
	}
}

func TestSGLangKV_ApplyBootstrapFieldsNewRoomPerCall(t *testing.T) {
	a, b := map[string]any{}, map[string]any{}
	for _, body := range []map[string]any{a, b} {
		if err := (sglangKV{}).ApplyBootstrapFields(context.Background(), "10.0.3.7:8000", body); err != nil {
			t.Fatalf("ApplyBootstrapFields: %v", err)
		}
	}
	if a[fieldBootstrapRoom] == b[fieldBootstrapRoom] {
		t.Errorf("two requests got the same room %v", a[fieldBootstrapRoom])
	}
}

func TestSerialConnectorsAreNotConcurrent(t *testing.T) {
	for _, name := range []string{NIXL, SharedStorage} {
		c, err := Build(name)
		if err != nil {
			t.Fatalf("Build(%q): %v", name, err)
		}
		if _, ok := c.(ConcurrentConnector); ok {
			t.Errorf("%q must not be a ConcurrentConnector", name)
		}
	}
}

func TestBuild_UnknownReturnsError(t *testing.T) {
	if _, err := Build("does-not-exist"); err == nil {
		t.Fatal("expected error for unknown connector")
	}
}

func TestBuild_EmptyReturnsDefault(t *testing.T) {
	c, err := Build("")
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
			c, err := Build(tc.name)
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
