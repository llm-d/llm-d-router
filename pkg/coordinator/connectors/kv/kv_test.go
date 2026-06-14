package kv

import (
	"reflect"
	"testing"

	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestSGLangKV_Params(t *testing.T) {
	c, err := Build(SGLang)
	if err != nil {
		t.Fatalf("Build(%q): %v", SGLang, err)
	}
	if c.Name() != SGLang {
		t.Fatalf("Name() = %q, want %q", c.Name(), SGLang)
	}

	reqCtx := &pipeline.RequestContext{
		KVTransferParams: map[string]any{
			fieldBootstrapHost: "10.0.0.42",
			fieldBootstrapPort: 8998,
			fieldBootstrapRoom: int64(12345),
		},
	}

	// Prefill: must have the required bootstrap fields; bootstrap_room is random so check type.
	prefill := c.PreparePrefillKVParams(reqCtx)
	if prefill["do_remote_decode"] != true {
		t.Errorf("prefill: do_remote_decode = %v, want true", prefill["do_remote_decode"])
	}
	if prefill["do_remote_prefill"] != false {
		t.Errorf("prefill: do_remote_prefill = %v, want false", prefill["do_remote_prefill"])
	}
	if prefill[fieldBootstrapPort] != sglangBootstrapPort {
		t.Errorf("prefill: %s = %v, want %d", fieldBootstrapPort, prefill[fieldBootstrapPort], sglangBootstrapPort)
	}
	room, ok := prefill[fieldBootstrapRoom].(string)
	if !ok || room == "" {
		t.Errorf("prefill: %s = %v (%T), want non-empty string", fieldBootstrapRoom, prefill[fieldBootstrapRoom], prefill[fieldBootstrapRoom])
	}

	// Decode: forwards prefill-response kv_transfer_params plus remote flags.
	wantDecode := map[string]any{
		fieldBootstrapHost:  "10.0.0.42",
		fieldBootstrapPort:  8998,
		fieldBootstrapRoom:  int64(12345),
		"do_remote_decode":  false,
		"do_remote_prefill": true,
	}
	if got := c.PrepareDecodeKVParams(reqCtx); !reflect.DeepEqual(got, wantDecode) {
		t.Errorf("decode params:\n got=%v\nwant=%v", got, wantDecode)
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

			if got := c.PreparePrefillKVParams(reqCtx); !reflect.DeepEqual(got, tc.wantPrefill) {
				t.Errorf("prefill params:\n got=%v\nwant=%v", got, tc.wantPrefill)
			}
			if got := c.PrepareDecodeKVParams(reqCtx); !reflect.DeepEqual(got, tc.wantDecode) {
				t.Errorf("decode params:\n got=%v\nwant=%v", got, tc.wantDecode)
			}
		})
	}
}
