package kv

import (
	"reflect"
	"testing"

	"github.com/llm-d/coordinator/pkg/connector"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestSGLangKV_Params(t *testing.T) {
	c, err := Build(connector.KVSGLang)
	if err != nil {
		t.Fatalf("Build(%q): %v", connector.KVSGLang, err)
	}
	if c.Name() != connector.KVSGLang {
		t.Fatalf("Name() = %q, want %q", c.Name(), connector.KVSGLang)
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
	if prefill[fieldBootstrapPort] != sglangBootstrapPort {
		t.Errorf("prefill: %s = %v, want %d", fieldBootstrapPort, prefill[fieldBootstrapPort], sglangBootstrapPort)
	}
	room, ok := prefill[fieldBootstrapRoom].(int64)
	if !ok || room == 0 {
		t.Errorf("prefill: %s = %v (%T), want non-zero int64", fieldBootstrapRoom, prefill[fieldBootstrapRoom], prefill[fieldBootstrapRoom])
	}

	// Decode: forwards prefill-response kv_transfer_params verbatim.
	wantDecode := map[string]any{
		fieldBootstrapHost: "10.0.0.42",
		fieldBootstrapPort: 8998,
		fieldBootstrapRoom: int64(12345),
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
			name: connector.KVNIXLv2,
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
				"do_remote_prefill": true,
				"block_id":          "block-999",
				"peer_host":         "10.0.0.42",
				"peer_port":         float64(7777),
			},
		},
		{
			name:           connector.KVSharedStorage,
			decodeIncoming: map[string]any{"ignored": "field"},
			wantPrefill:    map[string]any{"do_remote_decode": true},
			wantDecode:     map[string]any{"do_remote_prefill": true},
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
