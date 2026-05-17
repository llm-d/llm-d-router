package ec

import (
	"reflect"
	"testing"

	"github.com/llm-d/coordinator/pkg/connector"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func TestBuild_UnknownReturnsError(t *testing.T) {
	if _, err := Build("does-not-exist"); err == nil {
		t.Fatal("expected error for unknown ec_connector")
	}
}

func TestBuild_EmptyReturnsDefault(t *testing.T) {
	c, err := Build("")
	if err != nil {
		t.Fatal(err)
	}
	if c.Name() != DefaultECConnectorName {
		t.Fatalf("default = %q, want %q", c.Name(), DefaultECConnectorName)
	}
}

func TestBuild_NamedConnectors(t *testing.T) {
	for _, name := range []string{connector.NameNIXLv2, connector.NameSharedStorage} {
		t.Run(name, func(t *testing.T) {
			c, err := Build(name)
			if err != nil {
				t.Fatalf("Build(%q): %v", name, err)
			}
			if c.Name() != name {
				t.Fatalf("Name() = %q, want %q", c.Name(), name)
			}
		})
	}
}

// TestNIXL_MergeAndPrepare verifies that nixl appends each per-image encode
// response in order and emits a per-modality wrapped list on the prefill
// request: {"image": [{hash1: ...}, {hash2: ...}]}.
func TestNIXL_MergeAndPrepare(t *testing.T) {
	c, err := Build(connector.NameNIXLv2)
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{}

	if got := c.PreparePrefillECParams(reqCtx); got != nil {
		t.Fatalf("expected nil ec_transfer_params before encodes, got %v", got)
	}

	resp1 := map[string]any{"hash-a": map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501}}
	resp2 := map[string]any{"hash-b": map[string]any{"peer_host": "10.0.0.2", "peer_port": 5502}}

	c.MergeEncodeResponse(reqCtx, resp1)
	c.MergeEncodeResponse(reqCtx, resp2)

	if len(reqCtx.ECTransferParams) != 2 {
		t.Fatalf("expected 2 entries in ECTransferParams, got %d", len(reqCtx.ECTransferParams))
	}

	want := map[string]any{
		"image": []map[string]any{
			{"hash-a": map[string]any{"peer_host": "10.0.0.1", "peer_port": 5501}},
			{"hash-b": map[string]any{"peer_host": "10.0.0.2", "peer_port": 5502}},
		},
	}
	if got := c.PreparePrefillECParams(reqCtx); !reflect.DeepEqual(got, want) {
		t.Errorf("prefill ec_transfer_params:\n got=%v\nwant=%v", got, want)
	}
}

// TestNIXL_MergeIgnoresEmpty verifies that an empty encode response is not
// appended to the ordered list.
func TestNIXL_MergeIgnoresEmpty(t *testing.T) {
	c, err := Build(connector.NameNIXLv2)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}
	c.MergeEncodeResponse(reqCtx, nil)
	c.MergeEncodeResponse(reqCtx, map[string]any{})
	if len(reqCtx.ECTransferParams) != 0 {
		t.Fatalf("expected empty ECTransferParams, got %v", reqCtx.ECTransferParams)
	}
	if got := c.PreparePrefillECParams(reqCtx); got != nil {
		t.Fatalf("expected nil ec_transfer_params, got %v", got)
	}
}

// TestSharedStorage_NoWireFields verifies that the shared_storage EC
// connector emits nothing on the prefill request and does not mutate
// ECTransferParams on encode response.
func TestSharedStorage_NoWireFields(t *testing.T) {
	c, err := Build(connector.NameSharedStorage)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}

	c.MergeEncodeResponse(reqCtx, map[string]any{"hash-x": map[string]any{"peer_host": "10.0.0.9"}})
	if len(reqCtx.ECTransferParams) != 0 {
		t.Errorf("shared_storage should not populate ECTransferParams, got %v", reqCtx.ECTransferParams)
	}
	if got := c.PreparePrefillECParams(reqCtx); got != nil {
		t.Errorf("shared_storage should emit no ec_transfer_params, got %v", got)
	}
}
