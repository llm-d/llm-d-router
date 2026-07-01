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

package ec

import (
	"context"
	"reflect"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
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
	for _, name := range []string{NIXL, SharedStorage} {
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
// response in order and emits a flat map keyed by mm_hash on the prefill
// request: {hash1: {...}, hash2: {...}}.
func TestNIXL_MergeAndPrepare(t *testing.T) {
	c, err := Build(NIXL)
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{}

	got, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("PreparePrefillECParams: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected empty ec_transfer_params before encodes, got %v", got)
	}

	resp1 := map[string]any{"hash-a": map[string]any{"peer_port": 5501, "size_bytes": 1228800, "nixl_agent_metadata_b64": "bml4..."}}
	resp2 := map[string]any{"hash-b": map[string]any{"peer_port": 5502, "size_bytes": 1228800, "nixl_agent_metadata_b64": "QWdlbnQ..."}}

	c.MergeEncodeResponse(context.Background(), reqCtx, resp1)
	c.MergeEncodeResponse(context.Background(), reqCtx, resp2)

	if len(reqCtx.ECTransferParams) != 2 {
		t.Fatalf("expected 2 entries in ECTransferParams, got %d", len(reqCtx.ECTransferParams))
	}

	want := map[string]any{
		"hash-a": map[string]any{"peer_port": 5501, "size_bytes": 1228800, "nixl_agent_metadata_b64": "bml4..."},
		"hash-b": map[string]any{"peer_port": 5502, "size_bytes": 1228800, "nixl_agent_metadata_b64": "QWdlbnQ..."},
	}
	got, err = c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("PreparePrefillECParams: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("prefill ec_transfer_params:\n got=%v\nwant=%v", got, want)
	}
}

// TestNIXL_MergeAndPrepare_DuplicateHashes_Identical verifies that when the
// same mm_hash appears in multiple encode responses with byte-equal
// descriptors (e.g., the request references the same image twice), the
// connector dedups to a single entry without error.
func TestNIXL_MergeAndPrepare_DuplicateHashes_Identical(t *testing.T) {
	c, err := Build(NIXL)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}

	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{"hash-a": map[string]any{"peer_port": 5501, "nixl_agent_metadata_b64": "bml4..."}})
	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{"hash-a": map[string]any{"peer_port": 5501, "nixl_agent_metadata_b64": "bml4..."}})

	got, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("identical duplicate descriptors should not error: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 entry after dedup, got %d: %v", len(got), got)
	}
}

// TestNIXL_MergeAndPrepare_DuplicateHashes_Conflict verifies that conflicting
// descriptors for the same mm_hash are a hard error rather than a silent
// last-write-wins: two encoder replicas returning different peer_port /
// nixl_agent_metadata_b64 for one mm_hash means prefill cannot know which
// peer still holds the buffer.
func TestNIXL_MergeAndPrepare_DuplicateHashes_Conflict(t *testing.T) {
	c, err := Build(NIXL)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}

	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{"hash-a": map[string]any{"peer_port": 5501}})
	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{"hash-a": map[string]any{"peer_port": 5599}})

	got, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err == nil {
		t.Fatalf("expected error for conflicting descriptors, got %v", got)
	}
}

// TestNIXL_MergeEncodeResponse_MultiKeyResponse exercises a single encoder
// response carrying more than one {mm_hash: descriptor} pair: every key must
// survive into the flattened prefill map.
func TestNIXL_MergeEncodeResponse_MultiKeyResponse(t *testing.T) {
	c, err := Build(NIXL)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}
	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{
		"hash-a": map[string]any{"peer_port": 5501},
		"hash-b": map[string]any{"peer_port": 5502},
	})

	got, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("PreparePrefillECParams: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 entries from a multi-key response, got %d: %v", len(got), got)
	}
	for _, h := range []string{"hash-a", "hash-b"} {
		if _, ok := got[h]; !ok {
			t.Errorf("missing %q in prefill ec_transfer_params: %v", h, got)
		}
	}
}

// TestNIXL_MergeEncodeResponse_NilInnerValue pins the handling of an encoder
// response that maps a hash to a nil descriptor ({"hash-x": nil}). The
// response is non-empty so it is recorded, but a nil descriptor carries
// nothing to transfer and must be dropped rather than forwarded as
// "hash-x": null on the prefill body.
func TestNIXL_MergeEncodeResponse_NilInnerValue(t *testing.T) {
	c, err := Build(NIXL)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}
	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{"hash-x": nil})

	if len(reqCtx.ECTransferParams) != 1 {
		t.Fatalf("expected the non-empty response to be recorded, got %d", len(reqCtx.ECTransferParams))
	}

	got, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("PreparePrefillECParams: %v", err)
	}
	if _, ok := got["hash-x"]; ok {
		t.Errorf("nil descriptor should be dropped, got %v", got)
	}
}

// TestNIXL_PreparePrefillECParams_NonMapDescriptor pins copyDescriptor's
// non-map fallthrough: the descriptor is opaque, so a non-nil scalar value
// (not a map[string]any) carries no aliasing risk and must pass through to the
// prefill params unchanged rather than being dropped or mangled.
func TestNIXL_PreparePrefillECParams_NonMapDescriptor(t *testing.T) {
	c, err := Build(NIXL)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}
	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{"hash-x": "opaque-scalar"})

	got, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("PreparePrefillECParams: %v", err)
	}
	if got["hash-x"] != "opaque-scalar" {
		t.Errorf("non-map descriptor should pass through unchanged, got %v", got["hash-x"])
	}
}

// TestNIXL_PreparePrefillECParams_DefensiveCopy verifies that the returned
// ec_transfer_params is independent of reqCtx.ECTransferParams: mutating a
// returned descriptor must not leak back into the request context.
func TestNIXL_PreparePrefillECParams_DefensiveCopy(t *testing.T) {
	c, err := Build(NIXL)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}
	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{"hash-a": map[string]any{"peer_port": 5501}})

	got, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("PreparePrefillECParams: %v", err)
	}
	inner, ok := got["hash-a"].(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any for hash-a, got %T", got["hash-a"])
	}
	inner["peer_port"] = 9999

	stored, ok := reqCtx.ECTransferParams[0]["hash-a"].(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any in ECTransferParams[0][hash-a], got %T", reqCtx.ECTransferParams[0]["hash-a"])
	}
	if stored["peer_port"] != 5501 {
		t.Errorf("defensive copy broken: mutation of the returned map leaked into reqCtx.ECTransferParams; got peer_port=%v, want 5501", stored["peer_port"])
	}
}

// TestNIXL_PreparePrefillECParams_Idempotent verifies that calling
// PreparePrefillECParams twice returns the same result and does not mutate
// reqCtx.ECTransferParams.
func TestNIXL_PreparePrefillECParams_Idempotent(t *testing.T) {
	c, err := Build(NIXL)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}
	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{"hash-a": map[string]any{"peer_port": 5501}})

	before := len(reqCtx.ECTransferParams)
	first, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("PreparePrefillECParams: %v", err)
	}
	second, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("PreparePrefillECParams: %v", err)
	}

	if len(reqCtx.ECTransferParams) != before {
		t.Errorf("PreparePrefillECParams mutated ECTransferParams: len went from %d to %d", before, len(reqCtx.ECTransferParams))
	}
	if !reflect.DeepEqual(first, second) {
		t.Errorf("PreparePrefillECParams not idempotent:\n first=%v\nsecond=%v", first, second)
	}
	// Each call must return a fresh outer map; otherwise the function would be
	// caching state across calls.
	if reflect.ValueOf(first).Pointer() == reflect.ValueOf(second).Pointer() {
		t.Errorf("PreparePrefillECParams returned the same outer map on repeat call; expected a fresh map per call")
	}
}

// TestNIXL_MergeIgnoresEmpty verifies that an empty encode response is not
// appended to the ordered list.
func TestNIXL_MergeIgnoresEmpty(t *testing.T) {
	c, err := Build(NIXL)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}
	c.MergeEncodeResponse(context.Background(), reqCtx, nil)
	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{})
	if len(reqCtx.ECTransferParams) != 0 {
		t.Fatalf("expected empty ECTransferParams, got %v", reqCtx.ECTransferParams)
	}
	got, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("PreparePrefillECParams: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected empty ec_transfer_params, got %v", got)
	}
}

// TestSharedStorage_NoWireFields verifies that the ec-shared-storage EC
// connector emits nothing on the prefill request and does not mutate
// ECTransferParams on encode response.
func TestSharedStorage_NoWireFields(t *testing.T) {
	c, err := Build(SharedStorage)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{}

	c.MergeEncodeResponse(context.Background(), reqCtx, map[string]any{"hash-x": map[string]any{"peer_host": "10.0.0.9"}})
	if len(reqCtx.ECTransferParams) != 0 {
		t.Errorf("ec-shared-storage should not populate ECTransferParams, got %v", reqCtx.ECTransferParams)
	}
	got, err := c.PreparePrefillECParams(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("PreparePrefillECParams: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("ec-shared-storage should emit no ec_transfer_params, got %v", got)
	}
}
