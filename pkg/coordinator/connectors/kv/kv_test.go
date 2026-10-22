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
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/go-logr/logr/funcr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
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
	if prefill[reqcommon.FieldBootstrapPort] != sglangBootstrapPort {
		t.Errorf("prefill: %s = %v, want %d", reqcommon.FieldBootstrapPort, prefill[reqcommon.FieldBootstrapPort], sglangBootstrapPort)
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

func TestOffloadingKV_Params(t *testing.T) {
	c, err := Build(Offloading)
	if err != nil {
		t.Fatalf("Build(%q): %v", Offloading, err)
	}
	if c.Name() != Offloading {
		t.Fatalf("Name() = %q, want %q", c.Name(), Offloading)
	}

	reqCtx := &pipeline.RequestContext{
		// The prefill response's kv_transfer_params play no part in the protocol.
		KVTransferParams: map[string]any{"ignored": "field"},
	}

	// Prefill: remote_decoder with a fresh kv_request_id, stored on the context.
	prefill := c.PreparePrefillKVParams(context.Background(), reqCtx)
	if reqCtx.KVRequestID == "" {
		t.Fatal("prefill: KVRequestID not stored on the request context")
	}
	wantPrefill := map[string]any{
		reqcommon.FieldRemoteDecoder: map[string]any{reqcommon.FieldKVRequestID: reqCtx.KVRequestID},
	}
	if !reflect.DeepEqual(prefill, wantPrefill) {
		t.Errorf("prefill params:\n got=%v\nwant=%v", prefill, wantPrefill)
	}

	// Decode: remote_prefiller with the same kv_request_id and the prefill pod
	// read from the response headers.
	reqCtx.PrefillPeer = &pipeline.P2PPeer{Host: "10.0.0.42", Port: 7779}
	wantDecode := map[string]any{
		reqcommon.FieldRemotePrefiller: map[string]any{
			reqcommon.FieldKVRequestID: reqCtx.KVRequestID,
			reqcommon.FieldRemoteHost:  "10.0.0.42",
			reqcommon.FieldRemotePort:  7779,
		},
	}
	if got := c.PrepareDecodeKVParams(context.Background(), reqCtx); !reflect.DeepEqual(got, wantDecode) {
		t.Errorf("decode params:\n got=%v\nwant=%v", got, wantDecode)
	}
}

func TestOffloadingKV_DistinctKVRequestIDPerRequest(t *testing.T) {
	c, _ := Build(Offloading)
	first := &pipeline.RequestContext{}
	second := &pipeline.RequestContext{}
	c.PreparePrefillKVParams(context.Background(), first)
	c.PreparePrefillKVParams(context.Background(), second)
	if first.KVRequestID == second.KVRequestID {
		t.Fatalf("kv_request_id %q reused across requests", first.KVRequestID)
	}
}

// prefillResponseHeaders carries the header the EPP request-header-copy-handler
// copies from p2p-source-producer's request headers onto the prefill response.
func prefillResponseHeaders(p2pAddress string) http.Header {
	h := http.Header{}
	if p2pAddress != "" {
		h.Set(routing.KVCachePrefillHeader, p2pAddress)
	}
	return h
}

// The prefill pod's P2P tier host and port are read from the prefill
// response header.
func TestOffloadingKV_ReadPrefillResponse(t *testing.T) {
	cases := map[string]struct {
		p2pAddress string
		wantHost   string
		wantPort   int
	}{
		"ipv4": {p2pAddress: "10.0.0.7:7779", wantHost: "10.0.0.7", wantPort: 7779},
		// routing.ParseHostPort covers the other values; this one shows the
		// connector uses it, as the sidecar does.
		"surrounding whitespace": {p2pAddress: " 10.0.0.7:7779 ", wantHost: "10.0.0.7", wantPort: 7779},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			reqCtx := &pipeline.RequestContext{}
			if err := (offloadingKV{}).ReadPrefillResponse(reqCtx, prefillResponseHeaders(tc.p2pAddress)); err != nil {
				t.Fatal(err)
			}
			want := &pipeline.P2PPeer{Host: tc.wantHost, Port: tc.wantPort}
			if !reflect.DeepEqual(reqCtx.PrefillPeer, want) {
				t.Fatalf("read %+v, want %+v", reqCtx.PrefillPeer, want)
			}
		})
	}
}

// Without the header a decode request could not pull, so the read fails.
func TestOffloadingKV_ReadPrefillResponse_Rejects(t *testing.T) {
	const missing = "has no " + routing.KVCachePrefillHeader + " header"
	const malformed = "is not a host:port with a TCP port"
	cases := map[string]struct {
		headers http.Header
		wantMsg string
	}{
		"no header":       {headers: prefillResponseHeaders(""), wantMsg: missing},
		"whitespace only": {headers: prefillResponseHeaders("  "), wantMsg: missing},
		"malformed":       {headers: prefillResponseHeaders("10.0.0.7"), wantMsg: malformed},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			reqCtx := &pipeline.RequestContext{}
			err := (offloadingKV{}).ReadPrefillResponse(reqCtx, tc.headers)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("error %q does not contain %q", err, tc.wantMsg)
			}
			// Only a missing header points at the EPP plugins.
			if mentionsEPP := strings.Contains(err.Error(), "request-header-copy-handler"); mentionsEPP != (tc.wantMsg == missing) {
				t.Fatalf("error %q: mentions EPP plugins = %v", err, mentionsEPP)
			}
			if reqCtx.PrefillPeer != nil {
				t.Fatalf("context set on error: %+v", reqCtx.PrefillPeer)
			}
		})
	}
}

// logCapture returns a context whose logger appends each Info line to the
// returned slice.
func logCapture() (context.Context, *[]string) {
	var lines []string
	logger := funcr.New(func(_, args string) { lines = append(lines, args) }, funcr.Options{})
	return log.IntoContext(context.Background(), logger), &lines
}

// A pipeline without a prefill step leaves kv_transfer_params to the EPP
// p2p-source-producer: the decode params are an empty object, not nil, and
// one Info line names the cause, since TRACE is off in deployments. A prefill
// request prepared without its response read has no peer to name, so it is
// the same case rather than a remote_prefiller block with an empty host.
func TestOffloadingKV_NoPrefillPeer_EmptyDecodeParams(t *testing.T) {
	c, _ := Build(Offloading)
	noPrefill := &pipeline.RequestContext{}
	prefillNotRead := &pipeline.RequestContext{}
	c.PreparePrefillKVParams(context.Background(), prefillNotRead)
	for name, reqCtx := range map[string]*pipeline.RequestContext{"no prefill": noPrefill, "prefill response not read": prefillNotRead} {
		t.Run(name, func(t *testing.T) {
			ctx, lines := logCapture()
			got := c.PrepareDecodeKVParams(ctx, reqCtx)
			if got == nil || len(got) != 0 {
				t.Errorf("decode params = %v, want an empty map", got)
			}
			if len(*lines) != 1 || !strings.Contains((*lines)[0], "no prefill P2P peer") {
				t.Errorf("log lines = %v, want one naming the missing peer", *lines)
			}
		})
	}
}

// A decode request that names its prefill peer is the normal path, so it logs
// nothing at the default level.
func TestOffloadingKV_PrefillPeer_NoInfoLog(t *testing.T) {
	c, _ := Build(Offloading)
	ctx, lines := logCapture()
	c.PrepareDecodeKVParams(ctx, &pipeline.RequestContext{KVRequestID: "K", PrefillPeer: &pipeline.P2PPeer{Host: "10.0.0.7", Port: 7779}})
	if len(*lines) != 0 {
		t.Errorf("log lines = %v, want none", *lines)
	}
}
