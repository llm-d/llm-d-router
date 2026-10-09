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

package proxy

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

const testNIXLPushEndpoint = testPrefillHostIP1 + ":8000"

func testNIXLPushIdentity(engineID string) nixlPushIdentity {
	return nixlPushIdentity{
		reqcommon.FieldRemoteEngineID: engineID,
		reqcommon.FieldRemoteHost:     testPrefillHostIP1,
		reqcommon.FieldRemotePort:     float64(5600),
		requestFieldTPSize:            float64(2),
		requestFieldTransferMode:      nixlTransferModePush,
	}
}

func newTestNIXLPushIdentityCache(t *testing.T, size int) *nixlPushIdentityCache {
	t.Helper()
	cache, err := newNIXLPushIdentityCache(size)
	require.NoError(t, err)
	return cache
}

func TestExtractNIXLPushIdentity(t *testing.T) {
	// kv_transfer_params of a prefill answer from vLLM's NixlPushConnector.
	const pushAnswerKV = `{
		"do_remote_prefill": true,
		"do_remote_decode": false,
		"remote_block_ids": [1, 2, 3],
		"remote_engine_id": "prefill-engine",
		"remote_request_id": "cmpl-prefill",
		"remote_host": "10.0.0.1",
		"remote_port": 5600,
		"tp_size": 2,
		"pp_size": 1,
		"dcp_size": 1,
		"remote_num_tokens": 64,
		"transfer_mode": "push"
	}`
	decodedAnswer := func(t *testing.T) map[string]any {
		t.Helper()
		var kv map[string]any
		require.NoError(t, json.Unmarshal([]byte(pushAnswerKV), &kv))
		return kv
	}

	t.Run("keeps only the identity fields, as decoded", func(t *testing.T) {
		got, ok := extractNIXLPushIdentity(decodedAnswer(t))
		require.True(t, ok)
		require.Equal(t, nixlPushIdentity{
			reqcommon.FieldRemoteEngineID: "prefill-engine",
			reqcommon.FieldRemoteHost:     testPrefillHostIP1,
			reqcommon.FieldRemotePort:     float64(5600),
			requestFieldTPSize:            float64(2),
			requestFieldPPSize:            float64(1),
			requestFieldDCPSize:           float64(1),
			requestFieldTransferMode:      nixlTransferModePush,
		}, got)
	})

	t.Run("leaves out absent optional sizes", func(t *testing.T) {
		kv := decodedAnswer(t)
		delete(kv, requestFieldPPSize)
		delete(kv, requestFieldDCPSize)
		got, ok := extractNIXLPushIdentity(kv)
		require.True(t, ok)
		require.Equal(t, testNIXLPushIdentity("prefill-engine"), got)
	})

	rejected := []struct {
		name   string
		mutate func(kv map[string]any)
	}{
		{"without transfer_mode", func(kv map[string]any) { delete(kv, requestFieldTransferMode) }},
		{"in pull mode", func(kv map[string]any) { kv[requestFieldTransferMode] = "pull" }},
		{"without remote_engine_id", func(kv map[string]any) { delete(kv, reqcommon.FieldRemoteEngineID) }},
		{"without remote_host", func(kv map[string]any) { delete(kv, reqcommon.FieldRemoteHost) }},
		{"without remote_port", func(kv map[string]any) { delete(kv, reqcommon.FieldRemotePort) }},
		{"without tp_size", func(kv map[string]any) { delete(kv, requestFieldTPSize) }},
		{"with a null pp_size", func(kv map[string]any) { kv[requestFieldPPSize] = nil }},
		{"with a list as remote_port", func(kv map[string]any) { kv[reqcommon.FieldRemotePort] = []any{float64(5600)} }},
	}
	for _, tc := range rejected {
		t.Run("rejects an answer "+tc.name, func(t *testing.T) {
			kv := decodedAnswer(t)
			tc.mutate(kv)
			_, ok := extractNIXLPushIdentity(kv)
			require.False(t, ok)
		})
	}

	t.Run("rejects kv_transfer_params that are not an object", func(t *testing.T) {
		_, ok := extractNIXLPushIdentity(nil)
		require.False(t, ok)
	})
}

func TestNIXLPushIdentity_Equal(t *testing.T) {
	identity := testNIXLPushIdentity("prefill-engine")
	require.True(t, identity.equal(testNIXLPushIdentity("prefill-engine")))
	require.False(t, identity.equal(testNIXLPushIdentity("restarted-engine")))

	withPPSize := testNIXLPushIdentity("prefill-engine")
	withPPSize[requestFieldPPSize] = float64(1)
	require.False(t, identity.equal(withPPSize))
}

func TestNIXLPushIdentityCache_PutGet(t *testing.T) {
	cache := newTestNIXLPushIdentityCache(t, 4)
	_, ok := cache.get(testNIXLPushEndpoint)
	require.False(t, ok)

	cache.put(testNIXLPushEndpoint, testNIXLPushIdentity("prefill-engine"))
	got, ok := cache.get(testNIXLPushEndpoint)
	require.True(t, ok)
	require.Equal(t, testNIXLPushIdentity("prefill-engine"), got)
}

// Callers add request fields to the identity they get; those must not reach
// the next request to the same endpoint.
func TestNIXLPushIdentityCache_HoldsItsOwnCopy(t *testing.T) {
	cache := newTestNIXLPushIdentityCache(t, 4)
	stored := testNIXLPushIdentity("prefill-engine")
	cache.put(testNIXLPushEndpoint, stored)
	stored[reqcommon.FieldRemoteEngineID] = "changed-after-put"

	got, _ := cache.get(testNIXLPushEndpoint)
	got[requestFieldTransferID] = "xfer-1"

	got, _ = cache.get(testNIXLPushEndpoint)
	require.Equal(t, testNIXLPushIdentity("prefill-engine"), got)
}

func TestNIXLPushIdentityCache_PutReplaces(t *testing.T) {
	cache := newTestNIXLPushIdentityCache(t, 4)
	cache.put(testNIXLPushEndpoint, testNIXLPushIdentity("prefill-engine"))
	cache.put(testNIXLPushEndpoint, testNIXLPushIdentity("restarted-engine"))

	got, ok := cache.get(testNIXLPushEndpoint)
	require.True(t, ok)
	require.Equal(t, testNIXLPushIdentity("restarted-engine"), got)
}

func TestNIXLPushIdentityCache_DropIfMatches(t *testing.T) {
	cache := newTestNIXLPushIdentityCache(t, 4)
	older := testNIXLPushIdentity("prefill-engine")
	newer := testNIXLPushIdentity("restarted-engine")
	cache.put(testNIXLPushEndpoint, newer)

	require.False(t, cache.dropIfMatches(testNIXLPushEndpoint, older))
	require.False(t, cache.dropIfMatches(testPrefillHostIP2+":8000", newer))
	got, ok := cache.get(testNIXLPushEndpoint)
	require.True(t, ok)
	require.Equal(t, newer, got)

	require.True(t, cache.dropIfMatches(testNIXLPushEndpoint, newer))
	_, ok = cache.get(testNIXLPushEndpoint)
	require.False(t, ok)
	require.False(t, cache.dropIfMatches(testNIXLPushEndpoint, newer))
}

func TestNIXLPushIdentityCache_EvictsLeastRecentlyUsed(t *testing.T) {
	cache := newTestNIXLPushIdentityCache(t, 2)
	rank0, rank1, rank2 := testPrefillHostIP1+":8000", testPrefillHostIP1+":8001", testPrefillHostIP1+":8002"
	cache.put(rank0, testNIXLPushIdentity("prefill-engine_dp0"))
	cache.put(rank1, testNIXLPushIdentity("prefill-engine_dp1"))
	_, _ = cache.get(rank0)
	cache.put(rank2, testNIXLPushIdentity("prefill-engine_dp2"))

	_, ok := cache.get(rank1)
	require.False(t, ok)
	_, ok = cache.get(rank0)
	require.True(t, ok)
	_, ok = cache.get(rank2)
	require.True(t, ok)
}

// Request goroutines and the data-parallel rank servers share one cache, so
// the race detector must find no unguarded access.
func TestNIXLPushIdentityCache_ConcurrentUse(t *testing.T) {
	cache := newTestNIXLPushIdentityCache(t, 4)
	identity := testNIXLPushIdentity("prefill-engine")
	// Goroutines that run one after another are ordered by the WaitGroup and
	// would hide a missing lock from the race detector.
	start := make(chan struct{})
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			<-start
			for range 100 {
				cache.put(testNIXLPushEndpoint, identity)
				_, _ = cache.get(testNIXLPushEndpoint)
				cache.dropIfMatches(testNIXLPushEndpoint, identity)
			}
		})
	}
	close(start)
	wg.Wait()
}
