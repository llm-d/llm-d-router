/*
Copyright 2026 The Kubernetes Authors.

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

package zmqmetrics

import (
	"context"
	"encoding/hex"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// loadStatGoldenHex is the exact encoding of
// LoadStat(num_running_reqs=7, num_waiting_reqs=3, num_tokens=1024,
// max_total_num_tokens=8192, attn_dp_rank=2) pinned by SGLang's
// test_loadstat_golden_bytes; asserting the same bytes here closes the
// cross-language wire contract.
const loadStatGoldenHex = "96a84c6f6164537461740703cd0400cd200002"

func TestZMQExtractor_SGLang(t *testing.T) {
	ext, err := NewZMQMetricsExtractor("test-extractor", EngineSGLang)
	require.NoError(t, err)
	assert.Equal(t, ZMQExtractorType, ext.TypedName().Type)

	t.Run("golden bytes", func(t *testing.T) {
		payload, err := hex.DecodeString(loadStatGoldenHex)
		require.NoError(t, err)

		ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1"}, nil)
		err = ext.Extract(context.Background(), fwkdl.StreamInput[[]byte]{Payload: payload, Endpoint: ep})
		require.NoError(t, err)

		m := ep.GetMetrics()
		assert.Equal(t, 7, m.RunningRequestsSize)
		assert.Equal(t, 3, m.WaitingQueueSize)
		assert.Equal(t, 8192, m.KvCacheMaxTokenCapacity)
		assert.Equal(t, 0.125, m.KVCacheUsagePercent)
	})

	t.Run("null attn_dp_rank and trailing fields tolerated", func(t *testing.T) {
		payload, err := msgpack.Marshal([]any{"LoadStat", 1, 2, 100, 400, nil, "future-field"})
		require.NoError(t, err)

		ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.2"}, nil)
		err = ext.Extract(context.Background(), fwkdl.StreamInput[[]byte]{Payload: payload, Endpoint: ep})
		require.NoError(t, err)

		m := ep.GetMetrics()
		assert.Equal(t, 1, m.RunningRequestsSize)
		assert.Equal(t, 2, m.WaitingQueueSize)
		assert.Equal(t, 400, m.KvCacheMaxTokenCapacity)
		assert.Equal(t, 0.25, m.KVCacheUsagePercent)
	})

	t.Run("zero capacity keeps previous usage", func(t *testing.T) {
		ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.3"}, nil)

		first, err := msgpack.Marshal([]any{"LoadStat", 1, 0, 100, 400, 0})
		require.NoError(t, err)
		require.NoError(t, ext.Extract(context.Background(), fwkdl.StreamInput[[]byte]{Payload: first, Endpoint: ep}))

		second, err := msgpack.Marshal([]any{"LoadStat", 5, 6, 0, 0, 0})
		require.NoError(t, err)
		require.NoError(t, ext.Extract(context.Background(), fwkdl.StreamInput[[]byte]{Payload: second, Endpoint: ep}))

		m := ep.GetMetrics()
		assert.Equal(t, 5, m.RunningRequestsSize)
		assert.Equal(t, 6, m.WaitingQueueSize)
		assert.Equal(t, 400, m.KvCacheMaxTokenCapacity)
		assert.Equal(t, 0.25, m.KVCacheUsagePercent)
	})
}

func TestZMQExtractor_SGLangInvalidPayload(t *testing.T) {
	ext, err := NewZMQMetricsExtractor("test-extractor", EngineSGLang)
	require.NoError(t, err)

	cases := []struct {
		name    string
		payload []byte
	}{
		{"empty", nil},
		{"not msgpack", []byte("not-msgpack")},
		{"wrong tag", mustMarshal(t, []any{"KVEventBatch", 1, 2, 3, 4})},
		{"non-string tag", mustMarshal(t, []any{42, 1, 2, 3, 4})},
		{"too few fields", mustMarshal(t, []any{"LoadStat", 1, 2})},
		{"non-integer count", mustMarshal(t, []any{"LoadStat", "x", 2, 3, 4})},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ep := fwkdl.NewEndpoint(nil, nil)
			err := ext.Extract(context.Background(), fwkdl.StreamInput[[]byte]{Payload: tc.payload, Endpoint: ep})
			assert.Error(t, err)
		})
	}
}

func TestDecodeLoadStat(t *testing.T) {
	t.Run("maps positional fields", func(t *testing.T) {
		stat, err := decodeLoadStat(mustMarshal(t, []any{"LoadStat", 7, 3, 1024, 8192, 2}))
		require.NoError(t, err)
		assert.Equal(t, 7, stat.numRunningReqs)
		assert.Equal(t, 3, stat.numWaitingReqs)
		assert.Equal(t, 1024, stat.numTokens)
		assert.Equal(t, 8192, stat.maxTotalNumTokens)
	})

	t.Run("exactly minimum fields", func(t *testing.T) {
		stat, err := decodeLoadStat(mustMarshal(t, []any{"LoadStat", 1, 2, 3, 4}))
		require.NoError(t, err)
		assert.Equal(t, 4, stat.maxTotalNumTokens)
	})

	t.Run("zero values decode", func(t *testing.T) {
		stat, err := decodeLoadStat(mustMarshal(t, []any{"LoadStat", 0, 0, 0, 0, nil}))
		require.NoError(t, err)
		assert.Equal(t, 0, stat.numRunningReqs)
		assert.Equal(t, 0, stat.maxTotalNumTokens)
	})

	t.Run("error names the bad field", func(t *testing.T) {
		_, err := decodeLoadStat(mustMarshal(t, []any{"LoadStat", 1, 2, 3.5, 4}))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "num_tokens")
	})

	t.Run("msgpack map is rejected", func(t *testing.T) {
		_, err := decodeLoadStat(mustMarshal(t, map[string]any{"num_running_reqs": 1}))
		assert.Error(t, err)
	})
}

func TestAsInt(t *testing.T) {
	accepted := []any{
		int(1), int8(1), int16(1), int32(1), int64(1),
		uint(1), uint8(1), uint16(1), uint32(1), uint64(1),
	}
	for _, v := range accepted {
		val, ok := asInt(v)
		assert.True(t, ok, "%T should be accepted", v)
		assert.Equal(t, 1, val, "%T", v)
	}

	rejected := []any{nil, float32(1), float64(1), "1", true, []any{1}}
	for _, v := range rejected {
		_, ok := asInt(v)
		assert.False(t, ok, "%T should be rejected", v)
	}
}

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	b, err := msgpack.Marshal(v)
	require.NoError(t, err)
	return b
}
