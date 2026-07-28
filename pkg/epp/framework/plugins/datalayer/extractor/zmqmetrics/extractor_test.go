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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	sourcezmq "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/zmqmetrics"
)

func TestZMQExtractor_Extract(t *testing.T) {
	ext := NewZMQMetricsExtractor("test-extractor")
	assert.Equal(t, ZMQExtractorType, ext.TypedName().Type)

	t.Run("int and string cache config info", func(t *testing.T) {
		stats := sourcezmq.ZmqMetricsStats{
			NumRequestsRunning: 5,
			NumRequestsWaiting: 12,
			KVCacheUsagePerc:   0.45,
			CacheConfigInfo: map[string]any{
				"block_size":           "16",
				"num_gpu_blocks":       "7605",
				"kv_cache_size_tokens": "121680",
			},
			EngineID: "engine-1",
		}

		payload, err := msgpack.Marshal(stats)
		require.NoError(t, err)

		ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
			Address: "10.0.0.1",
		}, nil)

		in := fwkdl.StreamInput[[]byte]{
			Payload:  payload,
			Endpoint: ep,
		}

		err = ext.Extract(context.Background(), in)
		require.NoError(t, err)

		m := ep.GetMetrics()
		assert.Equal(t, 5, m.RunningRequestsSize)
		assert.Equal(t, 12, m.WaitingQueueSize)
		assert.Equal(t, 0.45, m.KVCacheUsagePercent)
		assert.Equal(t, 16, m.CacheBlockSize)
		assert.Equal(t, 7605, m.CacheNumBlocks)
		assert.Equal(t, 121680, m.KvCacheMaxTokenCapacity)
	})

	t.Run("tuple slice msgpack", func(t *testing.T) {
		tuplePayload, err := msgpack.Marshal([]any{
			3, 7, 0.25, map[string]any{"block_size": 16, "num_gpu_blocks": 1000}, "engine-2",
		})
		require.NoError(t, err)

		ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.2"}, nil)
		err = ext.Extract(context.Background(), fwkdl.StreamInput[[]byte]{Payload: tuplePayload, Endpoint: ep})
		require.NoError(t, err)

		m := ep.GetMetrics()
		assert.Equal(t, 3, m.RunningRequestsSize)
		assert.Equal(t, 7, m.WaitingQueueSize)
		assert.Equal(t, 0.25, m.KVCacheUsagePercent)
		assert.Equal(t, 16, m.CacheBlockSize)
		assert.Equal(t, 1000, m.CacheNumBlocks)
	})
}

func TestZMQExtractor_InvalidPayload(t *testing.T) {
	ext := NewZMQMetricsExtractor("test-extractor")
	ep := fwkdl.NewEndpoint(nil, nil)

	in := fwkdl.StreamInput[[]byte]{
		Payload:  []byte("not-msgpack"),
		Endpoint: ep,
	}

	err := ext.Extract(context.Background(), in)
	assert.Error(t, err)
}
