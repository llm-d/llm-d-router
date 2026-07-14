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

package engineadapter //nolint:testpackage // Benchmarks access unexported functions

import (
	"fmt"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/vmihailenco/msgpack/v5"
)

// buildBlockStoredPayload creates a BlockStored event with the given number of
// block hashes and token IDs, matching vLLM's array_like msgpack format.
func buildBlockStoredPayload(b *testing.B, numBlocks int, includeOptional, includeExtraKeys bool) []byte {
	b.Helper()

	hashes := make([]any, numBlocks)
	for i := range hashes {
		hashes[i] = uint64(1000 + i) //#nosec G115 -- bench test data, no overflow risk
	}

	tokens := make([]uint32, 64)
	for i := range tokens {
		tokens[i] = uint32(i + 1) //#nosec G115 -- bench test data, no overflow risk
	}

	event := []any{
		"BlockStored",
		hashes,
		uint64(999), // parent hash
		tokens,
		64, // block_size
	}

	if includeOptional {
		event = append(event, 42, "gpu", "lora-a")
	}

	if includeExtraKeys {
		extraKeys := make([]any, numBlocks)
		for i := range extraKeys {
			extraKeys[i] = []any{"uuid-" + fmt.Sprint(i), "salt"}
		}
		event = append(event, extraKeys)
	}

	data, err := msgpack.Marshal(event)
	if err != nil {
		b.Fatal(err)
	}
	return data
}

// buildBatchPayload wraps individual events into a vLLM event batch.
func buildBatchPayload(b *testing.B, events [][]byte) []byte {
	b.Helper()

	rawEvents := make([]any, len(events))
	for i, ev := range events {
		var decoded []any
		if err := msgpack.Unmarshal(ev, &decoded); err != nil {
			b.Fatal(err)
		}
		rawEvents[i] = decoded
	}

	batch := []any{
		1234567890.0, // timestamp
		rawEvents,
		nil, // data_parallel_rank
	}
	data, err := msgpack.Marshal(batch)
	if err != nil {
		b.Fatal(err)
	}
	return data
}

// BenchmarkDecodeVLLMEvent_SinglePass benchmarks single event decoding
// across vLLM v0.15.0 (minimal fields) and v0.18.0 (all fields) payloads.
func BenchmarkDecodeVLLMEvent_SinglePass(b *testing.B) {
	scenarios := []struct {
		name            string
		numBlocks       int
		includeOptional bool
		includeExtra    bool
	}{
		{"v0.15.0_minimal_1block", 1, false, false},
		{"v0.15.0_minimal_16blocks", 16, false, false},
		{"v0.18.0_full_1block", 1, true, true},
		{"v0.18.0_full_16blocks", 16, true, true},
		{"v0.18.0_full_64blocks", 64, true, true},
	}

	adapter := NewVLLMAdapter()

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			payload := buildBlockStoredPayload(b, sc.numBlocks, sc.includeOptional, sc.includeExtra)
			b.ReportAllocs()
			b.SetBytes(int64(len(payload)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := adapter.decodeVLLMEvent(payload)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkParseMessage_Batch benchmarks full batch ParseMessage (end-to-end).
func BenchmarkParseMessage_Batch(b *testing.B) {
	scenarios := []struct {
		name      string
		numEvents int
		optional  bool
		extraKeys bool
	}{
		{"v0.15.0_10events", 10, false, false},
		{"v0.15.0_100events", 100, false, false},
		{"v0.18.0_10events", 10, true, true},
		{"v0.18.0_100events", 100, true, true},
	}

	adapter := NewVLLMAdapter()

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			events := make([][]byte, sc.numEvents)
			for i := range events {
				events[i] = buildBlockStoredPayload(b, 4, sc.optional, sc.extraKeys)
			}
			batchPayload := buildBatchPayload(b, events)
			msg := &kvevents.RawMessage{Topic: "kv@pod-1@Qwen/Qwen3-32B", Payload: batchPayload}

			b.ReportAllocs()
			b.SetBytes(int64(len(batchPayload)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _, err := adapter.ParseMessage(msg)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
