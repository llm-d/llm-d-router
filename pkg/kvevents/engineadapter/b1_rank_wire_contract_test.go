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

package engineadapter

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// B1-U01: the engine adapter must keep the difference between rank 0, rank 1, a
// missing rank, and a malformed rank. Rank 0 is a rank, not "no rank".
//
// The wire shape is SGLang's, not the router's: EventBatch is the positional
// array [ts, events, attn_dp_rank] (sglang/srt/disaggregation/kv_events.py),
// encoded by msgspec with its defaults, so a publisher without a rank sends
// null in the third slot rather than dropping it, and rank 0 sends 0.
const (
	b1WireTopic = "kv@10.0.0.1:8000@test-model"
	b1WireModel = "test-model"
)

func b1Rank(n int) *int { return &n }

func TestSGLangParseMessage_DistinguishesRankPresence(t *testing.T) {
	adapter := NewSGLangAdapter()

	// Minimal SGLang BlockStored: [tag, block_hashes, parent, tokens, block_size].
	stored := []any{"BlockStored", []any{uint64(7)}, uint64(0), []any{1, 2, 3, 4}, 4}

	cases := []struct {
		name     string
		batch    []any
		wantRank *int
		wantErr  bool
	}{
		{
			name:     "explicit rank 0 is a rank",
			batch:    []any{1.0, []any{stored}, 0},
			wantRank: b1Rank(0),
		},
		{
			name:     "explicit rank 3 is a rank",
			batch:    []any{1.0, []any{stored}, 3},
			wantRank: b1Rank(3),
		},
		{
			name:     "null rank stays absent",
			batch:    []any{1.0, []any{stored}, nil},
			wantRank: nil,
		},
		{
			name:    "malformed rank is rejected rather than coerced to 0",
			batch:   []any{1.0, []any{stored}, "0"},
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			payload, err := msgpack.Marshal(tc.batch)
			require.NoError(t, err)

			podID, modelName, batch, err := adapter.ParseMessage(
				&kvevents.RawMessage{Topic: b1WireTopic, Payload: payload})
			if tc.wantErr {
				require.Error(t, err, "a non-integer rank must not decode as a rank")
				return
			}
			require.NoError(t, err)
			assert.Equal(t, "10.0.0.1:8000", podID)
			assert.Equal(t, b1WireModel, modelName)
			require.Len(t, batch.Events, 1)

			if tc.wantRank == nil {
				assert.Nil(t, batch.DataParallelRank,
					"an absent rank must stay absent, not become rank 0")
				return
			}
			require.NotNil(t, batch.DataParallelRank,
				"rank %d must survive decoding", *tc.wantRank)
			assert.Equal(t, *tc.wantRank, *batch.DataParallelRank)
		})
	}
}

// A batch with the rank element truncated must fail closed. SGLang's encoder
// always writes the third element (msgspec defaults, no omit_defaults on
// EventBatch), so a short array is a malformed batch; reading it as rank 0 or
// as a rank-less publisher would both be silent misattribution.
func TestSGLangParseMessage_TruncatedBatchIsRejected(t *testing.T) {
	adapter := NewSGLangAdapter()
	stored := []any{"BlockStored", []any{uint64(7)}, uint64(0), []any{1, 2, 3, 4}, 4}

	payload, err := msgpack.Marshal([]any{1.0, []any{stored}})
	require.NoError(t, err)

	_, _, _, err = adapter.ParseMessage(&kvevents.RawMessage{Topic: b1WireTopic, Payload: payload})
	require.Error(t, err, "a truncated batch must be rejected, not read as an absent rank")
}

// The vLLM adapter shares the batch struct with SGLang; the same rank
// distinction must hold there.
func TestVLLMParseMessage_DistinguishesRankPresence(t *testing.T) {
	adapter := NewVLLMAdapter()

	stored := []any{"BlockStored", []any{uint64(7)}, uint64(0), []any{1, 2, 3, 4}, 4}
	for _, tc := range []struct {
		name     string
		batch    []any
		wantRank *int
	}{
		{name: "rank 0", batch: []any{1.0, []any{stored}, 0}, wantRank: b1Rank(0)},
		{name: "rank 2", batch: []any{1.0, []any{stored}, 2}, wantRank: b1Rank(2)},
		{name: "absent", batch: []any{1.0, []any{stored}, nil}, wantRank: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload, err := msgpack.Marshal(tc.batch)
			require.NoError(t, err)

			_, _, batch, err := adapter.ParseMessage(
				&kvevents.RawMessage{Topic: b1WireTopic, Payload: payload})
			require.NoError(t, err)

			if tc.wantRank == nil {
				assert.Nil(t, batch.DataParallelRank)
				return
			}
			require.NotNil(t, batch.DataParallelRank)
			assert.Equal(t, *tc.wantRank, *batch.DataParallelRank)
		})
	}
}
