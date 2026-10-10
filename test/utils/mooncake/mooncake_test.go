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

package mooncake

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

func writeTrace(t *testing.T, lines ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "trace.jsonl")
	var data []byte
	for _, l := range lines {
		data = append(data, l+"\n"...)
	}
	require.NoError(t, os.WriteFile(path, data, 0o600))
	return path
}

func TestBuild(t *testing.T) {
	path := writeTrace(t,
		`{"timestamp": 0, "input_length": 4000, "output_length": 10, "hash_ids": [0, 1, 2, 3]}`,
		`{"timestamp": 1, "input_length": 4000, "output_length": 10, "hash_ids": [0, 1, 2, 4]}`,
		`{"timestamp": 2, "input_length": 0, "output_length": 10, "hash_ids": []}`,
	)
	r, err := Build(path)
	require.NoError(t, err)
	require.Len(t, r.Turns, 2*dupFactor, "records without hash_ids are dropped")

	first, second := r.Turns[0], r.Turns[1]
	shared := 3 * expandFactor
	require.Len(t, first.Query, 4*expandFactor)
	require.Len(t, second.Query, 4*expandFactor)
	require.Equal(t, first.Query[:shared], second.Query[:shared], "shared hash IDs give shared keys")
	require.NotEqual(t, first.Query[shared], second.Query[shared], "keys diverge with the hash IDs")
	require.Equal(t, first.Worker, second.Worker, "a shared routing prefix picks the same worker")
	require.Equal(t, first.Query, first.Stored, "a cold worker stores every key")
	require.Equal(t, second.Query[shared:], second.Stored, "a warm worker stores only new keys")

	for i, turn := range r.Turns {
		require.GreaterOrEqual(t, turn.Worker, 0)
		require.Less(t, turn.Worker, Workers)
		require.Empty(t, turn.Removed, "turn %d: the trace fits in one worker's cache", i)
		require.Equal(t, turn.Stored, turn.Warm, "turn %d: nothing was evicted", i)
		if i >= 2 {
			require.NotContains(t, r.Turns[0].Query, turn.Query[0], "turn %d: duplicates never share keys", i)
		}
	}
	require.Equal(t, float64(4*expandFactor), r.QueryBlocks)

	again, err := Build(path)
	require.NoError(t, err)
	require.True(t, slices.EqualFunc(r.Turns, again.Turns, func(a, b Turn) bool {
		return a.Worker == b.Worker && slices.Equal(a.Query, b.Query)
	}), "Build is deterministic")
}

func TestBuildRejectsEmptyTrace(t *testing.T) {
	_, err := Build(writeTrace(t, `{"hash_ids": []}`))
	require.Error(t, err)
}
