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

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/scheduling/capture"
)

func writeCapture(t *testing.T, records ...capture.Record) string {
	t.Helper()
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, r := range records {
		require.NoError(t, enc.Encode(r))
	}
	path := filepath.Join(t.TempDir(), "capture.ndjson")
	require.NoError(t, os.WriteFile(path, buf.Bytes(), 0o600))
	return path
}

func TestReadRecordsMergesReplicasByTime(t *testing.T) {
	// Each replica's capture is in its own time order; together they are not.
	replica0 := writeCapture(t, record(0, "e1", both, 1), record(2, "e1", both, 3))
	replica1 := writeCapture(t, record(1, "e2", both, 2), record(3, "e2", both, 4))

	records, err := readRecords([]string{replica0, replica1})
	require.NoError(t, err)
	got := make([]uint64, 0, len(records))
	for _, r := range records {
		got = append(got, r.BlockHashes[0])
	}
	require.Equal(t, []uint64{1, 2, 3, 4}, got)
}

func TestReadRecordsReportsBadLine(t *testing.T) {
	path := writeCapture(t, record(0, "e1", both, 1))
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	require.NoError(t, err)
	_, err = f.WriteString("{not json\n")
	require.NoError(t, err)
	require.NoError(t, f.Close())

	_, err = readRecords([]string{path})
	require.ErrorContains(t, err, path+":2")
}

func TestRunAnalyzePrintsLadder(t *testing.T) {
	r0 := record(0, "e1", both, 10, 11, 1)
	r1 := record(1, "e2", both, 10, 11, 2)
	r1.PredictedCachedBlocks = ptr(2)
	path := writeCapture(t, r0, r1)

	var out bytes.Buffer
	require.NoError(t, run([]string{"analyze", "--fleet-kv-blocks", "100", path}, &out))
	lines := strings.Split(out.String(), "\n")
	require.Contains(t, lines, "  Ideal              unbounded shared cache                33.3%")
	require.Contains(t, lines, "  Routed             the router's picks, same caches        0.0%")
	require.Contains(t, lines, "  Observed (router)  router-predicted cached blocks       66.7% (1 of 2 records)")
	require.Contains(t, lines, "  Observed (engine)  engine-reported cached tokens           n/a")
	require.Contains(t, lines, "records: 2 (0 failed, 0 without block hashes), endpoints: 2, blocks: 6")
}

func TestRunUsageErrors(t *testing.T) {
	path := writeCapture(t, record(0, "e1", both, 1))
	for name, args := range map[string][]string{
		"no command":      {},
		"unknown command": {"replay", path},
		"no files":        {"analyze", "--fleet-kv-blocks", "8"},
		"no capacity":     {"analyze", path},
	} {
		t.Run(name, func(t *testing.T) {
			require.Error(t, run(args, &bytes.Buffer{}))
		})
	}
}
