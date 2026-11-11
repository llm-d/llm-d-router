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
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/scheduling/capture"
)

var start = time.Date(2026, 11, 9, 0, 0, 0, 0, time.UTC)

// record builds the i-th request of a trace, routed to endpoint from candidates.
func record(i int, endpoint string, candidates []string, hashes ...uint64) capture.Record {
	return capture.Record{
		SchemaVersion: capture.SchemaVersion,
		Time:          start.Add(time.Duration(i) * time.Second),
		RequestID:     "req",
		Candidates:    candidates,
		Endpoint:      endpoint,
		HashScheme:    "prefixhash",
		BlockSize:     4,
		BlockHashes:   hashes,
	}
}

var both = []string{"e1", "e2"}

func TestLadderNoReuse(t *testing.T) {
	l, err := computeLadder([]capture.Record{
		record(0, "e1", both, 1, 2),
		record(1, "e2", both, 3, 4),
		record(2, "e1", both, 5, 6),
	}, 100)
	require.NoError(t, err)
	for _, r := range []rung{l.ideal, l.capacity, l.perfect, l.routed} {
		require.Equal(t, rung{hits: 0, blocks: 6}, r)
	}
}

func TestLadderSharedPrefixRoundRobin(t *testing.T) {
	// A two-block shared prefix (10, 11) and one unique block per request,
	// routed round-robin.
	l, err := computeLadder([]capture.Record{
		record(0, "e1", both, 10, 11, 1),
		record(1, "e2", both, 10, 11, 2),
		record(2, "e1", both, 10, 11, 3),
		record(3, "e2", both, 10, 11, 4),
	}, 100)
	require.NoError(t, err)
	require.Equal(t, rung{hits: 6, blocks: 12}, l.ideal)
	require.Equal(t, rung{hits: 6, blocks: 12}, l.capacity)
	require.Equal(t, rung{hits: 6, blocks: 12}, l.perfect)
	require.Equal(t, rung{hits: 4, blocks: 12}, l.routed)
	require.Equal(t, 4, l.records)
	require.Equal(t, 2, l.endpoints)
}

func TestLadderCapacityEvictsTailFirst(t *testing.T) {
	// One endpoint and room for three blocks. Storing B evicts one block of A;
	// the engine frees a request's blocks tail first, so A keeps its first block.
	one := []string{"e1"}
	l, err := computeLadder([]capture.Record{
		record(0, "e1", one, 1, 2),
		record(1, "e1", one, 3, 4),
		record(2, "e1", one, 1, 2),
	}, 3)
	require.NoError(t, err)
	require.Equal(t, rung{hits: 2, blocks: 6}, l.ideal)
	require.Equal(t, rung{hits: 1, blocks: 6}, l.capacity)
	require.Equal(t, rung{hits: 1, blocks: 6}, l.perfect)
	require.Equal(t, rung{hits: 1, blocks: 6}, l.routed)
}

func TestLadderCapacitySplitPerEndpoint(t *testing.T) {
	// Four blocks across two endpoints is two per endpoint: a three-block
	// prompt fits the shared cache but not one endpoint's share.
	l, err := computeLadder([]capture.Record{
		record(0, "e1", both, 1, 2, 3),
		record(1, "e1", both, 1, 2, 3),
	}, 4)
	require.NoError(t, err)
	require.Equal(t, rung{hits: 3, blocks: 6}, l.capacity)
	require.Equal(t, rung{hits: 2, blocks: 6}, l.perfect)
	require.Equal(t, rung{hits: 2, blocks: 6}, l.routed)
}

func TestLadderPerfectRoutingStaysInCandidates(t *testing.T) {
	// The second request can only go to e2, so e1's cache can't serve it.
	l, err := computeLadder([]capture.Record{
		record(0, "e1", both, 1, 2),
		record(1, "e2", []string{"e2"}, 1, 2),
	}, 100)
	require.NoError(t, err)
	require.Equal(t, rung{hits: 2, blocks: 4}, l.ideal)
	require.Equal(t, rung{hits: 0, blocks: 4}, l.perfect)
}

func TestLadderPerfectRoutingTieGoesToEmptierCache(t *testing.T) {
	// Both endpoints are cold for the second request; e2 holds fewer blocks,
	// so the request goes there and warms it for the third.
	l, err := computeLadder([]capture.Record{
		record(0, "e1", []string{"e1"}, 9),
		record(1, "e1", both, 1, 2),
		record(2, "e2", []string{"e2"}, 1, 2),
	}, 100)
	require.NoError(t, err)
	require.Equal(t, rung{hits: 2, blocks: 5}, l.perfect)
	require.Equal(t, rung{hits: 0, blocks: 5}, l.routed)
}

func TestLadderPerfectRoutingPicksLongestPrefix(t *testing.T) {
	l, err := computeLadder([]capture.Record{
		record(0, "e1", both, 1),
		record(1, "e2", both, 1, 2, 3),
		record(2, "e1", both, 1, 2, 3, 4),
	}, 100)
	require.NoError(t, err)
	// Perfect routing sends the second request to e1, which holds its first
	// block, and the third request finds three blocks there.
	require.Equal(t, rung{hits: 4, blocks: 8}, l.perfect)
	require.Equal(t, rung{hits: 1, blocks: 8}, l.routed)
}

func TestLadderObservedRungs(t *testing.T) {
	r0 := record(0, "e1", both, 1, 2)
	r0.PredictedCachedBlocks = ptr(5) // more than the prompt holds
	r0.EngineCachedTokens = ptr(8)    // two blocks of four tokens
	r1 := record(1, "e1", both, 1, 2)
	r1.EngineCachedTokens = ptr(3) // less than one block
	r2 := record(2, "e1", both, 1, 2)
	r2.PredictedCachedBlocks = ptr(1)

	l, err := computeLadder([]capture.Record{r0, r1, r2}, 100)
	require.NoError(t, err)
	require.Equal(t, rung{hits: 3, blocks: 4}, l.predicted)
	require.Equal(t, 2, l.predictedRecords)
	require.Equal(t, rung{hits: 2, blocks: 4}, l.engine)
	require.Equal(t, 2, l.engineRecords)

	l, err = computeLadder([]capture.Record{record(0, "e1", both, 1)}, 100)
	require.NoError(t, err)
	require.Equal(t, rung{}, l.predicted)
	require.Equal(t, rung{}, l.engine)
}

func TestLadderSkipsFailedAndEmptyRecords(t *testing.T) {
	// A failed request was served nowhere, so it warms no cache.
	l, err := computeLadder([]capture.Record{
		record(0, "", both, 1, 2),
		record(1, "e1", both),
		record(2, "e1", both, 1, 2),
	}, 100)
	require.NoError(t, err)
	require.Equal(t, rung{hits: 0, blocks: 2}, l.ideal)
	require.Equal(t, 1, l.records)
	require.Equal(t, 1, l.failed)
	require.Equal(t, 1, l.noHashes)
}

func TestLadderRejects(t *testing.T) {
	otherScheme := record(1, "e1", both, 1)
	otherScheme.HashScheme = "precise"
	otherSize := record(1, "e1", both, 1)
	otherSize.BlockSize = 16
	newer := record(1, "e1", both, 1)
	newer.SchemaVersion = capture.SchemaVersion + 1

	for name, tc := range map[string]struct {
		records     []capture.Record
		fleetBlocks int
		wantErr     string
	}{
		"mixed hash schemes":      {[]capture.Record{record(0, "e1", both, 1), otherScheme}, 100, "hash scheme"},
		"mixed block sizes":       {[]capture.Record{record(0, "e1", both, 1), otherSize}, 100, "block size"},
		"unknown schema version":  {[]capture.Record{newer}, 100, "schema version"},
		"no fleet capacity":       {[]capture.Record{record(0, "e1", both, 1)}, 0, "--fleet-kv-blocks"},
		"fewer blocks than pods":  {[]capture.Record{record(0, "e1", both, 1)}, 1, "--fleet-kv-blocks"},
		"no record with a prompt": {[]capture.Record{record(0, "e1", both)}, 100, "no record"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := computeLadder(tc.records, tc.fleetBlocks)
			require.ErrorContains(t, err, tc.wantErr)
		})
	}
}

func ptr(v int) *int { return &v }
