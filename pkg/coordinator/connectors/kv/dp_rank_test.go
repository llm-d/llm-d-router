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

import "testing"

// TestPickDPRankSingleDP verifies dpSize <= 1 short-circuits to 0 without
// hashing.
func TestPickDPRankSingleDP(t *testing.T) {
	for _, dpSize := range []int{0, 1, -1} {
		if got := pickDPRank("any-request-id", dpSize); got != 0 {
			t.Errorf("pickDPRank(_, %d) = %d; want 0", dpSize, got)
		}
	}
}

// TestPickDPRankDeterministicAndInRange verifies the two load-bearing
// invariants: the same requestID+dpSize always returns the same rank
// (otherwise the prefill and decode legs of one pair could land on
// different ranks), and the result is always in [0, dpSize).
func TestPickDPRankDeterministicAndInRange(t *testing.T) {
	requestIDs := []string{"req-1", "cmpl-abc-123", "00000000-0000-0000-0000-000000000000", ""}
	for _, dpSize := range []int{2, 3, 8, 16} {
		for _, rid := range requestIDs {
			first := pickDPRank(rid, dpSize)
			if first < 0 || first >= dpSize {
				t.Errorf("pickDPRank(%q, %d) = %d; want in [0, %d)", rid, dpSize, first, dpSize)
			}
			for i := range 3 {
				if got := pickDPRank(rid, dpSize); got != first {
					t.Errorf("pickDPRank(%q, %d) = %d on call %d, want %d (must be deterministic)", rid, dpSize, got, i, first)
				}
			}
		}
	}
}

func TestResolveDecodeDPRank(t *testing.T) {
	const dpSize = 8
	const rid = "cmpl-rank-test"
	hashFallback := pickDPRank(rid, dpSize)

	cases := []struct {
		name         string
		prefillKV    any
		dpSize       int
		wantRank     int
		wantReturned bool
	}{
		{name: "valid returned rank (float64)", prefillKV: map[string]any{remoteDPRankField: float64(3)}, dpSize: dpSize, wantRank: 3, wantReturned: true},
		{name: "valid returned rank (int)", prefillKV: map[string]any{remoteDPRankField: 5}, dpSize: dpSize, wantRank: 5, wantReturned: true},
		{name: "zero is valid", prefillKV: map[string]any{remoteDPRankField: float64(0)}, dpSize: dpSize, wantRank: 0, wantReturned: true},
		{name: "omitted falls back to hash", prefillKV: map[string]any{}, dpSize: dpSize, wantRank: hashFallback, wantReturned: false},
		{name: "nil kv falls back to hash", prefillKV: nil, dpSize: dpSize, wantRank: hashFallback, wantReturned: false},
		{name: "non-numeric falls back to hash", prefillKV: map[string]any{remoteDPRankField: "two"}, dpSize: dpSize, wantRank: hashFallback, wantReturned: false},
		{name: "fractional falls back to hash (no truncation)", prefillKV: map[string]any{remoteDPRankField: float64(3.5)}, dpSize: dpSize, wantRank: hashFallback, wantReturned: false},
		{name: "out-of-range high falls back to hash", prefillKV: map[string]any{remoteDPRankField: float64(8)}, dpSize: dpSize, wantRank: hashFallback, wantReturned: false},
		{name: "negative falls back to hash", prefillKV: map[string]any{remoteDPRankField: float64(-1)}, dpSize: dpSize, wantRank: hashFallback, wantReturned: false},
		{name: "single-DP always rank 0", prefillKV: map[string]any{remoteDPRankField: float64(3)}, dpSize: 1, wantRank: 0, wantReturned: false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotRank, gotReturned := resolveDecodeDPRank(c.prefillKV, rid, c.dpSize)
			if gotRank != c.wantRank || gotReturned != c.wantReturned {
				t.Errorf("resolveDecodeDPRank(%v, %q, %d) = (%d, %t); want (%d, %t)",
					c.prefillKV, rid, c.dpSize, gotRank, gotReturned, c.wantRank, c.wantReturned)
			}
		})
	}
}
