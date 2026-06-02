/*
Copyright 2025 The llm-d Authors.

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
	"crypto/rand"
	"encoding/hex"
	"testing"
)

// TestPickDPRankSingleDP verifies that dpSize <= 1 short-circuits to 0,
// matching the pre-patch behaviour for non-DP deployments. We must not
// hash and reduce mod 1 (which would also yield 0 but at the cost of a
// useless allocation on every request).
func TestPickDPRankSingleDP(t *testing.T) {
	for _, dpSize := range []int{0, 1, -1} {
		if got := pickDPRank("any-request-id", dpSize); got != 0 {
			t.Errorf("pickDPRank(_, %d) = %d; want 0", dpSize, got)
		}
	}
}

// TestPickDPRankDeterministic verifies the core invariant: the same
// requestID + dpSize always returns the same rank. Without this, the
// prefill and decode legs of one disagg pair would land on different
// DP ranks and the MoRI-IO handshake would deadlock.
func TestPickDPRankDeterministic(t *testing.T) {
	for i := 0; i < 64; i++ {
		var b [16]byte
		_, _ = rand.Read(b[:])
		rid := hex.EncodeToString(b[:])
		first := pickDPRank(rid, 8)
		for j := 0; j < 4; j++ {
			if got := pickDPRank(rid, 8); got != first {
				t.Errorf("pickDPRank(%q, 8) returned %d on call %d but %d on call 0",
					rid, got, j+1, first)
			}
		}
	}
}

// TestPickDPRankRange verifies the output is always in [0, dpSize).
// A faulty modulo (e.g. signed division on a negative seed) would
// produce out-of-range ranks that the kv_transfer_params consumer
// would silently treat as rank 0 or panic at port-base computation.
func TestPickDPRankRange(t *testing.T) {
	for _, dpSize := range []int{2, 4, 8, 16, 32, 64} {
		for i := 0; i < 256; i++ {
			var b [8]byte
			_, _ = rand.Read(b[:])
			rid := hex.EncodeToString(b[:])
			got := pickDPRank(rid, dpSize)
			if got < 0 || got >= dpSize {
				t.Errorf("pickDPRank(%q, %d) = %d; want in [0, %d)",
					rid, dpSize, got, dpSize)
			}
		}
	}
}

// TestPickDPRankReferenceValues pins the exact algorithm: BLAKE2s-256
// of the requestID, top 8 bytes interpreted as a big-endian uint64,
// reduced mod dpSize. The reference values were generated with:
//
//	python3 -c "
//	import hashlib
//	def pick(rid, dp):
//	    d = hashlib.blake2s(rid.encode(), digest_size=32).digest()
//	    return int.from_bytes(d[:8], 'big') % dp
//	"
//
// (Note: Python's hashlib.blake2s with digest_size=32 is equivalent to
// Go's golang.org/x/crypto/blake2s.New256(); the digest_size=8 variant
// that vLLM uses is a DIFFERENT BLAKE2s instance and would not match
// these values. See dp_rank.go for why this divergence is intentional.)
//
// Pinning these exact values makes accidental algorithm changes (e.g.
// switching to little-endian, or swapping in SHA-256, or reading sum[24:32]
// instead of sum[:8]) loud test failures rather than silent rebalancing
// of every in-flight request to a different rank.
func TestPickDPRankReferenceValues(t *testing.T) {
	cases := []struct {
		rid    string
		dpSize int
		want   int
	}{
		{rid: "abc", dpSize: 8, want: 2},
		{rid: "cmpl-foo-0", dpSize: 8, want: 5},
		{rid: "00000000-0000-0000-0000-000000000000", dpSize: 8, want: 6},
		{rid: "req-9f8a-2026-1p1d", dpSize: 8, want: 2},
		{rid: "cmpl-deadbeef-cafe-1234", dpSize: 8, want: 4},
		{rid: "abc", dpSize: 2, want: 0},
		{rid: "abc", dpSize: 4, want: 2},
		{rid: "abc", dpSize: 16, want: 2},
	}
	for _, c := range cases {
		if got := pickDPRank(c.rid, c.dpSize); got != c.want {
			t.Errorf("pickDPRank(%q, %d) = %d; want %d",
				c.rid, c.dpSize, got, c.want)
		}
	}
}

// TestPickDPRankUniform verifies that the rank distribution is roughly
// uniform across a large random sample. A skewed distribution would
// concentrate prefill load on a subset of ranks, defeating the point
// of data-parallel.
//
// Tolerance: each bucket should hold (N/dpSize) +/- 25% over N=10000.
// The expected std-dev for a uniform distribution is ~sqrt(N/dpSize *
// (1 - 1/dpSize)), so a 25% window is many sigmas away from the mean
// and will not flake. Empirically (sample run 2026-06-01) the actual
// counts for dpSize=8, N=10000 ranged 1166..1322 (mean 1250, target
// window 937..1562).
func TestPickDPRankUniform(t *testing.T) {
	const N = 10000
	const dpSize = 8
	const tol = 0.25
	expected := N / dpSize
	lo := int(float64(expected) * (1.0 - tol))
	hi := int(float64(expected) * (1.0 + tol))
	counts := make([]int, dpSize)
	for i := 0; i < N; i++ {
		var b [16]byte
		_, _ = rand.Read(b[:])
		counts[pickDPRank(hex.EncodeToString(b[:]), dpSize)]++
	}
	for r, c := range counts {
		if c < lo || c > hi {
			t.Errorf("rank %d count = %d; want in [%d, %d] (expected %d, tol %.0f%%)",
				r, c, lo, hi, expected, tol*100)
		}
	}
}
