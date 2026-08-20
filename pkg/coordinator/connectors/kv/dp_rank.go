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

import (
	"encoding/binary"
	"encoding/json"
	"math"

	"golang.org/x/crypto/blake2s"
)

// remoteDPRankField is the kv_transfer_params key a prefill worker may echo
// back to report which data-parallel rank it actually ran on. The plain NIXL
// P2P protocol this package implements never sets it itself: the decode leg
// addresses the prefill engine directly via remote_engine_id/remote_host/
// remote_port, so which rank handled prefill does not matter for the KV pull.
// resolveDecodeDPRank still checks for it so this connector stays correct if
// a future prefill response ever starts returning one.
const remoteDPRankField = "remote_dp_rank"

// pickDPRank returns a deterministic DP rank for a request as
// blake2s(requestID) mod dpSize, so pinning it as the same HTTP header value
// on both legs of a disaggregated pair keeps a DP>1 backend that shares its
// HTTP port across ranks (e.g. via SO_REUSEPORT) from splitting the pair
// across ranks. dpSize <= 1 returns 0 so single-DP deployments are unaffected.
func pickDPRank(requestID string, dpSize int) int {
	if dpSize <= 1 {
		return 0
	}
	h, err := blake2s.New256(nil)
	if err != nil {
		// Only fails on invalid key length, never for nil; fail safe to rank 0.
		return 0
	}
	_, _ = h.Write([]byte(requestID))
	sum := h.Sum(nil)
	return int(binary.BigEndian.Uint64(sum[:8]) % uint64(dpSize))
}

// resolveDecodeDPRank picks the DP rank for the decode leg. It prefers the
// rank the prefill leg reported in its kv_transfer_params (remoteDPRankField),
// but only when that value is a valid integer in [0, dpSize); otherwise it
// falls back to the deterministic hash of the request id. The second return
// value reports whether the prefill-returned rank was used.
func resolveDecodeDPRank(prefillKV any, requestID string, dpSize int) (rank int, usedReturned bool) {
	fallback := pickDPRank(requestID, dpSize)
	if dpSize <= 1 {
		return fallback, false
	}
	pkv, ok := prefillKV.(map[string]any)
	if !ok {
		return fallback, false
	}
	rv, present := pkv[remoteDPRankField]
	if !present {
		return fallback, false
	}
	if f, ok := rv.(float64); ok && f != math.Trunc(f) {
		return fallback, false
	}
	if ri, ok := toInt(rv); ok && ri >= 0 && ri < dpSize {
		return ri, true
	}
	return fallback, false
}

// toInt converts a JSON number value (float64, int, or json.Number) to int.
func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case float64:
		return int(n), true
	case json.Number:
		i, err := n.Int64()
		return int(i), err == nil
	}
	return 0, false
}
