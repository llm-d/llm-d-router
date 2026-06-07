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
	"encoding/binary"

	"golang.org/x/crypto/blake2s"
)

// pickDPRank picks a stable, deterministic DP rank for a request by
// hashing the canonical request UUID with BLAKE2s and reducing mod
// dpSize.
//
// Why this exists:
//
// In a disaggregated P/D deployment with vLLM data-parallel size > 1,
// vLLM's OpenAI server runs N ApiServer processes that share the same
// listening port via SO_REUSEPORT. The kernel picks one process per
// inbound TCP connection at random, so two HTTP calls with the same
// X-Request-Id (the prefill leg and the decode leg of one disagg pair)
// can land on different DP ranks. That mismatch breaks MoRI-IO: the
// producer-side handshake addresses a peer (host, port_base + rank)
// but the consumer is listening on a different rank's port, so the
// notify never arrives and the request hangs until the deferred-write
// timeout (default 300s) trips.
//
// The fix is for the sidecar -- which sees both legs and stamps the
// same canonical request UUID on each -- to pin both legs to a stable
// rank H = hash(uuid) mod dpSize and tell vLLM to use that rank
// directly. We tell vLLM via:
//
//  1. The X-Data-Parallel-Rank HTTP header (consumed by vLLM's
//     OpenAI server, which forwards it as data_parallel_rank to
//     AsyncLLM.add_request, bypassing the round-robin DP load
//     balancer).
//
//  2. The remote_dp_rank field in kv_transfer_params on the decode
//     leg, paired with remote_dp_rank_override=true so the
//     decode-side MoRIIOConnectorScheduler.request_finished honours
//     the sidecar value instead of recomputing its own hash.
//
// Determinism contract:
//
//   - Same requestID + same dpSize -> same rank, across processes,
//     across pods, across sidecar restarts. (BLAKE2s is a pure
//     function and we feed only requestID into it.)
//
//   - dpSize <= 1 short-circuits to 0 so single-DP deployments are
//     bit-identical to pre-patch behaviour.
//
//   - Cross-language note: vLLM uses hashlib.blake2s(rid,
//     digest_size=8) while we use blake2s.New256() truncated to 8
//     bytes. The two algorithms produce DIFFERENT digests because
//     BLAKE2's parameter block encodes the output length and
//     influences the IV, so digest_size=8 is not a prefix of
//     digest_size=32. This is intentional and harmless: the sidecar
//     stamps the rank into the wire (header + override field) and
//     vLLM uses that value verbatim, so vLLM's own hash never runs
//     when the sidecar is in path. The "agreement" required between
//     the two legs is sidecar-internal: pickDPRank is called twice
//     on the same uuidStr in the same Go binary and is deterministic.
func pickDPRank(requestID string, dpSize int) int {
	if dpSize <= 1 {
		return 0
	}
	h, err := blake2s.New256(nil)
	if err != nil {
		// blake2s.New256(nil) only returns an error when the key
		// length is invalid, which it never is for nil. Treat any
		// future hypothetical error as "fail safe to rank 0" rather
		// than panicking and taking down the proxy.
		return 0
	}
	_, _ = h.Write([]byte(requestID))
	sum := h.Sum(nil)
	return int(binary.BigEndian.Uint64(sum[:8]) % uint64(dpSize))
}
