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

package proxy

import (
	"encoding/binary"

	"golang.org/x/crypto/blake2s"
)

// pickDPRank returns a deterministic DP rank for a request as
// blake2s(requestID) mod dpSize. With dpSize > 1, vLLM's API servers share a
// port via SO_REUSEPORT and the kernel may route a disagg pair's two legs to
// different DP ranks; pinning both legs to the same rank keeps the MoRI-IO
// handshake from addressing a peer that is not listening. dpSize <= 1 returns
// 0 so single-DP deployments are unaffected.
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
