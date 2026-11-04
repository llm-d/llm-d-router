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

package disaggregatedsetrollout

import (
	"context"
	"encoding/binary"
	"math"

	"github.com/cespare/xxhash/v2"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

// affinityKey returns a deterministic hash key for the request and the source
// it came from, or sourceRandom when affinity is off or the request has nothing
// to key on.
//
// The first session identifier resolved from Affinity.Sources takes
// precedence over the text conversation prefix.
func (c *Screener) affinityKey(ctx context.Context, request *fwksched.InferenceRequest) (uint64, selectionSource) {
	if c.config.RevisionGating.affinityMode() != AffinityModePrefix || request == nil {
		return 0, sourceRandom
	}
	if session := sessionaffinity.ResolveSessionID(ctx, request, c.config.RevisionGating.Affinity.Sources); session != "" {
		return xxhash.Sum64String(session), sourceSession
	}
	var digest xxhash.Digest
	digest.Reset()
	if !request.Body.WriteConversationPrefix(&digest) {
		return 0, sourceRandom
	}
	return digest.Sum64(), sourcePrefix
}

// pickRendezvousRevision selects a revision in proportion to shares using
// weighted rendezvous hashing (-ln(u) / share, lowest wins). Changing one
// revision's share or coverage only moves keys to or from that revision; keys
// on the remaining revisions stay put regardless of revision names or count.
func pickRendezvousRevision(shares map[string]float64, key uint64) string {
	best := ""
	bestScore := math.Inf(1)
	for revision, share := range shares {
		if share <= 0 {
			continue
		}
		score := rendezvousScore(key, revision, share)
		if best == "" || score < bestScore || (score == bestScore && revision < best) {
			best = revision
			bestScore = score
		}
	}
	return best
}

func rendezvousScore(key uint64, revision string, share float64) float64 {
	var digest xxhash.Digest
	digest.Reset()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], key)
	_, _ = digest.Write(buf[:])
	_, _ = digest.WriteString(revision)
	// Map the top 53 hash bits onto (0, 1] so math.Log is finite and <= 0.
	u := float64((digest.Sum64()>>11)+1) / (1 << 53)
	return -math.Log(u) / share
}
