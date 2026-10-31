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

// Package sojourntime carries the per-endpoint sojourn digest snapshot the
// mrl-scorer-hub reads, published by the sojourn-time-observer-hub. The
// snapshot holds two serialized t-digest byte-slices, one over TTFT samples
// (firstChunkAt - dispatchedAt) and one over decode samples
// (endOfStreamAt - firstChunkAt).
package sojourntime

import (
	"bytes"

	"github.com/caio/go-tdigest/v5"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	observerconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sojourntimeobserver/constants"
)

// SojournEstimatorSnapshotDataKey binds the SojournEstimatorSnapshot value type
// to the sojourn-time-observer-hub as its default producer.
var SojournEstimatorSnapshotDataKey = plugin.NewDataKey(
	"SojournEstimatorSnapshotDataKey",
	observerconstants.SojournTimeObserverProducerType,
)

// SojournEstimatorSnapshot carries a serialized copy of the per-endpoint TTFT
// and decode t-digests. Both digest fields are populated only when the digests
// are warm; a cold endpoint has no snapshot entry in its AttributeMap at all.
//
// Values are wall-clock seconds.
type SojournEstimatorSnapshot struct {
	// TtftDigest is a caio/go-tdigest/v5 serialization of the TTFT samples
	// observed on the endpoint, TTFT = firstChunkAt - dispatchedAt.
	TtftDigest []byte
	// DecodeDigest is a caio/go-tdigest/v5 serialization of the decode
	// samples observed on the endpoint,
	// decode = endOfStreamAt - firstChunkAt.
	DecodeDigest []byte
}

// Clone returns an independent copy. The byte slices are deep-copied so a
// subsequent producer flush that overwrites the observer's snapshot pointer
// does not race with in-flight consumer reads.
func (s *SojournEstimatorSnapshot) Clone() fwkdl.Cloneable {
	if s == nil {
		return nil
	}
	cp := &SojournEstimatorSnapshot{}
	if s.TtftDigest != nil {
		cp.TtftDigest = make([]byte, len(s.TtftDigest))
		copy(cp.TtftDigest, s.TtftDigest)
	}
	if s.DecodeDigest != nil {
		cp.DecodeDigest = make([]byte, len(s.DecodeDigest))
		copy(cp.DecodeDigest, s.DecodeDigest)
	}
	return cp
}

// MrlTtft returns the mean-residual-life estimate on the TTFT digest at age a:
// E[TTFT - a | TTFT > a], the expected remaining TTFT for a request that has
// been in the TTFT phase for a seconds without emitting a first token.
//
// Returns 0 when the digest is empty, cannot be parsed, or a exceeds every
// observed centroid.
func (s *SojournEstimatorSnapshot) MrlTtft(a float64) float64 {
	return mrlFromDigest(s.TtftDigest, a)
}

// MrlDecode returns the mean-residual-life estimate on the decode digest at
// age a: E[decode - a | decode > a], the expected remaining decode for a
// request that has been decoding for a seconds without emitting the terminal
// chunk.
//
// Returns 0 when the digest is empty, cannot be parsed, or a exceeds every
// observed centroid.
func (s *SojournEstimatorSnapshot) MrlDecode(a float64) float64 {
	return mrlFromDigest(s.DecodeDigest, a)
}

// MeanDecode returns the overall mean of the decode digest, E[decode]. The
// mrl-scorer-hub uses this as the expected total decode time for a request
// still in the TTFT phase: after TTFT completes, a fresh decode phase
// remains.
//
// MrlDecode(0) equals MeanDecode() in the limit of many samples; MeanDecode
// is a direct helper that skips the tail-average path.
func (s *SojournEstimatorSnapshot) MeanDecode() float64 {
	if len(s.DecodeDigest) == 0 {
		return 0
	}
	td, err := tdigest.FromBytes(bytes.NewReader(s.DecodeDigest))
	if err != nil {
		return 0
	}
	// v5 has no plain Mean(); TrimmedMean over the full support is the mean.
	return td.TrimmedMean(0, 1)
}

// mrlFromDigest tail-averages centroids strictly above a and returns
// weighted_mean - a, clamped to zero. Faithful to the Python reference at
// simulation_study/sim/policies.py:795-808 (TtftDecodeDigest._mrl_from_digest).
func mrlFromDigest(digestBytes []byte, a float64) float64 {
	if len(digestBytes) == 0 {
		return 0
	}
	td, err := tdigest.FromBytes(bytes.NewReader(digestBytes))
	if err != nil {
		return 0
	}
	if a < 0 {
		a = 0
	}
	var total, weighted float64
	td.ForEachCentroid(func(mean float64, count uint64) bool {
		if mean > a {
			total += float64(count)
			weighted += mean * float64(count)
		}
		return true
	})
	if total == 0 {
		return 0
	}
	r := weighted/total - a
	if r < 0 {
		return 0
	}
	return r
}
