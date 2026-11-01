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

package sojourntime

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/caio/go-tdigest/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// digestBytes builds a digest from samples and returns its serialization.
func digestBytes(t *testing.T, samples []float64) []byte {
	t.Helper()
	td, err := tdigest.New(tdigest.Compression(500))
	require.NoError(t, err)
	for _, s := range samples {
		require.NoError(t, td.Add(s))
	}
	b, err := td.AsBytes()
	require.NoError(t, err)
	return b
}

// TestMrlTtft_EmptyBytes checks graceful degradation when a scorer reads a
// snapshot whose digest is empty (should not happen in practice — the observer
// only flushes warm digests — but defense in depth).
func TestMrlTtft_EmptyBytes(t *testing.T) {
	s := &SojournEstimatorSnapshot{}
	assert.Equal(t, 0.0, s.MrlTtft(0))
	assert.Equal(t, 0.0, s.MrlTtft(1.0))
	assert.Equal(t, 0.0, s.MrlDecode(0))
	assert.Equal(t, 0.0, s.MeanDecode())
}

// TestMrlTtft_CorruptBytes checks graceful degradation on undecodable bytes.
func TestMrlTtft_CorruptBytes(t *testing.T) {
	s := &SojournEstimatorSnapshot{TtftDigest: []byte("not a digest")}
	assert.Equal(t, 0.0, s.MrlTtft(0))
}

// TestMrlTtft_ExponentialMemorylessness verifies the defining behaviour of the
// exponential distribution: mrl(a) = mu for all a. Feeds 3000 Exp(mu=1)
// samples and asserts mrl at several ages is within 20% of 1.0.
func TestMrlTtft_ExponentialMemorylessness(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 1))
	samples := make([]float64, 3000)
	for i := range samples {
		samples[i] = r.ExpFloat64() // mean 1.0
	}
	s := &SojournEstimatorSnapshot{TtftDigest: digestBytes(t, samples)}

	for _, a := range []float64{0, 0.1, 0.5, 1.0, 2.0, 3.0} {
		mrl := s.MrlTtft(a)
		relErr := math.Abs(mrl-1.0) / 1.0
		assert.LessOrEqualf(t, relErr, 0.20,
			"MrlTtft(%.2f)=%.3f, expected ~1.0 under memorylessness", a, mrl)
	}
}

// TestMrlDecode_HeavyTailLengthBiased verifies that under a log-normal decode
// distribution, mrl grows with age. Length-biased survival: an old-still-alive
// realization is more likely to be one of the intrinsically long ones.
func TestMrlDecode_HeavyTailLengthBiased(t *testing.T) {
	// Log-normal with mu=0, sigma=1, so E[X] = exp(0.5) ~= 1.648.
	r := rand.New(rand.NewPCG(2, 2))
	samples := make([]float64, 5000)
	for i := range samples {
		// Box-Muller not built into rand/v2; use NormFloat64 which gives a
		// standard normal, then exponentiate for log-normal.
		samples[i] = math.Exp(r.NormFloat64())
	}
	s := &SojournEstimatorSnapshot{DecodeDigest: digestBytes(t, samples)}

	mrl0 := s.MrlDecode(0)
	mrl1 := s.MrlDecode(1)
	mrl3 := s.MrlDecode(3)
	assert.Greater(t, mrl1, mrl0, "MrlDecode should grow with age for a heavy-tailed distribution")
	assert.Greater(t, mrl3, mrl1)
}

// TestMrl_ZeroAboveMaxObservation checks the edge case where the query age
// exceeds every observed centroid — the slot is older than every sample ever
// seen, so its expected remaining is treated as 0.
func TestMrl_ZeroAboveMaxObservation(t *testing.T) {
	s := &SojournEstimatorSnapshot{TtftDigest: digestBytes(t, []float64{0.1, 0.2, 0.5, 1.0})}
	assert.Equal(t, 0.0, s.MrlTtft(2.0), "age above max sample should yield 0")
	assert.Equal(t, 0.0, s.MrlTtft(100.0))
}

// TestMrlDecode_ZeroEqualsMean confirms the identity mrl(0) == mean in the
// limit of many samples. mrl(0) = E[S - 0 | S > 0] = E[S] over the whole
// support.
func TestMrlDecode_ZeroEqualsMean(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 3))
	samples := make([]float64, 5000)
	for i := range samples {
		samples[i] = r.ExpFloat64() * 2.0 // mean ~= 2.0
	}
	s := &SojournEstimatorSnapshot{DecodeDigest: digestBytes(t, samples)}

	mrl0 := s.MrlDecode(0)
	mean := s.MeanDecode()
	relErr := math.Abs(mrl0-mean) / mean
	assert.LessOrEqualf(t, relErr, 0.02, "mrl(0)=%.3f vs Mean=%.3f, expected identity to within seed noise", mrl0, mean)
}

// TestClone_DeepCopiesBytes verifies Clone returns an independent copy of the
// digest slices — mutating the source after Clone must not affect the clone.
func TestClone_DeepCopiesBytes(t *testing.T) {
	src := &SojournEstimatorSnapshot{
		TtftDigest:   []byte{1, 2, 3, 4},
		DecodeDigest: []byte{5, 6, 7, 8},
	}
	cloned := src.Clone().(*SojournEstimatorSnapshot)

	// Mutate source in place.
	src.TtftDigest[0] = 0xff
	src.DecodeDigest[0] = 0xff

	assert.Equal(t, byte(1), cloned.TtftDigest[0], "clone must be independent of source mutation")
	assert.Equal(t, byte(5), cloned.DecodeDigest[0])
}

// TestClone_NilSnapshot returns nil for a nil receiver.
func TestClone_NilSnapshot(t *testing.T) {
	var s *SojournEstimatorSnapshot
	assert.Nil(t, s.Clone())
}

// TestClone_NilFields handles nil TtftDigest/DecodeDigest fields.
func TestClone_NilFields(t *testing.T) {
	src := &SojournEstimatorSnapshot{}
	cloned := src.Clone().(*SojournEstimatorSnapshot)
	assert.Nil(t, cloned.TtftDigest)
	assert.Nil(t, cloned.DecodeDigest)
}
