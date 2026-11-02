/*
Copyright 2025 The Kubernetes Authors.
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

// Package picker provides the standard implementations of the Picker interface for the scheduling
// framework. These plugins represent the final phase of the scheduling cycle, selecting target
// endpoints from the scored candidates.
//
// For detailed descriptions of the available pickers and their configuration, see the package
// README.md.
package picker

import (
	"hash/fnv"
	"math"
	"math/rand/v2"
	"sync"
	"time"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// DefaultMaxNumOfEndpoints is the fallback maximum number of endpoints to pick if not specified
	// in the configuration.
	DefaultMaxNumOfEndpoints = 1
)

// PickerParameters defines the common parameters for all pickers
type PickerParameters struct {
	MaxNumOfEndpoints int `json:"maxNumOfEndpoints"`
}

// maxTieCounters caps the per-pool state as pods come and go.
const maxTieCounters = 1024

type lockedRand struct {
	mu       sync.Mutex
	rand     *rand.Rand
	counters map[uint64]int // round-robin position per candidate pool
}

func newLockedRand() *lockedRand {
	seed := uint64(time.Now().UnixNano())

	return &lockedRand{
		rand: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)), //#nosec G404 -- endpoint selection, not security-sensitive
	}
}

func (r *lockedRand) Float64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.rand.Float64()
}

func (r *lockedRand) Shuffle(n int, swap func(i, j int)) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.rand.Shuffle(n, swap)
}

// NextCounter returns the next round-robin position for the candidate pool poolKey.
// Prefill and decode share one picker but rotate independently unless they see the same pods.
func (r *lockedRand) NextCounter(poolKey uint64) int {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.counters == nil || len(r.counters) >= maxTieCounters {
		r.counters = make(map[uint64]int)
	}
	c := r.counters[poolKey]
	r.counters[poolKey] = (c + 1) % math.MaxInt
	return c
}

// PoolKey hashes the endpoint names of a candidate set, independent of their order.
func PoolKey(scoredEndpoints []*fwksched.ScoredEndpoint) uint64 {
	var key uint64
	var buf []byte
	h := fnv.New64a()
	for _, ep := range scoredEndpoints {
		id := ep.GetMetadata().GetNamespacedName()
		buf = append(append(append(buf[:0], id.Namespace...), '/'), id.Name...)
		h.Reset()
		_, _ = h.Write(buf)
		key += h.Sum64()
	}
	return key
}

// PickerRand is a thread-safe random number generator shared by all pickers to avoid seeding
// overhead and ensure controlled randomization.
var PickerRand = newLockedRand()

// ShuffleScoredEndpoints randomly shuffles the scored endpoints.
func ShuffleScoredEndpoints(scoredEndpoints []*fwksched.ScoredEndpoint) {
	PickerRand.Shuffle(len(scoredEndpoints), func(i, j int) {
		scoredEndpoints[i], scoredEndpoints[j] = scoredEndpoints[j], scoredEndpoints[i]
	})
}

// RotateScoredEndpoints provides deterministic round-robin rotation for
// equal-score endpoints. Under concurrent load with prefix cache affinity,
// multiple requests can see identical scores simultaneously. Pure random
// shuffling causes them to converge on the same winner, creating severe
// routing imbalance. The caller supplies a counter drawn once per request,
// ensuring independent rotation across multiple tie groups.
func RotateScoredEndpoints(scoredEndpoints []*fwksched.ScoredEndpoint, counter int) {
	n := len(scoredEndpoints)
	if n <= 1 {
		return
	}

	shift := counter % n
	if shift > 0 {
		rotated := make([]*fwksched.ScoredEndpoint, n)
		copy(rotated, scoredEndpoints[shift:])
		copy(rotated[n-shift:], scoredEndpoints[:shift])
		copy(scoredEndpoints, rotated)
	}
}
