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

package outlensketch

import (
	"math"
	"math/rand"
	"sync"
)

// AdaptiveKey: online k-means over prompt embeddings whose centroids never hard
// freeze. The cluster a prompt falls into is its key component -- "what kind of
// request is this" -- so a key owns the output-length distribution of semantically
// similar prompts.
//
// Warm-up: the first observations are buffered; once a full batch has accumulated,
// Lloyd's algorithm fits the centroids. The batch is accepted only when enough
// centroids have members (so skewed early traffic does not freeze blind centroids);
// otherwise collection continues and the next full batch is tried. Before the
// centroids are ready the caller keys on the signal signature alone, so the
// histogram is usable from the first request.
//
// Online: every later observation nudges its nearest centroid by a cumulative-mean
// step with learning rate 1/(n+1), so the step decays on its own as a centroid
// accumulates members. The centroids converge without a freeze trigger and track
// slow workload drift. On stationary traffic the result matches a one-shot frozen
// fit, so adapting costs nothing at rest and only helps under drift.
const (
	phaseWarmup = iota
	phaseOnline
)

const lloydConvergenceEps = 1e-6

// adaptiveKMeansConfig carries the tunables for the adaptive key.
type adaptiveKMeansConfig struct {
	k           int     // number of centroids
	dim         int     // embedding dimension
	iters       int     // Lloyd iterations per fit
	warmupN     int     // batch size that triggers a fit attempt
	minMembers  int     // per-centroid member floor for the readiness check
	minCoverage float64 // fraction of centroids that must meet the floor
	seed        int64   // deterministic Lloyd initialization
}

// adaptiveKMeans is the mutable centroid state behind an RWMutex: predictions take
// the read lock for the nearest-centroid lookup, observations take the write lock to
// buffer or nudge.
type adaptiveKMeans struct {
	cfg adaptiveKMeansConfig

	mu        sync.RWMutex
	phase     int
	buffer    [][]float32
	centroids [][]float64 // k x dim, nil until warm
	counts    []float64   // per-centroid cumulative member count
}

func newAdaptiveKMeans(cfg adaptiveKMeansConfig) *adaptiveKMeans {
	return &adaptiveKMeans{cfg: cfg, phase: phaseWarmup}
}

// cluster returns the nearest-centroid id for a prompt embedding. ready is false
// while the centroids are still warming up, signaling the caller to key on signals
// alone.
func (a *adaptiveKMeans) cluster(v []float32) (id int, ready bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.phase != phaseOnline {
		return 0, false
	}
	return nearest(v, a.centroids), true
}

// observe folds one completed request's embedding into the centroids: during
// warm-up it buffers, and triggers a fit attempt once a full batch has accumulated;
// once online it nudges the nearest centroid. The Lloyd fit runs off-lock on a
// buffer snapshot so a warm-up fit never stalls concurrent predictions.
func (a *adaptiveKMeans) observe(v []float32) {
	a.mu.Lock()
	if a.phase == phaseOnline {
		a.nudgeLocked(v)
		a.mu.Unlock()
		return
	}
	a.buffer = append(a.buffer, cloneFloat32(v))
	if len(a.buffer)%a.cfg.warmupN != 0 {
		a.mu.Unlock()
		return
	}
	// Copy the slice header's pointers so the fit reads a stable batch while later
	// observations may append to (and reallocate) a.buffer. Buffered vectors are
	// never mutated after being appended, so sharing the inner slices is safe.
	snapshot := make([][]float32, len(a.buffer))
	copy(snapshot, a.buffer)
	a.mu.Unlock()

	centroids, counts, ok := a.fitLloyd(snapshot)
	if !ok {
		return // too few populated centroids; keep collecting for the next batch
	}
	a.mu.Lock()
	if a.phase == phaseWarmup { // a racing observation may have finalized already
		a.centroids, a.counts, a.buffer, a.phase = centroids, counts, nil, phaseOnline
	}
	a.mu.Unlock()
}

// nudgeLocked moves the nearest centroid toward v by the cumulative-mean step. The
// caller holds the write lock.
func (a *adaptiveKMeans) nudgeLocked(v []float32) {
	c := nearest(v, a.centroids)
	inv := 1.0 / (a.counts[c] + 1.0)
	centroid := a.centroids[c]
	for j, x := range v {
		centroid[j] += (float64(x) - centroid[j]) * inv
	}
	a.counts[c]++
}

// fitLloyd runs Lloyd's algorithm on the batch and reports whether enough centroids
// have members to go online. counts holds each centroid's final member count, which
// seeds the online 1/(n+1) step.
func (a *adaptiveKMeans) fitLloyd(points [][]float32) (centroids [][]float64, counts []float64, ok bool) {
	k := a.cfg.k
	if len(points) < k {
		k = len(points)
	}
	if k == 0 {
		return nil, nil, false
	}

	rng := rand.New(rand.NewSource(a.cfg.seed))
	centroids = make([][]float64, k)
	for i, p := range rng.Perm(len(points))[:k] {
		centroids[i] = float32ToFloat64(points[p])
	}

	assign := make([]int, len(points))
	for iter := 0; iter < a.cfg.iters; iter++ {
		for i, p := range points {
			assign[i] = nearest(p, centroids)
		}
		sums := make([][]float64, k)
		members := make([]int, k)
		for i := range sums {
			sums[i] = make([]float64, a.cfg.dim)
		}
		for i, p := range points {
			c := assign[i]
			members[c]++
			for j, x := range p {
				sums[c][j] += float64(x)
			}
		}
		var moved float64
		for c := 0; c < k; c++ {
			if members[c] == 0 {
				continue
			}
			var shift float64
			for j := 0; j < a.cfg.dim; j++ {
				nv := sums[c][j] / float64(members[c])
				d := nv - centroids[c][j]
				shift += d * d
				centroids[c][j] = nv
			}
			moved += math.Sqrt(shift)
		}
		if moved < lloydConvergenceEps {
			break
		}
	}

	counts = make([]float64, k)
	for _, p := range points {
		counts[nearest(p, centroids)]++
	}
	warm := 0
	for _, c := range counts {
		if c >= float64(a.cfg.minMembers) {
			warm++
		}
	}
	return centroids, counts, float64(warm)/float64(a.cfg.k) >= a.cfg.minCoverage
}

func (a *adaptiveKMeans) ready() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.phase == phaseOnline
}

// nearest returns the index of the centroid closest to v in squared L2 distance.
func nearest(v []float32, centroids [][]float64) int {
	best := 0
	bestDist := math.Inf(1)
	for c, centroid := range centroids {
		var dist float64
		for j, x := range v {
			d := float64(x) - centroid[j]
			dist += d * d
		}
		if dist < bestDist {
			bestDist, best = dist, c
		}
	}
	return best
}

func cloneFloat32(v []float32) []float32 {
	out := make([]float32, len(v))
	copy(out, v)
	return out
}

func float32ToFloat64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}
