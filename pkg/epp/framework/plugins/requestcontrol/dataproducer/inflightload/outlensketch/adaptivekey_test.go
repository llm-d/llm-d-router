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
	"sync"
	"testing"
)

func testKMeansConfig(k, dim, warmupN int) adaptiveKMeansConfig {
	return adaptiveKMeansConfig{
		k:           k,
		dim:         dim,
		iters:       25,
		warmupN:     warmupN,
		minMembers:  1,
		minCoverage: 0.9,
		seed:        5,
	}
}

func TestAdaptiveColdBeforeWarmup(t *testing.T) {
	a := newAdaptiveKMeans(testKMeansConfig(2, 2, 10))
	if _, ready := a.cluster([]float32{1, 1}); ready {
		t.Fatal("cluster must not be ready before warm-up completes")
	}
	if a.ready() {
		t.Fatal("ready() must be false before warm-up completes")
	}
}

func TestAdaptiveWarmupFitsTwoClusters(t *testing.T) {
	a := newAdaptiveKMeans(testKMeansConfig(2, 2, 20))
	// Two well-separated clouds; alternate so the batch fills at 20.
	for i := 0; i < 10; i++ {
		a.observe([]float32{0, 0})
		a.observe([]float32{10, 10})
	}
	if !a.ready() {
		t.Fatal("AdaptiveKey should be online after a full, well-covered batch")
	}
	// Points near each cloud must land in different clusters.
	near0, ready0 := a.cluster([]float32{0.1, -0.1})
	near10, ready10 := a.cluster([]float32{9.9, 10.1})
	if !ready0 || !ready10 {
		t.Fatal("cluster must be ready after warm-up")
	}
	if near0 == near10 {
		t.Errorf("separated points landed in the same cluster (%d)", near0)
	}
}

func TestAdaptiveSkewedBatchStaysWarmup(t *testing.T) {
	// All points identical: only one centroid gets members, so coverage (needs 90%
	// of 2 centroids) fails and the key keeps collecting.
	a := newAdaptiveKMeans(testKMeansConfig(2, 2, 20))
	for i := 0; i < 20; i++ {
		a.observe([]float32{3, 3})
	}
	if a.ready() {
		t.Error("a degenerate single-cluster batch must not satisfy the coverage check")
	}
}

func TestAdaptiveNudgeIsCumulativeMean(t *testing.T) {
	// Drive the online nudge directly: a centroid seeded as one point at the origin,
	// then two observations, must become the running mean of all three points.
	a := newAdaptiveKMeans(testKMeansConfig(1, 2, 1))
	a.phase = phaseOnline
	a.centroids = [][]float64{{0, 0}}
	a.counts = []float64{1}

	a.observe([]float32{2, 2})
	if got := a.centroids[0]; !approxVec(got, []float64{1, 1}) {
		t.Errorf("after one nudge centroid = %v, want [1 1]", got)
	}
	a.observe([]float32{4, 4})
	if got := a.centroids[0]; !approxVec(got, []float64{2, 2}) {
		t.Errorf("after two nudges centroid = %v, want [2 2] (mean of 0,2,4)", got)
	}
	if a.counts[0] != 3 {
		t.Errorf("centroid count = %v, want 3", a.counts[0])
	}
}

func TestAdaptiveConcurrentObserveCluster(t *testing.T) {
	a := newAdaptiveKMeans(testKMeansConfig(4, 3, 100))
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				v := []float32{float32((i + w) % 4), float32(i % 3), float32(w)}
				a.observe(v)
				a.cluster(v)
			}
		}(w)
	}
	wg.Wait()
}

func approxVec(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-9 {
			return false
		}
	}
	return true
}
