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

package thunderagent

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// The flow controller's saturation gate blocks whole bands, including the
// turns of admitted programs, so ThunderAgent never gates there: admission
// is per-program in Pick, fed by the fit view this hook maintains.
func TestSaturationNeverGates(t *testing.T) {
	a := newTestAgent(testConfig())
	sched := newTestEndpoints("pod1")
	dl := dlEndpoints("pod1")

	assert.Equal(t, 0.0, a.Saturation(context.Background(), nil))
	assert.Equal(t, 0.0, a.Saturation(context.Background(), dl))

	// Even a working set far past capacity does not gate: blocking admitted
	// programs' turns cannot shrink committed footprints, only completing
	// trajectories can.
	seedProgram(t, a, "program-a", sched[0], 5000)
	assert.Equal(t, 0.0, a.Saturation(context.Background(), dl))
}

func TestSaturationRefreshesFitView(t *testing.T) {
	cfg := testConfig()
	cfg.BufferTokensPerProgram = 100
	a := newTestAgent(cfg)
	sched := newTestEndpoints("pod1")
	dl := dlEndpoints("pod1")

	seedProgram(t, a, "program-a", sched[0], 350)
	a.Saturation(context.Background(), dl)

	pod := sched[0].GetMetadata().ID.String()
	a.table.mu.Lock()
	snap := a.table.snapshot[pod]
	a.table.mu.Unlock()
	require.NotNil(t, snap)
	assert.Equal(t, 1000.0, snap.capacity)
	assert.Equal(t, 450.0, snap.tokens, "350 committed plus the 100-token growth buffer")
}

func TestSaturationFitViewUsesRealCapacity(t *testing.T) {
	a := newTestAgent(testConfig())
	dl := []fwkdl.Endpoint{endpointWithCacheInfo("pod1", 16, 500)}

	a.Saturation(context.Background(), dl)

	a.table.mu.Lock()
	snap := a.table.snapshot["default/pod1"]
	a.table.mu.Unlock()
	require.NotNil(t, snap)
	assert.Equal(t, 8000.0, snap.capacity, "block_size * num_gpu_blocks overrides the configured fallback")
}

func TestSaturationRequireRealCapacity(t *testing.T) {
	cfg := testConfig()
	cfg.RequireRealCapacity = true
	a := newTestAgent(cfg)

	a.Saturation(context.Background(), dlEndpoints("pod1"))

	a.table.mu.Lock()
	_, ok := a.fitPodLocked(100, time.Now())
	a.table.mu.Unlock()
	assert.False(t, ok, "a pod without scraped capacity has no room for new programs")
}

// Per-stage detector calls each carry only that stage's pods; one stage's
// call must not erase the fit view of the other stage's pods.
func TestSaturationPerStageCallsKeepOtherPods(t *testing.T) {
	a := newTestAgent(testConfig())

	a.Saturation(context.Background(), dlEndpoints("prefill-pod"))
	a.Saturation(context.Background(), dlEndpoints("decode-pod"))

	a.table.mu.Lock()
	_, hasPrefill := a.table.snapshot["default/prefill-pod"]
	_, hasDecode := a.table.snapshot["default/decode-pod"]
	a.table.mu.Unlock()
	assert.True(t, hasPrefill)
	assert.True(t, hasDecode)
}

func TestSaturationDropsStalePods(t *testing.T) {
	a := newTestAgent(testConfig())

	a.Saturation(context.Background(), dlEndpoints("pod1"))
	a.table.mu.Lock()
	a.table.snapshot["default/pod1"].updatedAt = time.Now().Add(-2 * snapshotStaleAfter)
	a.table.mu.Unlock()

	a.Saturation(context.Background(), dlEndpoints("pod2"))

	a.table.mu.Lock()
	_, ok := a.table.snapshot["default/pod1"]
	a.table.mu.Unlock()
	assert.False(t, ok, "a pod no stage has reported within snapshotStaleAfter leaves the fit view")
}

// A pod starts on the configured fallback and switches to the scraped value
// once the metrics extractor has run. Reporting both would make
// "did we get real capacity" unanswerable.
func TestCapacitySourceSeriesIsExclusive(t *testing.T) {
	a := newTestAgent(testConfig())
	pod := "default/pod1"

	a.metrics.setPodCapacity(pod, capacityFallback, 1000)
	a.metrics.setPodCapacity(pod, capacityReal, 8000)

	assert.Equal(t, 8000.0, testutil.ToFloat64(a.metrics.podCapacityTokens.WithLabelValues(pod, string(capacityReal))))
	assert.Equal(t, 1, testutil.CollectAndCount(a.metrics.podCapacityTokens),
		"only the current source should have a series")
}
