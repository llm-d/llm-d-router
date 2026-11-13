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

package programaware

import (
	"context"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// BenchmarkPick_PrewarmedPrograms measures Pick with a single active flow while
// the per-program map holds n IDs that have been seen and are now idle.
func BenchmarkPick_PrewarmedPrograms(b *testing.B) {
	for _, n := range []int{0, 1_000, 10_000, 100_000, 200_000} {
		b.Run(fmt.Sprintf("programs=%d", n), func(b *testing.B) {
			benchmarkPickWithPrewarmedPrograms(b, n)
		})
	}
}

func benchmarkPickWithPrewarmedPrograms(b *testing.B, programs int) {
	p, band := newPrewarmedPlugin(b, programs)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := p.Pick(context.Background(), band); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPick_PrewarmedPrograms_P50 reports the median of batched per-Pick
// means. b.Elapsed keeps the measurement on the runtime's benchmark clock,
// which is finer than time.Now on some platforms.
func BenchmarkPick_PrewarmedPrograms_P50(b *testing.B) {
	const batch = 16
	for _, n := range []int{0, 200_000} {
		b.Run(fmt.Sprintf("programs=%d", n), func(b *testing.B) {
			p, band := newPrewarmedPlugin(b, n)
			durations := make([]time.Duration, 0, (b.N+batch-1)/batch)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i += batch {
				count := min(batch, b.N-i)
				start := b.Elapsed()
				for j := 0; j < count; j++ {
					if _, err := p.Pick(context.Background(), band); err != nil {
						b.Fatal(err)
					}
				}
				durations = append(durations, (b.Elapsed()-start)/time.Duration(count))
			}
			b.StopTimer()
			slices.Sort(durations)
			b.ReportMetric(float64(durations[len(durations)/2]), "p50-ns")
		})
	}
}

// newPrewarmedPlugin returns a plugin whose per-program map holds the given
// number of idle programs, plus a band with a single active flow.
func newPrewarmedPlugin(b *testing.B, programs int) (*ProgramAwarePlugin, *fwkfcmocks.MockPriorityBandAccessor) {
	b.Helper()
	strategy, err := newStrategy(DefaultConfig())
	if err != nil {
		b.Fatal(err)
	}
	p := &ProgramAwarePlugin{strategy: strategy}
	enqueue := time.Now().Add(-time.Millisecond)
	for i := 0; i < programs; i++ {
		m := p.getOrCreateMetrics(fmt.Sprintf("program-%d", i))
		m.RecordDispatched(enqueue)
		m.RecordCompletion(enqueue)
	}

	req := &fwksched.InferenceRequest{FairnessID: "active"}
	item := &fwkfcmocks.MockQueueItemAccessor{
		EnqueueTimeV:     time.Now(),
		OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{InferenceRequestV: req},
	}
	queue := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: "active"},
		PeekV:    item,
	}
	band := &fwkfcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) { cb(queue) },
	}
	return p, band
}
