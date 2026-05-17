/*
Copyright 2026 The Kubernetes Authors.

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

package scheduling

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/google/uuid"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/picker"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/loraaffinity"
	schedprefix "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
)

// BenchmarkSchedule measures the per-request cost of the scheduler hot path
// (Scheduler.Schedule → ProfileHandler.Pick → SchedulerProfile.Run →
// filter/score/pick) at a few representative fleet sizes. Use this as the
// baseline before optimizing per-request allocations or CPU in the scheduler
// framework code.
//
// Plugin stack mirrors the production-ish defaults used in TestSchedule:
// queue-depth + KV-cache-utilization + prefix-cache + LoRA-affinity scorers
// behind a max-score picker, all under a single profile handler. Plugins
// themselves are not the optimization target here, but they're real so the
// benchmark reflects realistic call-graph shapes and allocator pressure.
//
// Run:
//
//	go test -run='^$' -bench=BenchmarkSchedule -benchmem -count=10 \
//	    ./pkg/epp/scheduling/ | tee bench.out
//	benchstat bench.out
func BenchmarkSchedule(b *testing.B) {
	ctx := context.Background()

	kvCacheUtilizationScorer := kvcacheutilization.NewKVCacheUtilizationScorer()
	queueingScorer := queuedepth.NewQueueScorer()
	prefixCacheScorer, err := schedprefix.New(ctx)
	if err != nil {
		b.Fatalf("prefix scorer setup: %v", err)
	}
	loraAffinityScorer := loraaffinity.NewLoraAffinityScorer()

	profile := NewSchedulerProfile().
		WithScorers(
			NewWeightedScorer(kvCacheUtilizationScorer, 1),
			NewWeightedScorer(queueingScorer, 1),
			NewWeightedScorer(prefixCacheScorer, 1),
			NewWeightedScorer(loraAffinityScorer, 1),
		).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	cfg := NewSchedulerConfig(
		single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": profile},
	)
	scheduler := NewSchedulerWithConfig(cfg)

	// A representative request: non-empty prompt so the prefix scorer does real
	// work; a target model that matches one of the endpoint's ActiveModels so
	// lora-affinity has a non-zero signal.
	prompt := strings.Repeat("the quick brown fox jumps over the lazy dog. ", 8)
	req := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "critical",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: prompt},
			},
		},
	}

	// Fleet sizes chosen to cover small-team (5), medium-cluster (25), and
	// large-fleet (100) regimes; each tests different parts of the per-request
	// cost curve (constant scheduler overhead vs O(N) work in scoring/picking).
	for _, n := range []int{5, 25, 100} {
		endpoints := makeBenchmarkEndpoints(n)
		b.Run(fmt.Sprintf("endpoints=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result, err := scheduler.Schedule(ctx, req, endpoints)
				if err != nil {
					b.Fatalf("schedule failed: %v", err)
				}
				if result == nil {
					b.Fatal("nil result")
				}
			}
		})
	}
}

// makeBenchmarkEndpoints builds n endpoints with varied but realistic metrics
// so scorers produce differentiated outputs (matters for the picker's work).
// The mix below loosely models a heterogeneous fleet: queue depths span a
// range, KV-cache utilization is distributed, and ~1/3 of pods have the
// benchmark's target model loaded so lora-affinity has something to find.
func makeBenchmarkEndpoints(n int) []fwksched.Endpoint {
	endpoints := make([]fwksched.Endpoint, n)
	for i := 0; i < n; i++ {
		active := map[string]int{"baseline": 1}
		// Spread the target model across ~1/3 of the fleet.
		if i%3 == 0 {
			active["critical"] = 1
		}
		endpoints[i] = fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: fmt.Sprintf("pod%d", i)},
			},
			&fwkdl.Metrics{
				WaitingQueueSize:    i % 8,              // 0..7
				KVCacheUsagePercent: float64(i%10) / 10, // 0.0..0.9
				MaxActiveModels:     2,
				ActiveModels:        active,
			},
			fwkdl.NewAttributes(),
		)
	}
	return endpoints
}
