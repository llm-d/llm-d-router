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

package scheduling

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraaffinity"
	schedprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
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
	prefixCacheScorer, err := schedprefix.New(ctx, schedprefix.PrefixCacheScorerPluginType, "")
	if err != nil {
		b.Fatalf("prefix scorer setup: %v", err)
	}
	loraAffinityScorer := loraaffinity.NewLoraAffinityScorer()
	datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{
		kvCacheUtilizationScorer, queueingScorer, prefixCacheScorer, loraAffinityScorer,
	})

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
		Body: &fwkrequest.InferenceRequestBody{
			Completions: &fwkrequest.CompletionsRequest{
				Prompt: fwkrequest.Prompt{Raw: prompt},
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
	scrapedAt := time.Now()
	endpoints := make([]fwksched.Endpoint, n)
	for i := 0; i < n; i++ {
		active := map[string]int{"baseline": 1}
		// Spread the target model across ~1/3 of the fleet.
		if i%3 == 0 {
			active["critical"] = 1
		}
		endpoints[i] = fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{
				ID: k8stypes.NamespacedName{Name: fmt.Sprintf("pod%d", i)},
			},
			&fwkdl.Metrics{
				WaitingQueueSize:    i % 8,              // 0..7
				KVCacheUsagePercent: float64(i%10) / 10, // 0.0..0.9
				MaxActiveModels:     2,
				ActiveModels:        active,
				UpdateTime:          scrapedAt,
			},
			fwkdl.NewAttributes(),
		)
	}
	return endpoints
}

func BenchmarkDisaggSchedule(b *testing.B) {
	for _, tc := range []struct {
		name         string
		prefill      bool
		fallback     bool
		decodeAbsent bool
	}{
		{name: "decode-only"},
		{name: "pd", prefill: true},
		{name: "pd-with-fallback", prefill: true, fallback: true},
		{name: "fallback", prefill: true, fallback: true, decodeAbsent: true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			for _, n := range []int{5, 25, 100, 1000} {
				b.Run(fmt.Sprintf("endpoints=%d", n), func(b *testing.B) {
					base := disagg.NewDisaggProfileHandler("decode", "prefill", "", nil, nil)
					if tc.prefill {
						base = disagg.NewDisaggProfileHandler("decode", "prefill", "", &disagg.AlwaysDisaggPDDecider{}, nil)
					}
					var handler fwksched.ProfileHandler = base
					if tc.fallback {
						handler = base.WithFallbackProfile("fallback")
					}
					profiles := map[string]fwksched.SchedulerProfile{}
					for name, role := range map[string]string{"decode": bylabel.RoleDecode, "prefill": bylabel.RolePrefill, "fallback": bylabel.RolePrefillDecode} {
						selector, err := bylabel.NewSelector(name, &metav1.LabelSelector{MatchLabels: map[string]string{bylabel.RoleLabel: role}})
						if err != nil {
							b.Fatal(err)
						}
						profiles[name] = NewSchedulerProfile().WithFilters(selector).WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
					}
					endpoints := makeBenchmarkEndpoints(n)
					roles := []string{bylabel.RoleDecode, bylabel.RolePrefill, bylabel.RolePrefillDecode}
					for i, endpoint := range endpoints {
						role := roles[i%len(roles)]
						if tc.decodeAbsent && role == bylabel.RoleDecode {
							role = bylabel.RolePrefill
						}
						endpoint.GetMetadata().Labels = map[string]string{bylabel.RoleLabel: role}
					}
					scheduler := NewSchedulerWithConfig(NewSchedulerConfig(handler, profiles))
					request := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "critical", Headers: map[string]string{}}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						result, err := scheduler.Schedule(context.Background(), request, endpoints)
						if err != nil {
							b.Fatal(err)
						}
						want := "decode"
						if tc.decodeAbsent {
							want = "fallback"
						}
						if result.PrimaryProfileName != want {
							b.Fatalf("primary = %s, want %s", result.PrimaryProfileName, want)
						}
					}
				})
			}
		})
	}
}
