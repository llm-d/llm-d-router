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
	"math"
	"testing"

	"github.com/google/uuid"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/tokenload"
)

type workload struct {
	name     string
	requests []int64
}

var comparisonWorkloads = []workload{
	{
		name:     "uniform",
		requests: repeatTokens(4096, 200),
	},
	{
		name:     "heterogeneous",
		requests: mixTokens(128, 140, 32768, 60),
	},
	{
		name:     "extreme",
		requests: mixTokens(64, 100, 65536, 100),
	},
}

// BenchmarkScorerComparison simulates scheduling a heterogeneous stream of
// requests and measures how evenly each scoring strategy distributes the
// resulting token load across endpoints. The key metric is load_cv (coefficient
// of variation of total tokens per endpoint): lower values indicate more
// balanced placement.
//
// Run:
//
//	go test -run='^$' -bench=BenchmarkScorerComparison -benchmem -count=3 \
//	    ./pkg/epp/scheduling/
func BenchmarkScorerComparison(b *testing.B) {
	for _, wl := range comparisonWorkloads {
		b.Run("scorer=queue/workload="+wl.name, func(b *testing.B) {
			benchQueue(b, wl.requests)
		})
		b.Run("scorer=token-load/workload="+wl.name, func(b *testing.B) {
			benchTokenLoad(b, wl.requests)
		})
	}
}

const comparisonEndpointCount = 10

func benchQueue(b *testing.B, requests []int64) {
	b.Helper()
	ctx := context.Background()

	scorer := queuedepth.NewQueueScorer()
	scheduler := newComparisonScheduler(NewWeightedScorer(scorer, 1))

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		endpoints := makeIdleEndpoints(comparisonEndpointCount)
		tokenTotals := make([]int64, comparisonEndpointCount)

		for _, tokens := range requests {
			result, err := scheduler.Schedule(ctx, newComparisonRequest(), endpoints)
			if err != nil {
				b.Fatalf("schedule failed: %v", err)
			}
			chosen := result.ProfileResults["default"].TargetEndpoints[0]
			idx := endpointIndex(chosen)
			tokenTotals[idx] += tokens
			endpoints[idx].GetMetrics().WaitingQueueSize++
		}

		b.ReportMetric(coefficientOfVariation(tokenTotals), "load_cv")
	}
}

func benchTokenLoad(b *testing.B, requests []int64) {
	b.Helper()
	ctx := context.Background()

	plugin, err := tokenload.TokenLoadScorerFactory(tokenload.TokenLoadScorerType, nil, nil)
	if err != nil {
		b.Fatalf("token-load scorer setup: %v", err)
	}
	scorer := plugin.(fwksched.Scorer)
	scheduler := newComparisonScheduler(NewWeightedScorer(scorer, 1))

	inFlightKey := concurrency.InFlightLoadDataKey.WithNonEmptyProducerName("").String()
	uncachedKey := concurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName("").String()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		endpoints := makeIdleEndpoints(comparisonEndpointCount)
		tokenTotals := make([]int64, comparisonEndpointCount)

		for _, tokens := range requests {
			for j := range endpoints {
				endpoints[j].Put(uncachedKey, &concurrency.UncachedRequestTokens{Tokens: tokens})
			}

			result, err := scheduler.Schedule(ctx, newComparisonRequest(), endpoints)
			if err != nil {
				b.Fatalf("schedule failed: %v", err)
			}
			chosen := result.ProfileResults["default"].TargetEndpoints[0]
			idx := endpointIndex(chosen)
			tokenTotals[idx] += tokens

			prev := getInFlightTokens(endpoints[idx], inFlightKey)
			endpoints[idx].Put(inFlightKey, &concurrency.InFlightLoad{Tokens: prev + tokens})
		}

		b.ReportMetric(coefficientOfVariation(tokenTotals), "load_cv")
	}
}

func newComparisonScheduler(scorer *WeightedScorer) *Scheduler {
	profile := NewSchedulerProfile().
		WithScorers(scorer).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	cfg := NewSchedulerConfig(
		single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": profile},
	)
	return NewSchedulerWithConfig(cfg)
}

func makeIdleEndpoints(n int) []fwksched.Endpoint {
	endpoints := make([]fwksched.Endpoint, n)
	for i := range n {
		endpoints[i] = fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: fmt.Sprintf("pod%d", i)},
			},
			&fwkdl.Metrics{},
			fwkdl.NewAttributes(),
		)
	}
	return endpoints
}

func newComparisonRequest() *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "model",
	}
}

func endpointIndex(ep fwksched.Endpoint) int {
	name := ep.GetMetadata().NamespacedName.Name
	var idx int
	_, _ = fmt.Sscanf(name, "pod%d", &idx)
	return idx
}

func getInFlightTokens(ep fwksched.Endpoint, key string) int64 {
	if val, ok := ep.Get(key); ok {
		if load, ok := val.(*concurrency.InFlightLoad); ok && load != nil {
			return load.Tokens
		}
	}
	return 0
}

func coefficientOfVariation(values []int64) float64 {
	n := float64(len(values))
	if n == 0 {
		return 0
	}
	var sum float64
	for _, v := range values {
		sum += float64(v)
	}
	mean := sum / n
	if mean == 0 {
		return 0
	}
	var sqDiff float64
	for _, v := range values {
		d := float64(v) - mean
		sqDiff += d * d
	}
	stddev := math.Sqrt(sqDiff / n)
	return stddev / mean
}

func repeatTokens(size int64, count int) []int64 {
	r := make([]int64, count)
	for i := range r {
		r[i] = size
	}
	return r
}

func mixTokens(shortSize int64, shortCount int, longSize int64, longCount int) []int64 {
	total := shortCount + longCount
	r := make([]int64, total)
	longInterval := total / longCount
	placed := 0
	for i := range total {
		if placed < longCount && i%longInterval == 0 {
			r[i] = longSize
			placed++
		} else {
			r[i] = shortSize
		}
	}
	return r
}
