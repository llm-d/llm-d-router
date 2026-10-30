/*
Copyright 2025 The llm-d Authors.

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

package loadaware_test

import (
	"context"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	k8stypes "k8s.io/apimachinery/pkg/types" // Import config for thresholds

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loadaware"
	"github.com/llm-d/llm-d-router/test/utils"
)

func scraped(m fwkdl.Metrics) *fwkdl.Metrics {
	m.UpdateTime = time.Now()
	return &m
}

func TestLoadBasedScorer(t *testing.T) {
	endpointA := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod-a"}},
		scraped(fwkdl.Metrics{
			WaitingQueueSize: 2,
		}),
		nil,
	)
	endpointB := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod-b"}},
		scraped(fwkdl.Metrics{
			WaitingQueueSize: 0,
		}),
		nil,
	)
	endpointC := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod-c"}},
		scraped(fwkdl.Metrics{
			WaitingQueueSize: 15,
		}),
		nil,
	)

	tests := []struct {
		name       string
		scorer     scheduling.Scorer
		req        *scheduling.InferenceRequest
		input      []scheduling.Endpoint
		wantScores map[scheduling.Endpoint]float64
	}{
		{
			name:   "load based scorer",
			scorer: loadaware.NewLoadAware(utils.NewTestContext(t), 10),
			req: &scheduling.InferenceRequest{
				TargetModel: "critical",
			},
			input: []scheduling.Endpoint{
				endpointA, endpointB, endpointC,
			},
			wantScores: map[scheduling.Endpoint]float64{
				endpointA: 0.4,
				endpointB: 0.5,
				endpointC: 0,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := test.scorer.Score(context.Background(), nil, test.input)

			if diff := cmp.Diff(test.wantScores, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func TestLoadBasedScorerOmitsNeverScraped(t *testing.T) {
	busy := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod-busy"}},
		scraped(fwkdl.Metrics{WaitingQueueSize: 2}),
		nil,
	)
	idle := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod-idle"}},
		scraped(fwkdl.Metrics{WaitingQueueSize: 0}),
		nil,
	)
	neverScraped := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod-never-scraped"}},
		&fwkdl.Metrics{},
		nil,
	)

	scorer := loadaware.NewLoadAware(utils.NewTestContext(t), 10)
	got := scorer.Score(context.Background(), nil, []scheduling.Endpoint{busy, idle, neverScraped})

	want := map[scheduling.Endpoint]float64{
		busy: 0.4,
		idle: 0.5,
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Unexpected output (-want +got): %v", diff)
	}
}
