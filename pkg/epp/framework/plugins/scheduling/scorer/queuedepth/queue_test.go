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

package queuedepth

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func scraped(m fwkdl.Metrics) *fwkdl.Metrics {
	m.UpdateTime = time.Now()
	return &m
}

func TestQueueScorer(t *testing.T) {
	tests := []struct {
		name                   string
		endpoints              []fwksched.Endpoint
		expectedScoresEndpoint map[int]float64 // Map of endpoint index to expected score
	}{
		{
			name: "Different queue sizes",
			endpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 10}), nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 5}), nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 0}), nil),
			},
			expectedScoresEndpoint: map[int]float64{
				0: 0.0, // Longest queue (10) gets lowest score
				1: 0.5, // Medium queue (5) gets medium score
				2: 1.0, // Shortest queue (0) gets highest score
			},
		},
		{
			name: "Same queue sizes",
			endpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 5}), nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 5}), nil),
			},
			expectedScoresEndpoint: map[int]float64{
				0: 1.0, // When all pods have the same queue size, they get the same neutral score
				1: 1.0,
			},
		},
		{
			name: "Zero queue sizes",
			endpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 0}), nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 0}), nil),
			},
			expectedScoresEndpoint: map[int]float64{
				0: 1.0,
				1: 1.0,
			},
		},
	}

	scorer := &QueueScorer{}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scores := scorer.Score(context.Background(), &fwksched.InferenceRequest{}, test.endpoints)

			for i, endpoint := range test.endpoints {
				expectedScore := test.expectedScoresEndpoint[i]
				assert.InDelta(t, expectedScore, scores[endpoint], 0.0001, "Pod %d should have score %f", i, expectedScore)
			}
		})
	}
}

func TestQueueScorerOmitsNeverScraped(t *testing.T) {
	score := func(v float64) *float64 { return &v }
	tests := []struct {
		name      string
		endpoints []fwksched.Endpoint
		want      []*float64
	}{
		{
			name: "never-scraped endpoint is unscored and ignored in normalization",
			endpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 1}), nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 9}), nil),
			},
			want: []*float64{score(1.0), nil, score(0.0)},
		},
		{
			name: "equal observed queues stay neutral when a never-scraped endpoint is present",
			endpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 5}), nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 5}), nil),
			},
			want: []*float64{score(1.0), nil, score(1.0)},
		},
		{
			name: "nil metrics are unscored",
			endpoints: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, scraped(fwkdl.Metrics{WaitingQueueSize: 0}), nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, nil, nil),
			},
			want: []*float64{score(1.0), nil},
		},
	}

	scorer := &QueueScorer{}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scores := scorer.Score(context.Background(), &fwksched.InferenceRequest{}, test.endpoints)
			for i, want := range test.want {
				got, ok := scores[test.endpoints[i]]
				if want == nil {
					assert.False(t, ok, "endpoint %d should be unscored", i)
					continue
				}
				assert.True(t, ok, "endpoint %d should be scored", i)
				assert.InDelta(t, *want, got, 1e-9)
			}
		})
	}
}
