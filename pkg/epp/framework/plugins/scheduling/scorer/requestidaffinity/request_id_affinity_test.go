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

package requestidaffinity_test

import (
	"context"
	"strconv"
	"testing"

	"github.com/google/go-cmp/cmp"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/requestidaffinity"
)

func makeEndpoint(name string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{},
		nil,
	)
}

// endpoints named pod-0 .. pod-7, sorted alphabetically they map to indices 0-7.
func makeEndpoints(names ...string) []scheduling.Endpoint {
	eps := make([]scheduling.Endpoint, len(names))
	for i, n := range names {
		eps[i] = makeEndpoint(n)
	}
	return eps
}

func TestScore_NumericSuffixRouting(t *testing.T) {
	// 8 pods; sorted alphabetically: pod-0, pod-1, ..., pod-7
	eps := makeEndpoints("pod-3", "pod-0", "pod-5", "pod-1", "pod-7", "pod-2", "pod-6", "pod-4")

	// after sort: pod-0=idx0, pod-1=idx1, ..., pod-7=idx7
	byName := map[string]scheduling.Endpoint{}
	for _, ep := range eps {
		byName[ep.GetMetadata().NamespacedName.Name] = ep
	}

	scorer := requestidaffinity.New("test", "", 8)

	tests := []struct {
		requestID    string
		wantPodName  string
	}{
		{"req-0", "pod-0"},
		{"req-1", "pod-1"},
		{"req-8", "pod-0"},  // 8 % 8 = 0
		{"req-9", "pod-1"},  // 9 % 8 = 1
		{"req-16", "pod-0"}, // 16 % 8 = 0
		{"req-17", "pod-1"}, // 17 % 8 = 1
		{"req-7", "pod-7"},
		{"req-15", "pod-7"}, // 15 % 8 = 7
	}

	for _, tt := range tests {
		t.Run(tt.requestID, func(t *testing.T) {
			req := &scheduling.InferenceRequest{
				Headers: map[string]string{"x-request-id": tt.requestID},
			}
			scores := scorer.Score(context.Background(), req, eps)

			want := make(map[scheduling.Endpoint]float64, len(eps))
			for _, ep := range eps {
				want[ep] = 0.0
			}
			want[byName[tt.wantPodName]] = 1.0

			if diff := cmp.Diff(want, scores); diff != "" {
				t.Errorf("Score(%q) mismatch (-want +got):\n%s", tt.requestID, diff)
			}
		})
	}
}

func TestScore_DynamicModulus(t *testing.T) {
	// modulus=0: use len(endpoints)
	eps := makeEndpoints("pod-0", "pod-1", "pod-2")
	scorer := requestidaffinity.New("test", "", 0)

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{"x-request-id": "abc-5"}, // 5 % 3 = 2 -> pod-2
	}
	scores := scorer.Score(context.Background(), req, eps)
	if scores[makeEndpoint("pod-2")] != 1.0 {
		// Re-fetch from the actual slice to avoid map key mismatch.
		byName := map[string]scheduling.Endpoint{}
		for _, ep := range eps {
			byName[ep.GetMetadata().NamespacedName.Name] = ep
		}
		if scores[byName["pod-2"]] != 1.0 {
			t.Errorf("expected pod-2 to score 1.0, got %v", scores[byName["pod-2"]])
		}
	}
}

func TestScore_NoHeader(t *testing.T) {
	eps := makeEndpoints("pod-0", "pod-1")
	scorer := requestidaffinity.New("test", "", 2)

	req := &scheduling.InferenceRequest{Headers: map[string]string{}}
	scores := scorer.Score(context.Background(), req, eps)
	for _, ep := range eps {
		if scores[ep] != 0.0 {
			t.Errorf("expected 0.0 for %s without header, got %f", ep.GetMetadata().NamespacedName.Name, scores[ep])
		}
	}
}

func TestScore_NonNumericHeader(t *testing.T) {
	eps := makeEndpoints("pod-0", "pod-1")
	scorer := requestidaffinity.New("test", "", 2)

	req := &scheduling.InferenceRequest{Headers: map[string]string{"x-request-id": "no-numbers-here"}}
	scores := scorer.Score(context.Background(), req, eps)
	for _, ep := range eps {
		if scores[ep] != 0.0 {
			t.Errorf("expected 0.0 for non-numeric suffix, got %f", scores[ep])
		}
	}
}

func TestScore_EmptyEndpoints(t *testing.T) {
	scorer := requestidaffinity.New("test", "", 8)
	req := &scheduling.InferenceRequest{Headers: map[string]string{"x-request-id": "req-3"}}
	scores := scorer.Score(context.Background(), req, nil)
	if len(scores) != 0 {
		t.Errorf("expected empty score map for empty endpoints, got %v", scores)
	}
}

func TestScore_CustomHeader(t *testing.T) {
	eps := makeEndpoints("pod-0", "pod-1")
	scorer := requestidaffinity.New("test", "x-my-id", 2)

	// x-my-id header present
	req := &scheduling.InferenceRequest{Headers: map[string]string{"x-my-id": "req-3"}} // 3 % 2 = 1 -> pod-1
	scores := scorer.Score(context.Background(), req, eps)
	byName := map[string]scheduling.Endpoint{}
	for _, ep := range eps {
		byName[ep.GetMetadata().NamespacedName.Name] = ep
	}
	if scores[byName["pod-1"]] != 1.0 {
		t.Errorf("expected pod-1 to score 1.0, got %v", scores[byName["pod-1"]])
	}
	if scores[byName["pod-0"]] != 0.0 {
		t.Errorf("expected pod-0 to score 0.0, got %v", scores[byName["pod-0"]])
	}

	// default header is ignored when custom header is configured
	req2 := &scheduling.InferenceRequest{Headers: map[string]string{"x-request-id": "req-3"}}
	scores2 := scorer.Score(context.Background(), req2, eps)
	for _, ep := range eps {
		if scores2[ep] != 0.0 {
			t.Errorf("default header must be ignored when custom header is set, got %v", scores2[ep])
		}
	}
}

func TestScore_BucketWrapAround(t *testing.T) {
	// modulus=8 but only 3 endpoints — bucket wraps via bucket % len(endpoints)
	eps := makeEndpoints("pod-0", "pod-1", "pod-2")
	scorer := requestidaffinity.New("test", "", 8)

	byName := map[string]scheduling.Endpoint{}
	for _, ep := range eps {
		byName[ep.GetMetadata().NamespacedName.Name] = ep
	}

	tests := []struct {
		suffix      int
		wantPodName string
	}{
		{0, "pod-0"}, // 0 % 8 = 0; 0 % 3 = 0
		{3, "pod-0"}, // 3 % 8 = 3; 3 % 3 = 0
		{1, "pod-1"}, // 1 % 8 = 1; 1 % 3 = 1
		{4, "pod-1"}, // 4 % 8 = 4; 4 % 3 = 1
		{2, "pod-2"}, // 2 % 8 = 2; 2 % 3 = 2
		{5, "pod-2"}, // 5 % 8 = 5; 5 % 3 = 2
	}

	for _, tt := range tests {
		t.Run("suffix-"+itoa(tt.suffix), func(t *testing.T) {
			req := &scheduling.InferenceRequest{
				Headers: map[string]string{"x-request-id": "req-" + itoa(tt.suffix)},
			}
			scores := scorer.Score(context.Background(), req, eps)
			if scores[byName[tt.wantPodName]] != 1.0 {
				t.Errorf("suffix %d: expected %s to score 1.0, got %v", tt.suffix, tt.wantPodName, scores)
			}
		})
	}
}

func itoa(n int) string {
	return strconv.Itoa(n)
}
