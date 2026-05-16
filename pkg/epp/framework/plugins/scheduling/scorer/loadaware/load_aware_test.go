package loadaware_test

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	k8stypes "k8s.io/apimachinery/pkg/types" // Import config for thresholds

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	loadaware "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loadaware"
	"github.com/llm-d/llm-d-router/test/utils"
)

func TestLoadBasedScorer(t *testing.T) {
	endpointA := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{
			WaitingQueueSize: 2,
		},
		nil,
	)
	endpointB := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-b"}},
		&fwkdl.Metrics{
			WaitingQueueSize: 0,
		},
		nil,
	)
	endpointC := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-c"}},
		&fwkdl.Metrics{
			WaitingQueueSize: 15,
		},
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
			got := test.scorer.Score(context.Background(), nil, nil, test.input)

			if diff := cmp.Diff(test.wantScores, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
