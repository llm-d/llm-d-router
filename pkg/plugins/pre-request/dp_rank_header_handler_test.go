package prerequest

import (
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common"
)

func makeDPEndpoint(addr, port string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "dp-pod"},
			Address:        addr,
			Port:           port,
		},
		&fwkdl.Metrics{},
		nil,
	)
}

func makeSchedulingResult(profileName string, endpoints ...scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: profileName,
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			profileName: {
				TargetEndpoints: endpoints,
			},
		},
	}
}

func TestDPRankHeaderHandler_PreRequest(t *testing.T) {
	tests := []struct {
		name           string
		headers        map[string]string
		result         *scheduling.SchedulingResult
		expectRankSet  bool
		expectedRank   string
		expectInternal bool // whether internal header should remain
	}{
		{
			name: "rank 0 wins for selected pod",
			headers: map[string]string{
				common.DPWinningRanksHeader: `{"10.0.0.1:8000":0}`,
			},
			result:         makeSchedulingResult("default", makeDPEndpoint("10.0.0.1", "8000")),
			expectRankSet:  true,
			expectedRank:   "0",
			expectInternal: false,
		},
		{
			name: "rank 3 wins for selected pod",
			headers: map[string]string{
				common.DPWinningRanksHeader: `{"10.0.0.1:8000":3}`,
			},
			result:         makeSchedulingResult("default", makeDPEndpoint("10.0.0.1", "8000")),
			expectRankSet:  true,
			expectedRank:   "3",
			expectInternal: false,
		},
		{
			name: "multiple pods - correct rank for selected pod",
			headers: map[string]string{
				common.DPWinningRanksHeader: `{"10.0.0.1:8000":0,"10.0.0.2:8000":5}`,
			},
			result:         makeSchedulingResult("default", makeDPEndpoint("10.0.0.2", "8000")),
			expectRankSet:  true,
			expectedRank:   "5",
			expectInternal: false,
		},
		{
			name: "selected pod not in winning ranks (non-DP pod)",
			headers: map[string]string{
				common.DPWinningRanksHeader: `{"10.0.0.1:8000":0}`,
			},
			result:         makeSchedulingResult("default", makeDPEndpoint("10.0.0.99", "8000")),
			expectRankSet:  false,
			expectInternal: false,
		},
		{
			name:           "no internal header (External LB / non-DP)",
			headers:        map[string]string{},
			result:         makeSchedulingResult("default", makeDPEndpoint("10.0.0.1", "8000")),
			expectRankSet:  false,
			expectInternal: false,
		},
		{
			name: "empty internal header",
			headers: map[string]string{
				common.DPWinningRanksHeader: "",
			},
			result:         makeSchedulingResult("default", makeDPEndpoint("10.0.0.1", "8000")),
			expectRankSet:  false,
			expectInternal: false,
		},
		{
			name: "malformed JSON in internal header",
			headers: map[string]string{
				common.DPWinningRanksHeader: "not-json",
			},
			result:         makeSchedulingResult("default", makeDPEndpoint("10.0.0.1", "8000")),
			expectRankSet:  false,
			expectInternal: false,
		},
		{
			name: "nil scheduling result",
			headers: map[string]string{
				common.DPWinningRanksHeader: `{"10.0.0.1:8000":0}`,
			},
			result:         nil,
			expectRankSet:  false,
			expectInternal: false,
		},
		{
			name: "empty profile results",
			headers: map[string]string{
				common.DPWinningRanksHeader: `{"10.0.0.1:8000":0}`,
			},
			result: &scheduling.SchedulingResult{
				PrimaryProfileName: "default",
				ProfileResults:     map[string]*scheduling.ProfileRunResult{},
			},
			expectRankSet:  false,
			expectInternal: false,
		},
		{
			name: "internal header always removed even on success",
			headers: map[string]string{
				common.DPWinningRanksHeader: `{"10.0.0.1:8000":2}`,
			},
			result:         makeSchedulingResult("default", makeDPEndpoint("10.0.0.1", "8000")),
			expectRankSet:  true,
			expectedRank:   "2",
			expectInternal: false,
		},
	}

	handler := NewDPRankHeaderHandler().WithName("test-dp-rank")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := &scheduling.LLMRequest{
				Headers: tt.headers,
			}

			handler.PreRequest(t.Context(), request, tt.result)

			// Check X-data-parallel-rank header
			if tt.expectRankSet {
				assert.Equal(t, tt.expectedRank, request.Headers[common.DataParallelRankHeader],
					"expected X-data-parallel-rank to be set")
			} else {
				_, exists := request.Headers[common.DataParallelRankHeader]
				assert.False(t, exists, "expected no X-data-parallel-rank header")
			}

			// Internal header must always be removed
			_, internalExists := request.Headers[common.DPWinningRanksHeader]
			assert.Equal(t, tt.expectInternal, internalExists,
				"internal header existence mismatch")
		})
	}
}

func TestDPRankHeaderHandlerFactory_Success(t *testing.T) {
	p, err := DPRankHeaderHandlerFactory("my-dp-rank", nil, nil)
	assert.NoError(t, err)
	assert.Equal(t, "my-dp-rank", p.TypedName().Name)
	assert.Equal(t, DPRankHeaderHandlerType, p.TypedName().Type)
}
