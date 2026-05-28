package sessionaffinity

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
)

func createEndpoint(nsn k8stypes.NamespacedName) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: nsn,
		},
		&fwkdl.Metrics{},
		nil,
	)
}

func TestFactory(t *testing.T) {
	tests := []struct {
		name         string
		pluginName   string
		jsonParams   string
		expectErr    bool
		expectHeader string
	}{
		{
			name:         "valid configuration with custom header",
			pluginName:   "test-plugin",
			jsonParams:   `{"sessionTokenHeader": "custom-header"}`,
			expectErr:    false,
			expectHeader: "custom-header",
		},
		{
			name:         "empty configuration defaults to default header",
			pluginName:   "test-plugin",
			jsonParams:   `{}`,
			expectErr:    false,
			expectHeader: defaultSessionTokenHeader,
		},
		{
			name:       "invalid JSON should error",
			pluginName: "test-plugin",
			jsonParams: `{"sessionTokenHeader": }`,
			expectErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rawParams json.RawMessage
			if tt.jsonParams != "" {
				rawParams = json.RawMessage(tt.jsonParams)
			}
			p, err := Factory(tt.pluginName, rawParams, nil)

			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, p)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, p)
				pluginInstance, ok := p.(*Plugin)
				require.True(t, ok)
				assert.Equal(t, tt.expectHeader, pluginInstance.sessionTokenHeader)
			}
		})
	}
}

func TestSessionAffinityPlugin(t *testing.T) {
	ep1 := createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "pod-1"})
	ep2 := createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "pod-2"})
	endpoints := []scheduling.Endpoint{ep1, ep2}

	ctx := context.Background()

	t.Run("PrepareRequestData - no session ID", func(t *testing.T) {
		p := NewPlugin("test", defaultSessionTokenHeader)
		request := &scheduling.InferenceRequest{
			RequestID: "req-1",
			Headers:   make(map[string]string),
		}
		err := p.PrepareRequestData(ctx, request, endpoints)
		assert.NoError(t, err)

		// Verify all pods marked as targets
		for _, pod := range endpoints {
			val, ok := pod.Get(SessionAffinityInfoKey)
			require.True(t, ok)
			info, ok := val.(*SessionAffinityInfo)
			require.True(t, ok)
			assert.True(t, info.IsTarget)
		}
	})

	t.Run("PrepareRequestData - session ID but no state", func(t *testing.T) {
		p := NewPlugin("test", defaultSessionTokenHeader)
		request := &scheduling.InferenceRequest{
			RequestID: "req-1",
			Headers:   map[string]string{defaultSessionTokenHeader: "session-1"},
		}
		err := p.PrepareRequestData(ctx, request, endpoints)
		assert.NoError(t, err)

		// Verify all pods marked as targets
		for _, pod := range endpoints {
			val, ok := pod.Get(SessionAffinityInfoKey)
			require.True(t, ok)
			info, ok := val.(*SessionAffinityInfo)
			require.True(t, ok)
			assert.True(t, info.IsTarget)
		}
	})

	t.Run("Full Flow - Establish affinity, Prepare, and Filter", func(t *testing.T) {
		p := NewPlugin("test", defaultSessionTokenHeader)
		request := &scheduling.InferenceRequest{
			RequestID: "req-1",
			Headers:   map[string]string{defaultSessionTokenHeader: "session-1"},
		}

		// 1. Simulate PreRequest to establish affinity
		schedulingResult := &scheduling.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults: map[string]*scheduling.ProfileRunResult{
				"default": {
					TargetEndpoints: []scheduling.Endpoint{ep1},
				},
			},
		}
		p.PreRequest(ctx, request, schedulingResult)

		// 2. PrepareRequestData for next request
		request2 := &scheduling.InferenceRequest{
			RequestID: "req-2",
			Headers:   map[string]string{defaultSessionTokenHeader: "session-1"},
		}
		err := p.PrepareRequestData(ctx, request2, endpoints)
		assert.NoError(t, err)

		// Verify attributes
		val1, ok := ep1.Get(SessionAffinityInfoKey)
		require.True(t, ok)
		assert.True(t, val1.(*SessionAffinityInfo).IsTarget)

		val2, ok := ep2.Get(SessionAffinityInfoKey)
		require.True(t, ok)
		assert.False(t, val2.(*SessionAffinityInfo).IsTarget)

		// 3. Filter
		filtered := p.Filter(ctx, nil, request2, endpoints)
		require.Len(t, filtered, 1)
		assert.Equal(t, "pod-1", filtered[0].GetMetadata().NamespacedName.Name)
	})

	t.Run("Fallback - mapped endpoint missing", func(t *testing.T) {
		p := NewPlugin("test", defaultSessionTokenHeader)
		request := &scheduling.InferenceRequest{
			RequestID: "req-1",
			Headers:   map[string]string{defaultSessionTokenHeader: "session-1"},
		}

		// Establish affinity to pod-1
		schedulingResult := &scheduling.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults: map[string]*scheduling.ProfileRunResult{
				"default": {
					TargetEndpoints: []scheduling.Endpoint{ep1},
				},
			},
		}
		p.PreRequest(ctx, request, schedulingResult)

		// Candidate list only contains pod-2
		candidates := []scheduling.Endpoint{ep2}

		request2 := &scheduling.InferenceRequest{
			RequestID: "req-2",
			Headers:   map[string]string{defaultSessionTokenHeader: "session-1"},
		}
		err := p.PrepareRequestData(ctx, request2, candidates)
		assert.NoError(t, err)

		// Should evict and mark all candidates as targets
		val, ok := ep2.Get(SessionAffinityInfoKey)
		require.True(t, ok)
		assert.True(t, val.(*SessionAffinityInfo).IsTarget)

		// Verify state was evicted
		p.mu.RLock()
		_, exists := p.state["session-1"]
		p.mu.RUnlock()
		assert.False(t, exists)

		// Filter should return all candidates
		filtered := p.Filter(ctx, nil, request2, candidates)
		assert.Equal(t, candidates, filtered)
	})
}
