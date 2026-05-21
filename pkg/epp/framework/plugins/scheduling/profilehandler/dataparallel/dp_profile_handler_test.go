package dataparallel

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

const DefaultTestPodPort = "8000"

func createEndpoint(nsn k8stypes.NamespacedName, ipaddr, port string, labels map[string]string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: nsn,
			Address:        ipaddr,
			Port:           port,
			Labels:         labels,
		},
		nil,
		fwkdl.NewAttributes(),
	)
}

func newMockProfileRunResult(port string, endpointNames ...string) *scheduling.ProfileRunResult {
	endpoints := make([]scheduling.Endpoint, 0, len(endpointNames))
	for i, name := range endpointNames {
		ip := fmt.Sprintf("10.0.0.%d", i+1)
		endpoints = append(endpoints, createEndpoint(
			k8stypes.NamespacedName{Namespace: "default", Name: name},
			ip,
			port,
			map[string]string{},
		))
	}
	return &scheduling.ProfileRunResult{
		TargetEndpoints: endpoints,
	}
}

func newMockSchedulerProfile() scheduling.SchedulerProfile {
	return &mockSchedulerProfile{}
}

type mockSchedulerProfile struct{}

func (p *mockSchedulerProfile) Run(_ context.Context, _ *scheduling.InferenceRequest, _ *scheduling.CycleState, _ []scheduling.Endpoint) (*scheduling.ProfileRunResult, error) {
	return &scheduling.ProfileRunResult{}, nil
}

func TestProfileHandlerFactory(t *testing.T) {
	tests := []struct {
		name         string
		pluginName   string
		jsonParams   string
		expectErr    bool
		expectedPort string // expected primaryPort as string
	}{
		{
			name:         "use default primaryPort (8000)",
			pluginName:   "default-handler",
			jsonParams:   "{}",
			expectErr:    false,
			expectedPort: "8000",
		},
		{
			name:         "explicit primaryPort = 9000",
			pluginName:   "custom-port",
			jsonParams:   `{"primaryPort": 9000}`,
			expectErr:    false,
			expectedPort: "9000",
		},
		{
			name:         "primaryPort = 1 (minimum valid)",
			pluginName:   "min-port",
			jsonParams:   `{"primaryPort": 1}`,
			expectErr:    false,
			expectedPort: "1",
		},
		{
			name:         "primaryPort = 65535 (maximum valid)",
			pluginName:   "max-port",
			jsonParams:   `{"primaryPort": 65535}`,
			expectErr:    false,
			expectedPort: "65535",
		},
		{
			name:         "primaryPort = 0 is allowed (but results in '0' string)",
			pluginName:   "zero-port",
			jsonParams:   `{"primaryPort": 0}`,
			expectErr:    false,
			expectedPort: "0",
		},
		{
			name:       "primaryPort = -1 should error",
			pluginName: "negative-port",
			jsonParams: `{"primaryPort": -1}`,
			expectErr:  true,
		},
		{
			name:       "primaryPort = 65536 should error",
			pluginName: "port-too-high",
			jsonParams: `{"primaryPort": 65536}`,
			expectErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rawParams json.RawMessage
			if tt.jsonParams != "" {
				rawParams = json.RawMessage(tt.jsonParams)
			}
			handle := plugin.NewEppHandle(utils.NewTestContext(t), nil)
			plugin, err := ProfileHandlerFactory(tt.pluginName, rawParams, handle)

			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, plugin)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, plugin)
			}
		})
	}
}

func TestProfileHandlerFactoryInvalidJSON(t *testing.T) {
	invalidTests := []struct {
		name       string
		jsonParams string
	}{
		{
			name:       "malformed JSON",
			jsonParams: `{"primaryPort":`,
		},
		{
			name:       "primaryPort as string",
			jsonParams: `{"primaryPort": "8000"}`,
		},
		{
			name:       "primaryPort as boolean",
			jsonParams: `{"primaryPort": true}`,
		},
	}

	for _, tt := range invalidTests {
		t.Run(tt.name, func(t *testing.T) {

			rawParams := json.RawMessage(tt.jsonParams)
			handle := plugin.NewEppHandle(utils.NewTestContext(t), nil)
			plugin, err := ProfileHandlerFactory("test", rawParams, handle)

			assert.Error(t, err)
			assert.Nil(t, plugin)
		})
	}
}

func Test_ProfileHandler_Pick(t *testing.T) {
	tests := []struct {
		name              string
		profiles          map[string]scheduling.SchedulerProfile
		profileResults    map[string]*scheduling.ProfileRunResult
		expectEmptyResult bool
		expectLogError    bool
		description       string
	}{
		{
			name: "success: single profile, first call",
			profiles: map[string]scheduling.SchedulerProfile{
				"default": newMockSchedulerProfile(),
			},
			profileResults:    map[string]*scheduling.ProfileRunResult{},
			expectEmptyResult: false,
			expectLogError:    false,
			description:       "Should return the single profile to run",
		},
		{
			name: "success: single profile, second call (all already executed)",
			profiles: map[string]scheduling.SchedulerProfile{
				"default": newMockSchedulerProfile(),
			},
			profileResults: map[string]*scheduling.ProfileRunResult{
				"default": newMockProfileRunResult(DefaultTestPodPort, "pod1"),
			},
			expectEmptyResult: true,
			expectLogError:    false,
			description:       "Should return empty map since all profiles have been executed already in previous call",
		},
		{
			name: "error: multiple profiles configured in EPP",
			profiles: map[string]scheduling.SchedulerProfile{
				"profile1": newMockSchedulerProfile(),
				"profile2": newMockSchedulerProfile(),
			},
			profileResults:    map[string]*scheduling.ProfileRunResult{},
			expectEmptyResult: true,
			expectLogError:    true,
			description:       "Should return empty map and log error for multiple profiles",
		},
		{
			name:              "error: zero profiles configured in EPP",
			profiles:          map[string]scheduling.SchedulerProfile{},
			profileResults:    map[string]*scheduling.ProfileRunResult{},
			expectEmptyResult: true,
			expectLogError:    true,
			description:       "Should return empty map and log error for zero profiles",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := NewProfileHandler(8000).WithName("test-handler")
			ctx := context.Background()

			result := handler.Pick(ctx, &scheduling.CycleState{}, &scheduling.InferenceRequest{}, tt.profiles, tt.profileResults)

			if tt.expectEmptyResult {
				assert.Empty(t, result, tt.description)
			} else {
				assert.NotEmpty(t, result, tt.description)
				assert.Equal(t, len(tt.profiles), len(result), "Should return all profiles when valid")
			}
		})
	}
}

func Test_ProfileHandler_ProcessResults(t *testing.T) {
	tests := []struct {
		name           string
		primaryPort    int
		profileResults map[string]*scheduling.ProfileRunResult
		expectError    bool
		checkResult    func(*testing.T, *scheduling.SchedulingResult, map[string]string)
	}{
		{
			name: "error: multiple profiles not supported",
			profileResults: map[string]*scheduling.ProfileRunResult{
				"profile1": newMockProfileRunResult(DefaultTestPodPort, "pod1"),
				"profile2": newMockProfileRunResult(DefaultTestPodPort, "pod2"),
			},
			expectError: true,
		},
		{
			name: "error: single profile but result is nil",
			profileResults: map[string]*scheduling.ProfileRunResult{
				"nil-profile": nil,
			},
			expectError: true,
		},
		{
			name:        "success: single profile with primaryPort → port overridden, header set",
			primaryPort: 9000,
			profileResults: map[string]*scheduling.ProfileRunResult{
				"dp-profile": newMockProfileRunResult(DefaultTestPodPort, "pod1"),
			},
			expectError: false,
			checkResult: func(t *testing.T, res *scheduling.SchedulingResult, headers map[string]string) {
				assert.Equal(t, "dp-profile", res.PrimaryProfileName)

				pods := res.ProfileResults["dp-profile"].TargetEndpoints
				require.Len(t, pods, 1)
				assert.Equal(t, "9000", pods[0].GetMetadata().Port)                // overridden
				expectedHeader := net.JoinHostPort("10.0.0.1", DefaultTestPodPort) // original
				assert.Equal(t, expectedHeader, headers[routing.DataParallelEndpointHeader])
			},
		},
		{
			name:        "success: primaryPort=0 → port becomes '0'",
			primaryPort: 0,
			profileResults: map[string]*scheduling.ProfileRunResult{
				"dp": newMockProfileRunResult("8080", "pod1"),
			},
			expectError: false,
			checkResult: func(t *testing.T, res *scheduling.SchedulingResult, headers map[string]string) {
				pod := res.ProfileResults["dp"].TargetEndpoints[0]
				assert.Equal(t, "0", pod.GetMetadata().Port)
				assert.Equal(t, "10.0.0.1:8080", headers[routing.DataParallelEndpointHeader])
			},
		},
		{
			name:        "success: multiple target pods → all ports overridden",
			primaryPort: 8080,
			profileResults: map[string]*scheduling.ProfileRunResult{
				"dp-profile": newMockProfileRunResult(DefaultTestPodPort, "pod1", "pod2"),
			},
			expectError: false,
			checkResult: func(t *testing.T, res *scheduling.SchedulingResult, headers map[string]string) {
				pods := res.ProfileResults["dp-profile"].TargetEndpoints
				assert.Len(t, pods, 2)
				for _, p := range pods {
					assert.Equal(t, "8080", p.GetMetadata().Port)
				}
				assert.Equal(t, net.JoinHostPort("10.0.0.1", DefaultTestPodPort), headers[routing.DataParallelEndpointHeader])
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := NewProfileHandler(tt.primaryPort).WithName("test-handler")
			headers := make(map[string]string)
			req := &scheduling.InferenceRequest{Headers: headers}
			result, err := handler.ProcessResults(context.Background(), &scheduling.CycleState{}, req, tt.profileResults)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, result)
			tt.checkResult(t, result, headers)
		})
	}
}
