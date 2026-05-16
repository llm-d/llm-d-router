package disagg

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/test/utils"
)

func TestPdProfileHandlerFactory(t *testing.T) {
	ctx := utils.NewTestContext(t)
	tests := []struct {
		name       string
		pluginName string
		params     map[string]any
		expectErr  bool
	}{
		{
			name:       "valid configuration with all defaults",
			pluginName: "default-handler",
			params:     map[string]any{},
			expectErr:  false,
		},
		{
			name:       "valid configuration with custom values",
			pluginName: "custom-handler",
			params: map[string]any{
				"decodeProfile":     "my-decode",
				"prefillProfile":    "my-prefill",
				"prefixPluginName":  "my-prefix-cache",
				"primaryPort":       8080,
				"deciderPluginName": PrefixBasedPDDeciderPluginType,
			},
			expectErr: false,
		},
		{
			name:       "zero primaryPort is allowed",
			pluginName: "zero-port",
			params: map[string]any{
				"primaryPort": 0,
			},
			expectErr: false,
		},
		{
			name:       "nonCachedTokens = 0 is allowed",
			pluginName: "zero-non-cached-tokens",
			params: map[string]any{
				"deciderPluginName": PrefixBasedPDDeciderPluginType,
			},
			expectErr: false,
		},
		{
			name:       "primaryPort below range should error",
			pluginName: "port-too-low",
			params:     map[string]any{"primaryPort": 0}, // OK
			expectErr:  false,
		},
		{
			name:       "primaryPort = 1 is valid",
			pluginName: "port-min",
			params:     map[string]any{"primaryPort": 1},
			expectErr:  false,
		},
		{
			name:       "primaryPort = 65535 is valid",
			pluginName: "port-max",
			params:     map[string]any{"primaryPort": 65535},
			expectErr:  false,
		},
		{
			name:       "empty decodeProfile is valid",
			pluginName: "empty-decode",
			params:     map[string]any{"decodeProfile": ""},
			expectErr:  false,
		},
		{
			name:       "empty prefillProfile is valid",
			pluginName: "empty-prefill",
			params:     map[string]any{"prefillProfile": ""},
			expectErr:  false,
		},
		{
			name:       "empty prefixPluginName is valid",
			pluginName: "empty-prefix-plugin",
			params:     map[string]any{"prefixPluginName": ""},
			expectErr:  false,
		},
		{
			name:       "primaryPort = 65536 should error",
			pluginName: "port-too-high",
			params:     map[string]any{"primaryPort": 65536},
			expectErr:  true,
		},
		{
			name:       "primaryPort = -10 should error",
			pluginName: "port-negative",
			params:     map[string]any{"primaryPort": -10},
			expectErr:  true,
		},
	}

	handle, err := createHandleWithDeciderPlugins(ctx)
	assert.NoError(t, err)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rawParams json.RawMessage
			if tt.params != nil {
				bytes, err := json.Marshal(tt.params)
				assert.NoError(t, err)
				rawParams = json.RawMessage(bytes)
			}
			plugin, err := PdProfileHandlerFactory(tt.pluginName, rawParams, handle)

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

func TestPdProfileHandlerFactoryInvalidJSON(t *testing.T) {
	ctx := utils.NewTestContext(t)

	invalidTests := []struct {
		name       string
		jsonParams string
	}{
		{
			name:       "malformed JSON",
			jsonParams: `{"deciderPluginName": `, // incomplete
		},
		{
			name:       "invalid decider plugin type",
			jsonParams: `{"deciderPluginName": "INVALID"}`,
		},
		{
			name:       "primaryPort as float",
			jsonParams: `{"primaryPort": 8080.5}`,
		},
	}

	handle, err := createHandleWithDeciderPlugins(ctx)
	assert.NoError(t, err)

	for _, tt := range invalidTests {
		t.Run(tt.name, func(t *testing.T) {
			rawParams := json.RawMessage(tt.jsonParams)
			plugin, err := PdProfileHandlerFactory("test", rawParams, handle)

			assert.Error(t, err)
			assert.Nil(t, plugin)
		})
	}
}

const DefaultTestPodPort = "8000"

// createEndpoint creates a mock Endpoint with customizable IP and port.
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

// newMockProfileRunResult creates a ProfileRunResult with Pods using the given port.
func newMockProfileRunResult(port string, endpointNames ...string) *scheduling.ProfileRunResult { //nolint:unparam // port varies in other packages using this pattern
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

// creates and returns llm completion request forthe given prompt
func createRequest(prompt string) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: prompt},
			},
		},
	}
}

// returns array of profile names in the given profile pick result
func getProfilesFromResult(result map[string]scheduling.SchedulerProfile) []string {
	profiles := make([]string, len(result))
	index := 0

	for name := range result {
		profiles[index] = name
		index++
	}

	return profiles
}

func TestPdProfileHandler_Pick(t *testing.T) {
	ctx := utils.NewTestContext(t)
	request := createRequest("hello world hello world hello world")

	profiles := map[string]scheduling.SchedulerProfile{
		"decode":  newMockSchedulerProfile(),
		"prefill": newMockSchedulerProfile(),
	}

	tests := []struct {
		name                 string
		nonCachedTokensLimit int
		prefixPluginType     string
		prefixPluginName     string
		cachedTokens         int
		profileResults       map[string]*scheduling.ProfileRunResult
		expectedProfiles     []string
	}{
		{
			name:                 "decode not executed yet → run decode",
			nonCachedTokensLimit: 10,
			prefixPluginType:     prefix.PrefixCacheScorerPluginType,
			prefixPluginName:     prefix.PrefixCacheScorerPluginType,
			profileResults:       map[string]*scheduling.ProfileRunResult{},
			expectedProfiles:     []string{defaultDecodeProfile},
		},
		{
			name:                 "decode failed (nil result) → run nothing",
			nonCachedTokensLimit: 10,
			prefixPluginType:     prefix.PrefixCacheScorerPluginType,
			prefixPluginName:     prefix.PrefixCacheScorerPluginType,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: nil,
			},
			expectedProfiles: []string{},
		},
		{
			name:                 "all profiles already executed → run nothing",
			nonCachedTokensLimit: 10,
			prefixPluginType:     prefix.PrefixCacheScorerPluginType,
			prefixPluginName:     prefix.PrefixCacheScorerPluginType,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile:  newMockProfileRunResult(DefaultTestPodPort, "pod1"),
				defaultPrefillProfile: newMockProfileRunResult(DefaultTestPodPort, "pod2"),
			},
			expectedProfiles: []string{},
		},
		{
			name: "has enough not-cached tokens → run prefill",
			// Need at least 4 non-cached tokens (16+ chars) to trigger disaggregated prefill
			// In this case: prompt length is 35 chars (8 tokens), cached length is 2 tokens -> disaggregated prefill should trigger
			nonCachedTokensLimit: 4,
			cachedTokens:         2,
			prefixPluginType:     prefix.PrefixCacheScorerPluginType,
			prefixPluginName:     prefix.PrefixCacheScorerPluginType,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
			},
			expectedProfiles: []string{defaultPrefillProfile},
		},
		{
			name: "short non-cached suffix → skip prefill",
			// Need at least 4 non-cached tokens (16+ chars) to trigger disaggregated prefill
			// In this case: prompt length is 35 chars (8 tokens), cached length is 5 tokens -> skip prefill
			nonCachedTokensLimit: 4,
			cachedTokens:         5,
			prefixPluginType:     prefix.PrefixCacheScorerPluginType,
			prefixPluginName:     prefix.PrefixCacheScorerPluginType,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
			},
			expectedProfiles: []string{},
		},
	}

	for _, tt := range tests {
		deciderPlugin, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: tt.nonCachedTokensLimit})
		assert.NoError(t, err)

		t.Run(tt.name, func(t *testing.T) {
			handler, err := NewPdProfileHandler(
				defaultPrefillProfile,
				defaultDecodeProfile,
				tt.prefixPluginType,
				tt.prefixPluginName,
				0,
				deciderPlugin,
			)
			assert.NoError(t, err)

			// set prefix to the given cached tokens number for pod "pod1" in decode profile results
			inputTokens := len(request.Body.Completions.Prompt.Raw) / AverageCharactersPerToken

			for profileName, profileRes := range tt.profileResults {
				if profileName == defaultDecodeProfile && profileRes != nil {
					for _, pod := range profileRes.TargetEndpoints {
						pod.Put(attrprefix.PrefixCacheMatchInfoKey,
							attrprefix.NewPrefixCacheMatchInfo(tt.cachedTokens, inputTokens, 1))
					}
				}
			}
			result := handler.Pick(ctx, nil, request, profiles, tt.profileResults)
			assert.ElementsMatch(t, tt.expectedProfiles, getProfilesFromResult(result))
		})
	}
}

func TestPdProfileHandler_PickSeries(t *testing.T) {
	ctx := context.Background()
	prompt := "hello world, hello world, hello world, hello world, hello world, hello world, hello world!"
	request := createRequest(prompt)
	longerRequest := createRequest(prompt + "123")
	longRequest := createRequest(prompt + prompt)

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  newMockSchedulerProfile(),
		defaultPrefillProfile: newMockSchedulerProfile(),
	}
	profileResults := map[string]*scheduling.ProfileRunResult{
		defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
	}

	type testData struct {
		request          *scheduling.InferenceRequest
		cachedTokens     int
		expectedProfiles []string
	}
	tests := []struct {
		name                 string
		nonCachedTokensLimit int
		tests                []testData
	}{
		{
			name:                 "same request twice",
			nonCachedTokensLimit: 2,
			tests: []testData{{
				request:          request,
				cachedTokens:     0,
				expectedProfiles: []string{defaultPrefillProfile},
			}, {
				request:          request,
				cachedTokens:     len(request.Body.Completions.Prompt.Raw) / AverageCharactersPerToken,
				expectedProfiles: []string{},
			}},
		}, {
			name: "short request and a little bit longer after it",
			// Need at least 2 non-cached tokens (8+ chars) to trigger disaggregated prefill
			// In this case: longer request is longer in 4 chars than the request -> no disaggregated prefill
			nonCachedTokensLimit: 2,
			tests: []testData{{
				request:          request,
				cachedTokens:     0,
				expectedProfiles: []string{defaultPrefillProfile},
			}, {
				request:          longerRequest,
				cachedTokens:     len(request.Body.Completions.Prompt.Raw) / AverageCharactersPerToken,
				expectedProfiles: []string{},
			}},
		}, {
			name: "short request and a long one after it",
			// Need at least 2 non-cached tokens (8+ chars) to trigger disaggregated prefill
			// In this case: long request is longer enough than the request -> should have disaggregated prefill
			nonCachedTokensLimit: 2,
			tests: []testData{{
				request:          request,
				cachedTokens:     0,
				expectedProfiles: []string{defaultPrefillProfile},
			}, {
				request:          longRequest,
				cachedTokens:     len(request.Body.Completions.Prompt.Raw) / AverageCharactersPerToken,
				expectedProfiles: []string{defaultPrefillProfile},
			}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deciderPlugin, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: tt.nonCachedTokensLimit})
			assert.NoError(t, err)

			handler, err := NewPdProfileHandler(
				defaultPrefillProfile,
				defaultDecodeProfile,
				prefix.PrefixCacheScorerPluginType,
				prefix.PrefixCacheScorerPluginType,
				0,
				deciderPlugin,
			)
			assert.NoError(t, err)

			// run sequences of request
			for _, innerTest := range tt.tests {
				cs := &scheduling.CycleState{}

				// set prefix to the given cached tokens number for pod "pod1" in decode profile results
				inputTokens := len(innerTest.request.Body.Completions.Prompt.Raw) / AverageCharactersPerToken

				for profileName, profileRes := range profileResults {
					if profileName == defaultDecodeProfile && profileRes != nil {
						for _, endpoint := range profileRes.TargetEndpoints {
							endpoint.Put(attrprefix.PrefixCacheMatchInfoKey,
								attrprefix.NewPrefixCacheMatchInfo(innerTest.cachedTokens, inputTokens, 1))
						}
					}
				}

				result := handler.Pick(ctx, cs, innerTest.request, profiles, profileResults)
				assert.ElementsMatch(t, innerTest.expectedProfiles, getProfilesFromResult(result))
			}
		})
	}
}

func TestPdProfileHandler_ProcessResults(t *testing.T) {
	tests := []struct {
		name           string
		primaryPort    int
		profileResults map[string]*scheduling.ProfileRunResult
		expectError    bool
		checkResult    func(*testing.T, *scheduling.SchedulingResult, map[string]string)
	}{
		{
			name: "decode failed → error",
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: nil,
			},
			expectError: true,
		},
		{
			name:        "decode success, no prefill, no primaryPort",
			primaryPort: 0,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
			},
			expectError: false,
			checkResult: func(t *testing.T, res *scheduling.SchedulingResult, headers map[string]string) {
				assert.Equal(t, defaultDecodeProfile, res.PrimaryProfileName)
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultPrefillProfile)
				metadata := res.ProfileResults[defaultDecodeProfile].TargetEndpoints[0].GetMetadata()
				assert.Equal(t, DefaultTestPodPort, metadata.Port)
				assert.Empty(t, headers[routing.DataParallelEndpointHeader])
			},
		},
		{
			name:        "decode success, with prefill",
			primaryPort: 0,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile:  newMockProfileRunResult(DefaultTestPodPort, "pod1"),
				defaultPrefillProfile: newMockProfileRunResult(DefaultTestPodPort, "pod2"),
			},
			expectError: false,
			checkResult: func(t *testing.T, res *scheduling.SchedulingResult, _ map[string]string) {
				assert.Equal(t, defaultDecodeProfile, res.PrimaryProfileName)
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.Contains(t, res.ProfileResults, defaultPrefillProfile)
			},
		},
		{
			name:        "with primaryPort → port updated and header set",
			primaryPort: 9000,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: newMockProfileRunResult(DefaultTestPodPort, "pod1"),
			},
			expectError: false,
			checkResult: func(t *testing.T, res *scheduling.SchedulingResult, headers map[string]string) {
				metadata := res.ProfileResults[defaultDecodeProfile].TargetEndpoints[0].GetMetadata()
				assert.Equal(t, "9000", metadata.Port)

				hostPort := headers[routing.DataParallelEndpointHeader]
				assert.Equal(t, "10.0.0.1:8000", hostPort)
			},
		},
	}

	for _, tt := range tests {
		deciderPlugin, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 0})
		assert.NoError(t, err)

		t.Run(tt.name, func(t *testing.T) {
			handler, err := NewPdProfileHandler(
				defaultPrefillProfile,
				defaultDecodeProfile,
				prefix.PrefixCacheScorerPluginType,
				prefix.PrefixCacheScorerPluginType,
				tt.primaryPort,
				deciderPlugin,
			)
			assert.NoError(t, err)

			headers := make(map[string]string)
			req := &scheduling.InferenceRequest{
				Headers: headers,
			}
			result, err := handler.ProcessResults(context.Background(), &scheduling.CycleState{}, req, tt.profileResults)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, result)
			tt.checkResult(t, result, headers)
		})
	}
}

func createHandleWithDeciderPlugins(ctx context.Context) (plugin.Handle, error) {
	handle := plugin.NewEppHandle(ctx, nil)
	plugin1, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 4})
	if err != nil {
		return nil, err
	}
	handle.AddPlugin(PrefixBasedPDDeciderPluginType, plugin1)
	plugin2 := newAlwaysDisaggPDDecider()
	handle.AddPlugin(AlwaysDisaggPDDeciderPluginType, plugin2)

	return handle, nil
}
