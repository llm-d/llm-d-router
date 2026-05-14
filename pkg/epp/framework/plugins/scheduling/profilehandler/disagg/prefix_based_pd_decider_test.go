package disagg

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

const (
	testEndpointAddr = "10.0.0.1"
	testEndpointPort = "8000"
)

// notPrefixCacheMatchInfo is a Cloneable type that is not *PrefixCacheMatchInfo, used to test type assertion failure.
type notPrefixCacheMatchInfo struct{}

func (n *notPrefixCacheMatchInfo) Clone() fwkdl.Cloneable { return &notPrefixCacheMatchInfo{} }

const (
	testTotalTokens = 10
	testBlockSize   = 1
)

func makeTestEndpointBase() scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "test-pod"},
			Address:        testEndpointAddr,
			Port:           testEndpointPort,
		},
		nil,
		fwkdl.NewAttributes(),
	)
}

func makeTestEndpoint(cachedTokens int) scheduling.Endpoint {
	ep := makeTestEndpointBase()
	ep.Put(attrprefix.PrefixCacheMatchInfoKey,
		attrprefix.NewPrefixCacheMatchInfo(cachedTokens, testTotalTokens, testBlockSize))
	return ep
}

// makeRequestWithTokens creates a completions request whose prompt yields the given token count
// via getUserInputLenInTokens (len(prompt) / AverageCharactersPerToken).
func makeRequestWithTokens(tokens int) *scheduling.InferenceRequest {
	return completionsRequest(strings.Repeat("x", tokens*AverageCharactersPerToken))
}

func TestGetUserInputLenInTokens(t *testing.T) {
	tests := []struct {
		name     string
		req      *scheduling.InferenceRequest
		wantMin  int // at least this many tokens
		wantZero bool
	}{
		{
			name:    "completions prompt",
			req:     completionsRequest("hello world hello world"), // 23 chars → 5 tokens
			wantMin: 5,
		},
		{
			name:    "chat completions",
			req:     chatRequest(false, false, false),
			wantMin: 1,
		},
		{
			name:     "empty completions prompt",
			req:      completionsRequest(""),
			wantZero: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens, err := getUserInputLenInTokens(tt.req)
			assert.NoError(t, err)
			if tt.wantZero {
				assert.Zero(t, tokens)
			} else {
				assert.GreaterOrEqual(t, tokens, tt.wantMin)
			}
		})
	}
}

func TestPrefixBasedPDDeciderConfigValidation(t *testing.T) {
	tests := []struct {
		name      string
		config    PrefixBasedPDDeciderConfig
		expectErr bool
	}{
		{
			name:      "zero is valid",
			config:    PrefixBasedPDDeciderConfig{NonCachedTokens: 0},
			expectErr: false,
		},
		{
			name:      "positive is valid",
			config:    PrefixBasedPDDeciderConfig{NonCachedTokens: 100},
			expectErr: false,
		},
		{
			name:      "negative is invalid",
			config:    PrefixBasedPDDeciderConfig{NonCachedTokens: -1},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewPrefixBasedPDDecider(tt.config)
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPrefixBasedPDDeciderFactory(t *testing.T) {
	tests := []struct {
		name             string
		pluginName       string
		rawParams        string
		expectErr        bool
		expectNonCached  int
		expectPluginName string
	}{
		{
			name:             "default parameters (nil)",
			pluginName:       "my-decider",
			rawParams:        "",
			expectErr:        false,
			expectNonCached:  0,
			expectPluginName: "my-decider",
		},
		{
			name:             "custom nonCachedTokens",
			pluginName:       "custom-decider",
			rawParams:        `{"nonCachedTokens": 50}`,
			expectErr:        false,
			expectNonCached:  50,
			expectPluginName: "custom-decider",
		},
		{
			name:       "negative nonCachedTokens",
			pluginName: "bad-decider",
			rawParams:  `{"nonCachedTokens": -5}`,
			expectErr:  true,
		},
		{
			name:       "invalid json",
			pluginName: "bad-json",
			rawParams:  `{invalid}`,
			expectErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var raw json.RawMessage
			if tt.rawParams != "" {
				raw = json.RawMessage(tt.rawParams)
			}

			p, err := PrefixBasedPDDeciderPluginFactory(tt.pluginName, plugin.StrictDecoder(raw), nil)
			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, p)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, p)

			decider, ok := p.(*PrefixBasedPDDecider)
			require.True(t, ok)
			assert.Equal(t, tt.expectPluginName, decider.TypedName().Name)
			assert.Equal(t, tt.expectNonCached, decider.config.NonCachedTokens)
		})
	}
}

func TestDisaggregate(t *testing.T) {
	ctx := utils.NewTestContext(t)

	tests := []struct {
		name               string
		nonCachedTokens    int
		request            *scheduling.InferenceRequest
		endpoint           scheduling.Endpoint
		expectDisaggregate bool
	}{
		{
			name:               "threshold zero disables disaggregation",
			nonCachedTokens:    0,
			request:            makeRequestWithTokens(10),
			endpoint:           makeTestEndpoint(5),
			expectDisaggregate: false,
		},
		{
			name:               "threshold zero with nil endpoint disables disaggregation",
			nonCachedTokens:    0,
			request:            makeRequestWithTokens(10),
			endpoint:           nil,
			expectDisaggregate: false,
		},
		{
			name:               "nil endpoint returns false",
			nonCachedTokens:    5,
			request:            makeRequestWithTokens(10),
			endpoint:           nil,
			expectDisaggregate: false,
		},
		{
			name:               "input shorter than threshold",
			nonCachedTokens:    20,
			request:            makeRequestWithTokens(10),
			endpoint:           makeTestEndpoint(0),
			expectDisaggregate: false,
		},
		{
			name:               "non-cached suffix below threshold",
			nonCachedTokens:    5,
			request:            makeRequestWithTokens(10),
			endpoint:           makeTestEndpoint(8),
			expectDisaggregate: false,
		},
		{
			name:               "non-cached suffix equals threshold",
			nonCachedTokens:    5,
			request:            makeRequestWithTokens(10),
			endpoint:           makeTestEndpoint(5),
			expectDisaggregate: true,
		},
		{
			name:               "non-cached suffix above threshold",
			nonCachedTokens:    3,
			request:            makeRequestWithTokens(10),
			endpoint:           makeTestEndpoint(2),
			expectDisaggregate: true,
		},
		{
			name:               "fully cached prompt",
			nonCachedTokens:    1,
			request:            makeRequestWithTokens(10),
			endpoint:           makeTestEndpoint(10),
			expectDisaggregate: false,
		},
		{
			name:               "no cache hit at all",
			nonCachedTokens:    5,
			request:            makeRequestWithTokens(10),
			endpoint:           makeTestEndpoint(0),
			expectDisaggregate: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: tt.nonCachedTokens})
			require.NoError(t, err)

			result := decider.disaggregate(ctx, tt.request, tt.endpoint)
			assert.Equal(t, tt.expectDisaggregate, result)
		})
	}
}

func TestDisaggregateNoPrefixInfo(t *testing.T) {
	ctx := utils.NewTestContext(t)

	ep := makeTestEndpointBase()

	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 5})
	require.NoError(t, err)

	assert.False(t, decider.disaggregate(ctx, makeRequestWithTokens(100), ep))
}

func TestDisaggregateWrongPrefixInfoType(t *testing.T) {
	ctx := utils.NewTestContext(t)

	ep := makeTestEndpointBase()
	ep.Put(attrprefix.PrefixCacheMatchInfoKey, &notPrefixCacheMatchInfo{})

	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 5})
	require.NoError(t, err)

	assert.False(t, decider.disaggregate(ctx, makeRequestWithTokens(100), ep))
}

func TestConsumes(t *testing.T) {
	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 0})
	require.NoError(t, err)

	handler, err := NewPdProfileHandler("prefill", "decode", PrefixBasedPDDeciderPluginType, "test", 0, decider)
	require.NoError(t, err)

	consumed := handler.Consumes()
	assert.Contains(t, consumed, attrprefix.PrefixCacheMatchInfoKey)
}

func TestWithName(t *testing.T) {
	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 0})
	require.NoError(t, err)

	decider.WithName("my-decider")
	assert.Equal(t, "my-decider", decider.TypedName().Name)

	decider.WithName("renamed")
	assert.Equal(t, "renamed", decider.TypedName().Name)
}
