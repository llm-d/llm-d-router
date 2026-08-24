package disagg

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/test/utils"
)

const (
	testEndpointAddr = "10.0.0.1"
	testEndpointPort = "8000"

	// averageCharactersPerToken derives token counts from character-length
	// prompt fixtures in tests.
	averageCharactersPerToken = 4
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
			ID:      k8stypes.NamespacedName{Namespace: "default", Name: "test-pod"},
			Address: testEndpointAddr,
			Port:    testEndpointPort,
		},
		nil,
		fwkdl.NewAttributes(),
	)
}

func makeTestEndpoint(cachedTokens int) scheduling.Endpoint {
	ep := makeTestEndpointBase()
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey,
		attrprefix.NewPrefixCacheMatchInfo(cachedTokens, testTotalTokens, testBlockSize))
	return ep
}

// makeRequestWithTokens creates a completions request whose tokenized prompt carries
// the given token count, which getUserInputLenInTokens reads as the input length.
func makeRequestWithTokens(tokens int) *scheduling.InferenceRequest {
	return completionsRequest(strings.Repeat("x", tokens*averageCharactersPerToken))
}

// withTokens sets the tokenized prompt to carry n token IDs, which the decider reads
// as the input token count. Any existing tokenized prompt is preserved.
func withTokens(req *scheduling.InferenceRequest, n int) *scheduling.InferenceRequest {
	if req.Body.TokenizedRequest == nil {
		req.Body.TokenizedRequest = &fwkrh.TokenizedRequest{}
	}
	req.Body.TokenizedRequest.Prompts = []fwkrh.PromptTokens{{TokenIDs: make([]uint32, n)}}
	return req
}

func completionsRequestWithPrompt(prompt fwkrh.Prompt) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: prompt,
			},
		},
	}
}

func embeddingsRequestWithInput(input fwkrh.EmbeddingsInput) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Embeddings: &fwkrh.EmbeddingsRequest{
				Input: input,
			},
		},
	}
}

func TestGetUserInputLenInTokens(t *testing.T) {
	tests := []struct {
		name     string
		req      *scheduling.InferenceRequest
		wantMin  int // at least this many tokens
		wantZero bool
		want     int
	}{
		{
			name:    "completions prompt",
			req:     completionsRequest("hello world hello world"), // 23 chars → 5 tokens
			wantMin: 5,
		},
		{
			name:    "chat completions",
			req:     withTokens(chatRequest(false, false, false), 1),
			wantMin: 1,
		},
		{
			name: "completions string array prompt",
			req: &scheduling.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{
							Strings: []string{"hello world", "foo bar baz"},
						},
					},
					TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: make([]uint32, 5)}}},
				},
			},
			want: 5,
		},
		{
			name:     "empty completions prompt",
			req:      completionsRequest(""),
			wantZero: true,
		},
		{
			name: "completions prompt array",
			req: withTokens(completionsRequestWithPrompt(fwkrh.Prompt{
				Strings: []string{"hello", "world"},
			}), 2),
			wantMin: 2,
		},
		{
			name: "completions token ids uses exact hint",
			req: withTokens(completionsRequestWithPrompt(fwkrh.Prompt{
				TokenIDs: []uint32{1, 2, 3, 4},
			}), 4),
			want: 4,
		},
		{
			name: "embeddings input array",
			req: withTokens(embeddingsRequestWithInput(fwkrh.EmbeddingsInput{
				Strings: []string{"hello", "world"},
			}), 2),
			wantMin: 2,
		},
		{
			name: "embeddings token ids uses exact hint",
			req: withTokens(embeddingsRequestWithInput(fwkrh.EmbeddingsInput{
				TokenIDs: []uint32{1, 2, 3},
			}), 3),
			want: 3,
		},
		{
			name: "generate request returns exact token count",
			req: &scheduling.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Generate:         &fwkrh.GenerateRequest{TokenIDs: []uint32{1, 2, 3, 4, 5, 6, 7}},
					TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: make([]uint32, 7)}}},
				},
			},
			want: 7,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens, err := getUserInputLenInTokens(tt.req)
			assert.NoError(t, err)
			switch {
			case tt.wantZero:
				assert.Zero(t, tokens)
			case tt.want > 0:
				assert.Equal(t, tt.want, tokens)
			default:
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

			p, err := PrefixBasedPDDeciderPluginFactory(tt.pluginName, fwkplugin.StrictDecoder(raw), nil)
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
		promptTokens       int
		request            *scheduling.InferenceRequest
		endpoint           scheduling.Endpoint
		expectDisaggregate bool
		expectErr          bool
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
			name:               "input shorter than promptTokens threshold",
			nonCachedTokens:    5,
			promptTokens:       20,
			request:            makeRequestWithTokens(10),
			endpoint:           makeTestEndpoint(0),
			expectDisaggregate: false,
		},
		{
			name:            "negative promptTokens is invalid",
			nonCachedTokens: 5,
			promptTokens:    -1,
			request:         makeRequestWithTokens(10),
			endpoint:        makeTestEndpoint(0),
			expectErr:       true,
		},
		{
			name:               "input shorter than threshold",
			nonCachedTokens:    20,
			request:            makeRequestWithTokens(10),
			endpoint:           makeTestEndpoint(0),
			expectDisaggregate: false,
		},
		{
			name:               "input equals promptTokens threshold",
			nonCachedTokens:    5,
			promptTokens:       10,
			request:            makeRequestWithTokens(10),
			endpoint:           makeTestEndpoint(5),
			expectDisaggregate: true,
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
			decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{
				NonCachedTokens: tt.nonCachedTokens,
				PromptTokens:    tt.promptTokens,
			})
			if tt.expectErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)

			result := decider.disaggregate(ctx, tt.request, tt.endpoint)
			assert.Equal(t, tt.expectDisaggregate, result)
		})
	}
}

// TestDisaggregate_UsesUnweightedCachedBlockCount reproduces the #1047
// RAM-cache misrouting scenario. The precise prefix cache scorer stores a
// device-tier-weighted match score in matchBlocks (RAM tier = 0.8), but the
// literal cached-block count lives in cachedBlockCount. The decider must use
// the unweighted count so a mostly-RAM-cached prompt is not pushed onto the
// remote-prefill path.
//
// Issue parameters: blockSize=16, inputTokens=4096, a 240-block contiguous hit
// (3840 cached tokens, real non-cached suffix 256), threshold 512.
func TestDisaggregate_UsesUnweightedCachedBlockCount(t *testing.T) {
	ctx := utils.NewTestContext(t)

	const (
		blockSize        = 16
		inputTokens      = 4096
		totalBlocks      = inputTokens / blockSize // 256
		cachedBlocks     = 240                     // contiguous hit (3840 tokens)
		ramWeightedScore = 192                     // int(240 * 0.8) as stored in matchBlocks
		nonCachedTokens  = 512
	)

	newEndpoint := func(info *attrprefix.PrefixCacheMatchInfo) scheduling.Endpoint {
		ep := makeTestEndpointBase()
		ep.Put(attrprefix.PrefixCacheMatchInfoDataKey, info)
		return ep
	}
	// Each scenario uses its own request: the decider memoizes its decision on
	// the request, and production only ever pairs one request with one endpoint.
	newRequest := func() *scheduling.InferenceRequest {
		return withTokens(completionsRequestWithPrompt(fwkrh.Prompt{}), inputTokens)
	}

	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: nonCachedTokens})
	require.NoError(t, err)

	// Fixed behavior: cachedBlockCount carries the true 240 blocks, so
	// nonCached = 4096 - 240*16 = 256 < 512 → decode-only (no remote prefill).
	fixed := newEndpoint(attrprefix.NewPrefixCacheMatchInfo(ramWeightedScore, totalBlocks, blockSize).
		WithCachedBlockCount(cachedBlocks))
	assert.False(t, decider.disaggregate(ctx, newRequest(), fixed),
		"RAM-cached prefix must stay decode-only when the unweighted cached-block count is used")

	// Buggy behavior guard: if only the tier-weighted score (192) were
	// available as the block count, nonCached = 4096 - 192*16 = 1024 >= 512
	// would misroute to remote prefill.
	weightedOnly := newEndpoint(attrprefix.NewPrefixCacheMatchInfo(ramWeightedScore, totalBlocks, blockSize))
	assert.True(t, decider.disaggregate(ctx, newRequest(), weightedOnly),
		"sanity: the tier-weighted score alone undercounts cached blocks and misroutes")
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
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey, &notPrefixCacheMatchInfo{})

	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 5})
	require.NoError(t, err)

	assert.False(t, decider.disaggregate(ctx, makeRequestWithTokens(100), ep))
}

func TestConsumes(t *testing.T) {
	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 0})
	require.NoError(t, err)

	handler, err := NewPdProfileHandler(
		"test-handler",
		PdProfileHandlerParameters{
			PrefillProfile:              "prefill",
			DecodeProfile:               "decode",
			PrefixMatchInfoProducerName: "test",
		},
		decider,
	)
	require.NoError(t, err)

	consumed := handler.Consumes()
	assert.Contains(t, consumed.Required, attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("test"))
}

func TestWithName(t *testing.T) {
	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 0})
	require.NoError(t, err)

	decider.WithName("my-decider")
	assert.Equal(t, "my-decider", decider.TypedName().Name)

	decider.WithName("renamed")
	assert.Equal(t, "renamed", decider.TypedName().Name)
}

// withHeaders returns req with Headers set to the given map. Existing entries
// are replaced.
func withHeaders(req *scheduling.InferenceRequest, headers map[string]string) *scheduling.InferenceRequest {
	req.Headers = headers
	return req
}

// resultWithEndpoint builds a SchedulingResult with a single decode endpoint
// under the "decode" primary profile. Passing nil produces a result whose
// primary profile has a nil endpoint slot.
func resultWithEndpoint(ep scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode": {TargetEndpoints: []scheduling.Endpoint{ep}},
		},
	}
}

func TestPreRequest_ConditionalDecodeGate(t *testing.T) {
	ctx := utils.NewTestContext(t)

	preferIfAvailable := map[string]string{"prefer": "if-available"}
	returnMinimal := map[string]string{"prefer": "return=minimal"}

	// endpointNoAttr is a decode endpoint whose attributes carry no
	// PrefixCacheMatchInfo entry (simulates missing prefix-cache producer).
	endpointNoAttr := makeTestEndpointBase()

	// endpointWrongType puts a Cloneable of the wrong concrete type under the
	// PrefixCacheMatchInfo key.
	endpointWrongType := makeTestEndpointBase()
	endpointWrongType.Put(attrprefix.PrefixCacheMatchInfoDataKey, &notPrefixCacheMatchInfo{})

	tests := []struct {
		name            string
		nonCachedTokens int
		promptTokens    int
		headers         map[string]string
		result          *scheduling.SchedulingResult
		request         *scheduling.InferenceRequest
		wantReject      bool
	}{
		{
			name:            "no Prefer header forwards even without cache",
			nonCachedTokens: 5,
			headers:         nil,
			result:          nil,
			request:         makeRequestWithTokens(10),
			wantReject:      false,
		},
		{
			name:            "unrelated Prefer token forwards even without cache",
			nonCachedTokens: 5,
			headers:         returnMinimal,
			result:          nil,
			request:         makeRequestWithTokens(10),
			wantReject:      false,
		},
		{
			name:            "threshold zero disables the gate",
			nonCachedTokens: 0,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(makeTestEndpoint(0)),
			request:         makeRequestWithTokens(10),
			wantReject:      false,
		},
		{
			name:            "nil scheduling result rejects",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result:          nil,
			request:         makeRequestWithTokens(10),
			wantReject:      true,
		},
		{
			name:            "unknown primary profile rejects",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result: &scheduling.SchedulingResult{
				PrimaryProfileName: "decode",
				ProfileResults: map[string]*scheduling.ProfileRunResult{
					"other": {TargetEndpoints: []scheduling.Endpoint{makeTestEndpoint(10)}},
				},
			},
			request:    makeRequestWithTokens(10),
			wantReject: true,
		},
		{
			name:            "nil primary profile rejects",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result: &scheduling.SchedulingResult{
				PrimaryProfileName: "decode",
				ProfileResults:     map[string]*scheduling.ProfileRunResult{"decode": nil},
			},
			request:    makeRequestWithTokens(10),
			wantReject: true,
		},
		{
			name:            "empty TargetEndpoints rejects",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result: &scheduling.SchedulingResult{
				PrimaryProfileName: "decode",
				ProfileResults: map[string]*scheduling.ProfileRunResult{
					"decode": {TargetEndpoints: nil},
				},
			},
			request:    makeRequestWithTokens(10),
			wantReject: true,
		},
		{
			name:            "nil endpoint rejects",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(nil),
			request:         makeRequestWithTokens(10),
			wantReject:      true,
		},
		{
			name:            "missing PrefixCacheMatchInfo attribute forwards",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(endpointNoAttr),
			request:         makeRequestWithTokens(10),
			wantReject:      false,
		},
		{
			name:            "wrong-type attribute forwards",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(endpointWrongType),
			request:         makeRequestWithTokens(10),
			wantReject:      false,
		},
		{
			name:            "non-cached suffix below threshold forwards",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(makeTestEndpoint(8)),
			request:         makeRequestWithTokens(10),
			wantReject:      false,
		},
		{
			name:            "non-cached suffix equals threshold rejects",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(makeTestEndpoint(5)),
			request:         makeRequestWithTokens(10),
			wantReject:      true,
		},
		{
			name:            "non-cached suffix above threshold rejects",
			nonCachedTokens: 3,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(makeTestEndpoint(2)),
			request:         makeRequestWithTokens(10),
			wantReject:      true,
		},
		{
			name:            "zero blocks matched rejects",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(makeTestEndpoint(0)),
			request:         makeRequestWithTokens(10),
			wantReject:      true,
		},
		{
			name:            "no TokenizedPrompt: fails soft, forwards",
			nonCachedTokens: 5,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(makeTestEndpoint(0)),
			request:         completionsRequestWithPrompt(fwkrh.Prompt{}),
			wantReject:      false,
		},
		{
			name:            "input below promptTokens forwards (short-prompt shortcut)",
			nonCachedTokens: 5,
			promptTokens:    100,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(makeTestEndpoint(0)),
			request:         makeRequestWithTokens(10),
			wantReject:      false,
		},
		{
			name:            "input at or above promptTokens still rejects on cache miss",
			nonCachedTokens: 5,
			promptTokens:    10,
			headers:         preferIfAvailable,
			result:          resultWithEndpoint(makeTestEndpoint(0)),
			request:         makeRequestWithTokens(10),
			wantReject:      true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{
				NonCachedTokens: tt.nonCachedTokens,
				PromptTokens:    tt.promptTokens,
			})
			require.NoError(t, err)

			req := withHeaders(tt.request, tt.headers)
			err = decider.PreRequest(ctx, req, tt.result)
			if !tt.wantReject {
				assert.NoError(t, err)
				return
			}
			var e errcommon.Error
			require.ErrorAs(t, err, &e)
			assert.Equal(t, errcommon.PreconditionFailed, e.Code)
		})
	}
}

// TestPreRequest_UsesUnweightedCachedBlockCount is the PreRequest twin of
// TestDisaggregate_UsesUnweightedCachedBlockCount and pins the same #1047
// invariant: a RAM-cached prefix (unweighted count 240, tier-weighted score
// 192) must be seen as covering the prompt so the gate forwards it. If the
// gate ever reverts to reading MatchBlocks() it would misroute to 412.
func TestPreRequest_UsesUnweightedCachedBlockCount(t *testing.T) {
	ctx := utils.NewTestContext(t)

	const (
		blockSize        = 16
		inputTokens      = 4096
		totalBlocks      = inputTokens / blockSize // 256
		cachedBlocks     = 240                     // contiguous hit (3840 tokens)
		ramWeightedScore = 192                     // int(240 * 0.8) as stored in matchBlocks
		nonCachedTokens  = 512
	)

	newEndpoint := func(info *attrprefix.PrefixCacheMatchInfo) scheduling.Endpoint {
		ep := makeTestEndpointBase()
		ep.Put(attrprefix.PrefixCacheMatchInfoDataKey, info)
		return ep
	}
	// Each scenario uses its own request: the decider memoizes its decision on
	// the request, and production only ever pairs one request with one endpoint.
	newRequest := func() *scheduling.InferenceRequest {
		return withHeaders(
			withTokens(completionsRequestWithPrompt(fwkrh.Prompt{}), inputTokens),
			map[string]string{"prefer": "if-available"},
		)
	}

	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: nonCachedTokens})
	require.NoError(t, err)

	fixed := newEndpoint(attrprefix.NewPrefixCacheMatchInfo(ramWeightedScore, totalBlocks, blockSize).
		WithCachedBlockCount(cachedBlocks))
	assert.NoError(t, decider.PreRequest(ctx, newRequest(), resultWithEndpoint(fixed)),
		"RAM-cached prefix must forward when the unweighted cached-block count is used")

	weightedOnly := newEndpoint(attrprefix.NewPrefixCacheMatchInfo(ramWeightedScore, totalBlocks, blockSize))
	err = decider.PreRequest(ctx, newRequest(), resultWithEndpoint(weightedOnly))
	var e errcommon.Error
	require.ErrorAs(t, err, &e, "the tier-weighted score alone undercounts cached blocks and misroutes")
	assert.Equal(t, errcommon.PreconditionFailed, e.Code)
}

// TestNeedsRemotePrefill_MemoizesOnRequest pins the guarantee that PreRequest
// reuses the decision disaggregate made during scheduling: after disaggregate
// records "no remote prefill", PreRequest returns nil even against an endpoint
// whose cache state would otherwise trip the 412 gate.
func TestNeedsRemotePrefill_MemoizesOnRequest(t *testing.T) {
	ctx := utils.NewTestContext(t)

	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 5})
	require.NoError(t, err)

	req := withHeaders(makeRequestWithTokens(10), map[string]string{"prefer": "if-available"})

	// disaggregate against an endpoint whose non-cached suffix is under the
	// threshold: 10 - 8 = 2 < 5, so the memoized outcome is "no remote prefill".
	assert.False(t, decider.disaggregate(ctx, req, makeTestEndpoint(8)))

	// A follow-up PreRequest with an endpoint that would fail the gate on its
	// own (0 cached blocks → non-cached suffix 10 >= 5) must still forward,
	// proving PreRequest reused the memoized decision.
	assert.NoError(t, decider.PreRequest(ctx, req, resultWithEndpoint(makeTestEndpoint(0))))
}

// TestPreRequest_ClaimsConditionalDecodeAttribute pins the contract with the
// director's default-deny check: whenever PreRequest sees a Prefer:if-available
// request it must mark ConditionalDecodeHandledAttributeKey so the director
// knows some plugin owned the header. Non-conditional-decode requests must
// leave the attribute unset.
func TestPreRequest_ClaimsConditionalDecodeAttribute(t *testing.T) {
	ctx := utils.NewTestContext(t)

	tests := []struct {
		name            string
		nonCachedTokens int
		headers         map[string]string
		wantClaim       bool
	}{
		{
			name:            "conditional-decode request is claimed even when gate is disabled",
			nonCachedTokens: 0,
			headers:         map[string]string{"prefer": "if-available"},
			wantClaim:       true,
		},
		{
			name:            "conditional-decode request is claimed when gate is enabled",
			nonCachedTokens: 5,
			headers:         map[string]string{"prefer": "if-available"},
			wantClaim:       true,
		},
		{
			name:            "non-conditional-decode request is not claimed",
			nonCachedTokens: 5,
			headers:         map[string]string{"prefer": "return=minimal"},
			wantClaim:       false,
		},
		{
			name:            "request without Prefer header is not claimed",
			nonCachedTokens: 5,
			headers:         nil,
			wantClaim:       false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: tt.nonCachedTokens})
			require.NoError(t, err)

			req := withHeaders(makeRequestWithTokens(10), tt.headers)
			_ = decider.PreRequest(ctx, req, resultWithEndpoint(makeTestEndpoint(10)))

			_, claimed := req.GetAttribute(fwkrc.ConditionalDecodeHandledAttributeKey)
			assert.Equal(t, tt.wantClaim, claimed)
		})
	}
}
