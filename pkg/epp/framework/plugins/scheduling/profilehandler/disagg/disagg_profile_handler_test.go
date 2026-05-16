package disagg

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/test/utils"
)

// ── Shared test helpers ──────────────────────────────────────────────────────

const (
	testPodPort = "8000"

	// Custom profile names for testing user-defined configurations.
	customDecodeProfile  = "my-decode"
	customPrefillProfile = "my-prefill"
	customEncodeProfile  = "my-encode"

	// Test prompts
	testLongPrompt = "hello world hello world hello world"
)

func makeEndpoint(nsn k8stypes.NamespacedName, ip, port string, labels map[string]string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: nsn, Address: ip, Port: port, Labels: labels},
		nil,
		fwkdl.NewAttributes(),
	)
}

func makeProfileRunResult(names ...string) *scheduling.ProfileRunResult {
	eps := make([]scheduling.Endpoint, 0, len(names))
	for i, name := range names {
		eps = append(eps, makeEndpoint(
			k8stypes.NamespacedName{Namespace: "default", Name: name},
			fmt.Sprintf("10.0.0.%d", i+1), testPodPort, nil,
		))
	}
	return &scheduling.ProfileRunResult{TargetEndpoints: eps}
}

type mockProfile struct{}

func (p *mockProfile) Run(_ context.Context, _ *scheduling.InferenceRequest, _ *scheduling.CycleState, _ []scheduling.Endpoint) (*scheduling.ProfileRunResult, error) {
	return &scheduling.ProfileRunResult{}, nil
}

func profileNames(m map[string]scheduling.SchedulerProfile) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// completionsRequest builds a text-only InferenceRequest.
func completionsRequest(prompt string) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}},
		},
	}
}

// chatRequest builds a chat-completions InferenceRequest with optional multimodal blocks.
func chatRequest(hasImage, hasVideo, hasAudio bool) *scheduling.InferenceRequest {
	blocks := []fwkrh.ContentBlock{{Type: "text", Text: "describe this"}}
	if hasImage {
		blocks = append(blocks, fwkrh.ContentBlock{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/img.jpg"}})
	}
	if hasVideo {
		blocks = append(blocks, fwkrh.ContentBlock{Type: "video_url"})
	}
	if hasAudio {
		blocks = append(blocks, fwkrh.ContentBlock{Type: "input_audio"})
	}
	return &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Structured: blocks}}},
			},
		},
	}
}

// withPrompt adds a completions body to a chat request so the PD decider can estimate tokens.
func withPrompt(req *scheduling.InferenceRequest, prompt string) *scheduling.InferenceRequest {
	req.Body.Completions = &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}}
	return req
}

// injectPrefixCache sets prefix-cache match info on the decode endpoint for decider evaluation.
func injectPrefixCache(profileResults map[string]*scheduling.ProfileRunResult, cachedTokens, inputTokens int) {
	res, ok := profileResults[defaultDecodeProfile]
	if !ok || res == nil {
		return
	}
	for _, ep := range res.TargetEndpoints {
		ep.Put(attrprefix.PrefixCacheMatchInfoKey,
			attrprefix.NewPrefixCacheMatchInfo(cachedTokens, inputTokens, 1))
	}
}

// handleWithDeciders creates a plugin handle pre-loaded with all decider types.
func handleWithDeciders(ctx context.Context) plugin.Handle {
	h := plugin.NewEppHandle(ctx, nil)
	p1, _ := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 4})
	h.AddPlugin(PrefixBasedPDDeciderPluginType, p1)
	h.AddPlugin(AlwaysDisaggPDDeciderPluginType, newAlwaysDisaggPDDecider())
	h.AddPlugin(AlwaysDisaggMulimodalPluginType, newAlwaysDisaggEncodeDecider())
	return h
}

type mockEncodeDecider struct {
	allow bool
}

func (m *mockEncodeDecider) TypedName() plugin.TypedName { return plugin.TypedName{} }

func (m *mockEncodeDecider) disaggregate(_ context.Context, _ *scheduling.InferenceRequest, _ scheduling.Endpoint) bool {
	return m.allow
}

// ── Helper function tests ────────────────────────────────────────────────────

func TestHasMultimodalContent(t *testing.T) {
	tests := []struct {
		name     string
		req      *scheduling.InferenceRequest
		expected bool
	}{
		{"nil request", nil, false},
		{"nil body", &scheduling.InferenceRequest{Body: nil}, false},
		{"nil chat completions", &scheduling.InferenceRequest{Body: &fwkrh.InferenceRequestBody{}}, false},
		{"text only", chatRequest(false, false, false), false},
		{"image", chatRequest(true, false, false), true},
		{"video", chatRequest(false, true, false), true},
		{"audio", chatRequest(false, false, true), true},
		{"all types", chatRequest(true, true, true), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, hasMultimodalContent(tt.req))
		})
	}
}

// ── TypedName / WithName ─────────────────────────────────────────────────────

func TestHandler_TypedName(t *testing.T) {
	h := NewDisaggProfileHandler(defaultDecodeProfile, "", defaultEncodeProfile, nil, nil)
	assert.Equal(t, DisaggProfileHandlerType, h.TypedName().Type)
	assert.Empty(t, h.TypedName().Name)

	h.WithName("my-handler")
	assert.Equal(t, "my-handler", h.TypedName().Name)
	assert.Equal(t, DisaggProfileHandlerType, h.TypedName().Type)
}

// ── Factory tests ─────────────────────────────────────────────────────────────

func TestHandlerFactory(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := handleWithDeciders(ctx)

	tests := []struct {
		name      string
		params    map[string]any
		expectErr bool
	}{
		// decode-only (no prefill, no encode)
		{"decode only defaults", map[string]any{}, false},

		// P/D style (prefill + decode)
		{"PD style", map[string]any{
			"deciders": map[string]any{"prefill": AlwaysDisaggPDDeciderPluginType},
		}, false},
		{"PD custom profiles", map[string]any{
			"profiles": map[string]any{"decode": "my-decode", "prefill": "my-prefill"},
			"deciders": map[string]any{"prefill": PrefixBasedPDDeciderPluginType},
		}, false},

		// E/PD style (encode + decode)
		{"EPD style", map[string]any{
			"profiles": map[string]any{"encode": "encode"},
		}, false},
		{"EPD with encode decider", map[string]any{
			"profiles": map[string]any{"encode": "encode"},
			"deciders": map[string]any{"encode": AlwaysDisaggMulimodalPluginType},
		}, false},

		// E/P/D style (all three)
		{"full EPD", map[string]any{
			"profiles": map[string]any{"prefill": "prefill", "encode": "encode"},
			"deciders": map[string]any{
				"prefill": PrefixBasedPDDeciderPluginType,
				"encode":  AlwaysDisaggMulimodalPluginType,
			},
		}, false},

		// decider errors
		{"prefill without pdDecider is ok (stage inactive)", map[string]any{
			"profiles": map[string]any{"prefill": "prefill"},
		}, false},
		{"unknown pdDecider", map[string]any{
			"profiles": map[string]any{"prefill": "prefill"},
			"deciders": map[string]any{"prefill": "INVALID"},
		}, true},
		{"unknown encodeDecider", map[string]any{
			"deciders": map[string]any{"encode": "INVALID"},
		}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, _ := json.Marshal(tt.params)
			p, err := HandlerFactory("h", b, handle)
			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, p)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, p)
			}
		})
	}
}

func TestHandlerFactory_DeprecatedFlatParams(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := handleWithDeciders(ctx)

	tests := []struct {
		name      string
		params    map[string]any
		expectErr bool
	}{
		{"deprecated prefillDeciderPluginName", map[string]any{
			"prefillDeciderPluginName": PrefixBasedPDDeciderPluginType,
		}, false},
		{"deprecated encodeDeciderPluginName", map[string]any{
			"encodeDeciderPluginName": AlwaysDisaggMulimodalPluginType,
		}, false},
		{"deprecated custom profile names", map[string]any{
			"decodeProfile":            "my-decode",
			"prefillProfile":           "my-prefill",
			"encodeProfile":            "my-encode",
			"prefillDeciderPluginName": PrefixBasedPDDeciderPluginType,
		}, false},
		{"nested format with unknown extra fields is accepted", map[string]any{
			"profiles":     map[string]any{"decode": "decode"},
			"unknownField": "ignored",
		}, false},
		{"mixing deprecated and nested fields is an error", map[string]any{
			"decodeProfile": "my-decode",
			"profiles":      map[string]any{"decode": "other-decode"},
		}, true},
		{"mixing deprecated decider and nested deciders is an error", map[string]any{
			"prefillDeciderPluginName": PrefixBasedPDDeciderPluginType,
			"deciders":                 map[string]any{"prefill": AlwaysDisaggPDDeciderPluginType},
		}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, _ := json.Marshal(tt.params)
			p, err := HandlerFactory("h", b, handle)
			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, p)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, p)
			}
		})
	}
}

// TestHandlerFactory_PdProfileHandlerParams verifies that
// Handler accepts the exact parameter format of the deprecated
// pd-profile-handler, enabling a zero-change migration between the two types.
func TestHandlerFactory_PdProfileHandlerParams(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := handleWithDeciders(ctx)

	tests := []struct {
		name      string
		params    map[string]any
		expectErr bool
	}{
		{"pd-profile-handler defaults (no params)", map[string]any{}, false},
		{"pd-profile-handler with deciderPluginName", map[string]any{
			"decodeProfile":     "decode",
			"prefillProfile":    "prefill",
			"deciderPluginName": PrefixBasedPDDeciderPluginType,
		}, false},
		{"pd-profile-handler with all params including ignored fields", map[string]any{
			"decodeProfile":     "decode",
			"prefillProfile":    "prefill",
			"deciderPluginName": PrefixBasedPDDeciderPluginType,
			"prefixPluginType":  "prefix-cache-scorer", // ignored by Handler
			"prefixPluginName":  "prefix-cache-scorer", // ignored by Handler
			"primaryPort":       8080,                  // ignored by Handler
		}, false},
		{"pd-profile-handler unknown deciderPluginName", map[string]any{
			"deciderPluginName": "INVALID",
		}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, _ := json.Marshal(tt.params)
			p, err := HandlerFactory("h", b, handle)
			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, p)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, p)
			}
		})
	}
}

func TestHandlerFactory_InvalidJSON(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := handleWithDeciders(ctx)
	for _, raw := range []string{`{"deciders": `} {
		p, err := HandlerFactory("h", json.RawMessage(raw), handle)
		assert.Error(t, err)
		assert.Nil(t, p)
	}
}

// ── P/D Pick tests ───────────────────────────────────────────────────────────

func TestHandler_Pick_PD(t *testing.T) {
	ctx := utils.NewTestContext(t)
	req := completionsRequest("hello world hello world hello world") // ~8 tokens

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  &mockProfile{},
		defaultPrefillProfile: &mockProfile{},
	}

	tests := []struct {
		name            string
		nonCachedTokens int
		cachedTokens    int
		profileResults  map[string]*scheduling.ProfileRunResult
		want            []string
	}{
		{
			name:            "decode not run → run decode",
			nonCachedTokens: 4,
			profileResults:  map[string]*scheduling.ProfileRunResult{},
			want:            []string{defaultDecodeProfile},
		},
		{
			name:            "decode failed → done",
			nonCachedTokens: 4,
			profileResults:  map[string]*scheduling.ProfileRunResult{defaultDecodeProfile: nil},
			want:            []string{},
		},
		{
			name:            "all profiles done → done",
			nonCachedTokens: 4,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile:  makeProfileRunResult("pod1"),
				defaultPrefillProfile: makeProfileRunResult("pod2"),
			},
			want: []string{},
		},
		{
			name:            "enough uncached tokens → run prefill",
			nonCachedTokens: 4, cachedTokens: 2,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want: []string{defaultPrefillProfile},
		},
		{
			name:            "short uncached suffix → skip prefill",
			nonCachedTokens: 4, cachedTokens: 5,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: tt.nonCachedTokens})
			assert.NoError(t, err)

			h := NewDisaggProfileHandler(defaultDecodeProfile, defaultPrefillProfile, "",
				decider, nil)

			inputTokens := len(req.Body.Completions.Prompt.Raw) / AverageCharactersPerToken
			injectPrefixCache(tt.profileResults, tt.cachedTokens, inputTokens)

			got := h.Pick(ctx, nil, req, profiles, tt.profileResults)
			assert.ElementsMatch(t, tt.want, profileNames(got))
		})
	}
}

func TestHandler_Pick_PD_InputTokenError(t *testing.T) {
	ctx := utils.NewTestContext(t)
	// Request with neither Completions nor ChatCompletions → getUserInputLenInTokens fails.
	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{},
	}
	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  &mockProfile{},
		defaultPrefillProfile: &mockProfile{},
	}
	results := map[string]*scheduling.ProfileRunResult{
		defaultDecodeProfile: makeProfileRunResult("pod1"),
	}

	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 1})
	assert.NoError(t, err)
	h := NewDisaggProfileHandler(defaultDecodeProfile, defaultPrefillProfile, "",
		decider, nil)

	got := h.Pick(ctx, nil, req, profiles, results)
	assert.Empty(t, got, "should return empty map on input token estimation error")
}

func TestHandler_Pick_PD_Series(t *testing.T) {
	ctx := context.Background()
	short := completionsRequest("hello world, hello world!")
	long := completionsRequest("hello world, hello world! and some additional padding text here")

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  &mockProfile{},
		defaultPrefillProfile: &mockProfile{},
	}
	tests := []struct {
		name            string
		nonCachedTokens int
		steps           []struct {
			req          *scheduling.InferenceRequest
			cachedTokens int
			want         []string
		}
	}{
		{
			name:            "same request twice: first disaggregates, second hits cache",
			nonCachedTokens: 2,
			steps: []struct {
				req          *scheduling.InferenceRequest
				cachedTokens int
				want         []string
			}{
				{short, 0, []string{defaultPrefillProfile}},
				{short, len(short.Body.Completions.Prompt.Raw) / AverageCharactersPerToken, []string{}},
			},
		},
		{
			name:            "short then long: long triggers disaggregation",
			nonCachedTokens: 2,
			steps: []struct {
				req          *scheduling.InferenceRequest
				cachedTokens int
				want         []string
			}{
				{short, 0, []string{defaultPrefillProfile}},
				{long, len(short.Body.Completions.Prompt.Raw) / AverageCharactersPerToken, []string{defaultPrefillProfile}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: tt.nonCachedTokens})
			assert.NoError(t, err)
			h := NewDisaggProfileHandler(defaultDecodeProfile, defaultPrefillProfile, "",
				decider, nil)

			for _, step := range tt.steps {
				// Fresh results per step to avoid mutation leaking between iterations.
				results := map[string]*scheduling.ProfileRunResult{
					defaultDecodeProfile: makeProfileRunResult("pod1"),
				}
				inputTokens := len(step.req.Body.Completions.Prompt.Raw) / AverageCharactersPerToken
				injectPrefixCache(results, step.cachedTokens, inputTokens)
				got := h.Pick(ctx, &scheduling.CycleState{}, step.req, profiles, results)
				assert.ElementsMatch(t, step.want, profileNames(got))
			}
		})
	}
}

// ── P/D ProcessResults tests ─────────────────────────────────────────────────

func TestHandler_ProcessResults_PD(t *testing.T) {
	tests := []struct {
		name      string
		results   map[string]*scheduling.ProfileRunResult
		expectErr bool
		check     func(*testing.T, *scheduling.SchedulingResult)
	}{
		{
			name:      "decode failed → error",
			results:   map[string]*scheduling.ProfileRunResult{defaultDecodeProfile: nil},
			expectErr: true,
		},
		{
			name: "decode only",
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Equal(t, defaultDecodeProfile, res.PrimaryProfileName)
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultPrefillProfile)
				assert.Equal(t, testPodPort, res.ProfileResults[defaultDecodeProfile].TargetEndpoints[0].GetMetadata().Port)
			},
		},
		{
			name: "decode + prefill",
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile:  makeProfileRunResult("pod1"),
				defaultPrefillProfile: makeProfileRunResult("pod2"),
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.Contains(t, res.ProfileResults, defaultPrefillProfile)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decider, _ := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{})
			h := NewDisaggProfileHandler(defaultDecodeProfile, defaultPrefillProfile, "",
				decider, nil)

			req := &scheduling.InferenceRequest{Headers: map[string]string{}}
			res, err := h.ProcessResults(context.Background(), nil, req, tt.results)
			if tt.expectErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			tt.check(t, res)
		})
	}
}

func TestHandler_ProcessResults_NilRequest(t *testing.T) {
	h := NewDisaggProfileHandler(defaultDecodeProfile, defaultPrefillProfile, "",
		nil, nil)
	results := map[string]*scheduling.ProfileRunResult{
		defaultDecodeProfile: makeProfileRunResult("pod1"),
	}
	_, err := h.ProcessResults(context.Background(), nil, nil, results)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "request is nil")
}

// ── Custom profile name tests ─────────────────────────────────────────────────

func TestHandler_Pick_CustomProfiles(t *testing.T) {
	ctx := utils.NewTestContext(t)

	profiles := map[string]scheduling.SchedulerProfile{
		customDecodeProfile:  &mockProfile{},
		customPrefillProfile: &mockProfile{},
		customEncodeProfile:  &mockProfile{},
	}

	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 1})
	assert.NoError(t, err)

	h := NewDisaggProfileHandler(
		customDecodeProfile, customPrefillProfile, customEncodeProfile,
		decider, newAlwaysDisaggEncodeDecider(),
	)

	// Stage 1: decode not run → run decode
	got := h.Pick(ctx, nil, chatRequest(true, false, false), profiles, map[string]*scheduling.ProfileRunResult{})
	assert.ElementsMatch(t, []string{customDecodeProfile}, profileNames(got))

	// Stage 2: decode done, multimodal → run encode
	results := map[string]*scheduling.ProfileRunResult{
		customDecodeProfile: makeProfileRunResult("pod1"),
	}
	got = h.Pick(ctx, nil, chatRequest(true, false, false), profiles, results)
	assert.ElementsMatch(t, []string{customEncodeProfile}, profileNames(got))
}

func TestHandler_ProcessResults_CustomProfiles(t *testing.T) {
	h := NewDisaggProfileHandler(
		customDecodeProfile, customPrefillProfile, customEncodeProfile,
		nil, nil,
	)

	results := map[string]*scheduling.ProfileRunResult{
		customDecodeProfile:  makeProfileRunResult("pod1"),
		customPrefillProfile: makeProfileRunResult("pod2"),
		customEncodeProfile:  makeProfileRunResult("pod3"),
	}

	req := &scheduling.InferenceRequest{Headers: map[string]string{}}
	res, err := h.ProcessResults(context.Background(), nil, req, results)
	assert.NoError(t, err)
	assert.Equal(t, customDecodeProfile, res.PrimaryProfileName)
	assert.Contains(t, res.ProfileResults, customDecodeProfile)
	assert.Contains(t, res.ProfileResults, customPrefillProfile)
	assert.Contains(t, res.ProfileResults, customEncodeProfile)
}

// ── E/PD Pick tests ──────────────────────────────────────────────────────────

func TestHandler_Pick_EPD(t *testing.T) {
	ctx := utils.NewTestContext(t)

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile: &mockProfile{},
		defaultEncodeProfile: &mockProfile{},
	}

	tests := []struct {
		name    string
		req     *scheduling.InferenceRequest
		results map[string]*scheduling.ProfileRunResult
		want    []string
	}{
		{
			name:    "decode not run → run decode",
			req:     chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{},
			want:    []string{defaultDecodeProfile},
		},
		{
			name:    "decode failed → done",
			req:     chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{defaultDecodeProfile: nil},
			want:    []string{},
		},
		{
			name: "no multimodal → skip encode",
			req:  chatRequest(false, false, false),
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want: []string{},
		},
		{
			name: "image → run encode",
			req:  chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want: []string{defaultEncodeProfile},
		},
		{
			name: "video → run encode",
			req:  chatRequest(false, true, false),
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want: []string{defaultEncodeProfile},
		},
		{
			name: "audio → run encode",
			req:  chatRequest(false, false, true),
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want: []string{defaultEncodeProfile},
		},
		{
			name: "encode failed → done",
			req:  chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile: nil,
			},
			want: []string{},
		},
		{
			name: "all profiles done → done",
			req:  chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile: makeProfileRunResult("pod2"),
			},
			want: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(defaultDecodeProfile, "", defaultEncodeProfile, nil, newAlwaysDisaggEncodeDecider())
			got := h.Pick(ctx, nil, tt.req, profiles, tt.results)
			assert.ElementsMatch(t, tt.want, profileNames(got))
		})
	}
}

func TestHandler_Pick_EPD_EncodeDecider(t *testing.T) {
	ctx := utils.NewTestContext(t)

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile: &mockProfile{},
		defaultEncodeProfile: &mockProfile{},
	}
	results := map[string]*scheduling.ProfileRunResult{
		defaultDecodeProfile: makeProfileRunResult("pod1"),
	}

	tests := []struct {
		name  string
		allow bool
		want  []string
	}{
		{"decider approves → run encode", true, []string{defaultEncodeProfile}},
		{"decider rejects → skip encode", false, nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(defaultDecodeProfile, "", defaultEncodeProfile,
				nil, &mockEncodeDecider{allow: tt.allow})
			got := h.Pick(ctx, nil, chatRequest(true, false, false), profiles, results)
			assert.ElementsMatch(t, tt.want, profileNames(got))
		})
	}
}

// ── E/PD ProcessResults tests ────────────────────────────────────────────────

func TestHandler_ProcessResults_EPD(t *testing.T) {
	tests := []struct {
		name      string
		results   map[string]*scheduling.ProfileRunResult
		expectErr bool
		check     func(*testing.T, *scheduling.SchedulingResult)
	}{
		{
			name:      "decode failed → error",
			results:   map[string]*scheduling.ProfileRunResult{defaultDecodeProfile: nil},
			expectErr: true,
		},
		{
			name: "decode only",
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultEncodeProfile)
			},
		},
		{
			name: "decode + encode",
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile: makeProfileRunResult("pod2"),
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.Contains(t, res.ProfileResults, defaultEncodeProfile)
			},
		},
		{
			name: "encode nil (rejected) → omitted",
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile: nil,
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultEncodeProfile)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(defaultDecodeProfile, "", defaultEncodeProfile, nil, newAlwaysDisaggEncodeDecider())
			res, err := h.ProcessResults(context.Background(), nil, &scheduling.InferenceRequest{}, tt.results)
			if tt.expectErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, defaultDecodeProfile, res.PrimaryProfileName)
			tt.check(t, res)
		})
	}
}

// ── E/P/D Pick tests ─────────────────────────────────────────────────────────

func TestHandler_Pick_EPD_Full(t *testing.T) {
	ctx := utils.NewTestContext(t)

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  &mockProfile{},
		defaultPrefillProfile: &mockProfile{},
		defaultEncodeProfile:  &mockProfile{},
	}

	multimodalLong := withPrompt(chatRequest(true, false, false), testLongPrompt)

	tests := []struct {
		name            string
		req             *scheduling.InferenceRequest
		nonCachedTokens int
		cachedTokens    int
		results         map[string]*scheduling.ProfileRunResult
		want            []string
	}{
		{
			name:            "decode not run → run decode",
			req:             multimodalLong,
			nonCachedTokens: 1,
			results:         map[string]*scheduling.ProfileRunResult{},
			want:            []string{defaultDecodeProfile},
		},
		{
			name:            "decode failed → done",
			req:             multimodalLong,
			nonCachedTokens: 1,
			results:         map[string]*scheduling.ProfileRunResult{defaultDecodeProfile: nil},
			want:            []string{},
		},
		{
			name:            "multimodal → run encode next",
			req:             multimodalLong,
			nonCachedTokens: 1,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want: []string{defaultEncodeProfile},
		},
		{
			name:            "text-only, high uncached tokens → skip encode, run prefill",
			req:             completionsRequest(testLongPrompt),
			nonCachedTokens: 1, cachedTokens: 0,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want: []string{defaultPrefillProfile},
		},
		{
			name:            "text-only, prefill not needed → done",
			req:             completionsRequest(testLongPrompt),
			nonCachedTokens: 100,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want: []string{},
		},
		{
			name:            "encode failed → fall through, run prefill",
			req:             multimodalLong,
			nonCachedTokens: 1, cachedTokens: 0,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile: nil,
			},
			want: []string{defaultPrefillProfile},
		},
		{
			name:            "encode done → run prefill",
			req:             multimodalLong,
			nonCachedTokens: 1, cachedTokens: 0,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile: makeProfileRunResult("pod2"),
			},
			want: []string{defaultPrefillProfile},
		},
		{
			name:            "all three done → done",
			req:             multimodalLong,
			nonCachedTokens: 1,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile:  makeProfileRunResult("pod1"),
				defaultEncodeProfile:  makeProfileRunResult("pod2"),
				defaultPrefillProfile: makeProfileRunResult("pod3"),
			},
			want: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: tt.nonCachedTokens})
			assert.NoError(t, err)

			h := NewDisaggProfileHandler(
				defaultDecodeProfile, defaultPrefillProfile, defaultEncodeProfile,
				decider, newAlwaysDisaggEncodeDecider(),
			)

			inputTokens := 0
			if tt.req.Body.Completions != nil {
				inputTokens = len(tt.req.Body.Completions.Prompt.Raw) / AverageCharactersPerToken
			} else if tt.req.Body.ChatCompletions != nil {
				b, _ := json.Marshal(tt.req.Body.ChatCompletions.Messages)
				inputTokens = len(b) / AverageCharactersPerToken
			}
			injectPrefixCache(tt.results, tt.cachedTokens, inputTokens)

			got := h.Pick(ctx, nil, tt.req, profiles, tt.results)
			assert.ElementsMatch(t, tt.want, profileNames(got))
		})
	}
}

func TestHandler_Pick_EPD_Full_EncodeDecider(t *testing.T) {
	ctx := utils.NewTestContext(t)

	multimodalLong := withPrompt(chatRequest(true, false, false), testLongPrompt)

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  &mockProfile{},
		defaultPrefillProfile: &mockProfile{},
		defaultEncodeProfile:  &mockProfile{},
	}

	tests := []struct {
		name     string
		allow    bool
		wantNext []string // expected next profile from Pick (encode not yet run)
	}{
		{"decider approves → run encode next", true, []string{defaultEncodeProfile}},
		{"decider rejects → skip encode, run prefill next", false, []string{defaultPrefillProfile}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 1})
			assert.NoError(t, err)

			h := NewDisaggProfileHandler(
				defaultDecodeProfile, defaultPrefillProfile, defaultEncodeProfile,
				decider, &mockEncodeDecider{allow: tt.allow},
			)

			results := map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			}

			inputTokens := len(testLongPrompt) / AverageCharactersPerToken
			injectPrefixCache(results, 0, inputTokens)

			got := h.Pick(ctx, nil, multimodalLong, profiles, results)
			assert.ElementsMatch(t, tt.wantNext, profileNames(got))
		})
	}
}

// ── E/P/D ProcessResults tests ───────────────────────────────────────────────

func TestHandler_ProcessResults_EPD_Full(t *testing.T) {
	tests := []struct {
		name      string
		results   map[string]*scheduling.ProfileRunResult
		expectErr bool
		check     func(*testing.T, *scheduling.SchedulingResult)
	}{
		{
			name:      "decode failed → error",
			results:   map[string]*scheduling.ProfileRunResult{defaultDecodeProfile: nil},
			expectErr: true,
		},
		{
			name: "decode only",
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultEncodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultPrefillProfile)
			},
		},
		{
			name: "all three stages",
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile:  makeProfileRunResult("pod1"),
				defaultEncodeProfile:  makeProfileRunResult("pod2"),
				defaultPrefillProfile: makeProfileRunResult("pod3"),
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.Contains(t, res.ProfileResults, defaultEncodeProfile)
				assert.Contains(t, res.ProfileResults, defaultPrefillProfile)
			},
		},
		{
			name: "encode nil → omitted",
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile:  makeProfileRunResult("pod1"),
				defaultEncodeProfile:  nil,
				defaultPrefillProfile: makeProfileRunResult("pod3"),
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultEncodeProfile)
				assert.Contains(t, res.ProfileResults, defaultPrefillProfile)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decider, _ := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{})
			h := NewDisaggProfileHandler(
				defaultDecodeProfile, defaultPrefillProfile, defaultEncodeProfile,
				decider, newAlwaysDisaggEncodeDecider(),
			)
			res, err := h.ProcessResults(context.Background(), nil, &scheduling.InferenceRequest{}, tt.results)
			if tt.expectErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, defaultDecodeProfile, res.PrimaryProfileName)
			tt.check(t, res)
		})
	}
}

// ── Nil decider tests ────────────────────────────────────────────────────────

func TestHandler_Pick_NilDeciders(t *testing.T) {
	ctx := utils.NewTestContext(t)

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  &mockProfile{},
		defaultPrefillProfile: &mockProfile{},
		defaultEncodeProfile:  &mockProfile{},
	}

	multimodalLong := withPrompt(chatRequest(true, false, false), testLongPrompt)

	tests := []struct {
		name          string
		pdDecider     deciderPlugin
		encodeDecider deciderPlugin
		req           *scheduling.InferenceRequest
		results       map[string]*scheduling.ProfileRunResult
		want          []string
		description   string
	}{
		{
			name:          "both deciders nil, decode not run → run decode",
			pdDecider:     nil,
			encodeDecider: nil,
			req:           multimodalLong,
			results:       map[string]*scheduling.ProfileRunResult{},
			want:          []string{defaultDecodeProfile},
			description:   "Should run decode first regardless of nil deciders",
		},
		{
			name:          "both deciders nil, decode done → skip both encode and prefill",
			pdDecider:     nil,
			encodeDecider: nil,
			req:           multimodalLong,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want:        []string{},
			description: "With nil deciders, both encode and prefill should be skipped",
		},
		{
			name:          "pdDecider nil, encodeDecider present, multimodal → run encode",
			pdDecider:     nil,
			encodeDecider: newAlwaysDisaggEncodeDecider(),
			req:           multimodalLong,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want:        []string{defaultEncodeProfile},
			description: "Nil pdDecider should not affect encode stage",
		},
		{
			name:          "pdDecider nil, encodeDecider present, encode done → skip prefill",
			pdDecider:     nil,
			encodeDecider: newAlwaysDisaggEncodeDecider(),
			req:           multimodalLong,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile: makeProfileRunResult("pod2"),
			},
			want:        []string{},
			description: "Nil pdDecider should cause prefill to be skipped",
		},
		{
			name:          "encodeDecider nil, pdDecider present, text-only → run prefill",
			pdDecider:     newAlwaysDisaggPDDecider(),
			encodeDecider: nil,
			req:           completionsRequest(testLongPrompt),
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want:        []string{defaultPrefillProfile},
			description: "Nil encodeDecider should not affect prefill stage for text-only",
		},
		{
			name:          "encodeDecider nil, pdDecider present, multimodal → skip encode, run prefill",
			pdDecider:     newAlwaysDisaggPDDecider(),
			encodeDecider: nil,
			req:           multimodalLong,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want:        []string{defaultPrefillProfile},
			description: "Nil encodeDecider should skip encode even for multimodal, then run prefill",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(
				defaultDecodeProfile, defaultPrefillProfile, defaultEncodeProfile,
				tt.pdDecider, tt.encodeDecider,
			)

			// Inject prefix cache if needed for PD decider
			if tt.req.Body.Completions != nil {
				inputTokens := len(tt.req.Body.Completions.Prompt.Raw) / AverageCharactersPerToken
				injectPrefixCache(tt.results, 0, inputTokens)
			}

			got := h.Pick(ctx, nil, tt.req, profiles, tt.results)
			assert.ElementsMatch(t, tt.want, profileNames(got), tt.description)
		})
	}
}

func TestHandler_ProcessResults_NilDeciders(t *testing.T) {
	tests := []struct {
		name          string
		pdDecider     deciderPlugin
		encodeDecider deciderPlugin
		results       map[string]*scheduling.ProfileRunResult
		expectErr     bool
		check         func(*testing.T, *scheduling.SchedulingResult)
		description   string
	}{
		{
			name:          "both deciders nil, decode only",
			pdDecider:     nil,
			encodeDecider: nil,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultEncodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultPrefillProfile)
			},
			description: "Should only include decode profile when both deciders are nil",
		},
		{
			name:          "pdDecider nil, encode ran successfully",
			pdDecider:     nil,
			encodeDecider: newAlwaysDisaggEncodeDecider(),
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile: makeProfileRunResult("pod2"),
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.Contains(t, res.ProfileResults, defaultEncodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultPrefillProfile)
			},
			description: "Should include decode and encode, but not prefill when pdDecider is nil",
		},
		{
			name:          "encodeDecider nil, prefill ran successfully",
			pdDecider:     newAlwaysDisaggPDDecider(),
			encodeDecider: nil,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile:  makeProfileRunResult("pod1"),
				defaultPrefillProfile: makeProfileRunResult("pod3"),
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.NotContains(t, res.ProfileResults, defaultEncodeProfile)
				assert.Contains(t, res.ProfileResults, defaultPrefillProfile)
			},
			description: "Should include decode and prefill, but not encode when encodeDecider is nil",
		},
		{
			name:          "both deciders nil, decode failed → error",
			pdDecider:     nil,
			encodeDecider: nil,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: nil,
			},
			expectErr:   true,
			description: "Should error when decode fails, regardless of nil deciders",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(
				defaultDecodeProfile, defaultPrefillProfile, defaultEncodeProfile,
				tt.pdDecider, tt.encodeDecider,
			)

			res, err := h.ProcessResults(context.Background(), nil, &scheduling.InferenceRequest{}, tt.results)
			if tt.expectErr {
				assert.Error(t, err, tt.description)
				return
			}
			assert.NoError(t, err, tt.description)
			assert.Equal(t, defaultDecodeProfile, res.PrimaryProfileName)
			if tt.check != nil {
				tt.check(t, res)
			}
		})
	}
}

func TestHandler_Factory_NilDeciders(t *testing.T) {
	ctx := utils.NewTestContext(t)
	handle := handleWithDeciders(ctx)

	tests := []struct {
		name        string
		params      map[string]any
		expectErr   bool
		description string
	}{
		{
			name: "prefillProfile set, no pdDecider → valid (decider optional)",
			params: map[string]any{
				"profiles": map[string]any{"prefill": "prefill"},
			},
			expectErr:   false,
			description: "Should allow profiles.prefill without deciders.prefill",
		},
		{
			name: "encodeProfile set, no encodeDecider → valid (decider optional)",
			params: map[string]any{
				"profiles": map[string]any{"encode": "encode"},
			},
			expectErr:   false,
			description: "Should allow profiles.encode without deciders.encode",
		},
		{
			name: "both profiles set, no deciders → valid",
			params: map[string]any{
				"profiles": map[string]any{"prefill": "prefill", "encode": "encode"},
			},
			expectErr:   false,
			description: "Should allow both profiles without any deciders",
		},
		{
			name:        "no profiles, no deciders → valid (decode-only)",
			params:      map[string]any{},
			expectErr:   false,
			description: "Should allow decode-only configuration",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, _ := json.Marshal(tt.params)
			p, err := HandlerFactory("h", b, handle)
			if tt.expectErr {
				assert.Error(t, err, tt.description)
				assert.Nil(t, p)
			} else {
				assert.NoError(t, err, tt.description)
				assert.NotNil(t, p)
			}
		})
	}
}

// TestBothProfileAndHeadersHandlerPreRequest verifies that when both
// disagg-profile-handler and the deprecated disagg-headers-handler are
// active, both PreRequest hooks run without error. The result is redundant
// (same header written twice) but not conflicting.
func TestBothProfileAndHeadersHandlerPreRequest(t *testing.T) {
	ctx := utils.NewTestContext(t)

	profileHandler := NewDisaggProfileHandler("decode", "prefill", "encode", nil, nil).WithName("profile")
	headersHandler := NewHeadersHandler("prefill", "encode").WithName("headers") //nolint:staticcheck // intentional: testing deprecated path

	podAddr := "10.0.0.5"
	podPort := "8080"
	ep := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "prefill-pod"},
			Address:        podAddr,
			Port:           podPort,
		},
		&fwkdl.Metrics{},
		nil,
	)

	request := &scheduling.InferenceRequest{
		RequestID: "req-both",
		Headers:   map[string]string{},
	}
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"prefill": {TargetEndpoints: []scheduling.Endpoint{ep}},
		},
	}

	profileHandler.PreRequest(ctx, request, result)
	headersHandler.PreRequest(ctx, request, result)

	expected := net.JoinHostPort(podAddr, podPort)
	assert.Equal(t, expected, request.Headers[routing.PrefillEndpointHeader],
		"both handlers set the same prefill header — redundant but no conflict")
}
