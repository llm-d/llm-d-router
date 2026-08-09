package disagg

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
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
		&fwkdl.EndpointMetadata{ID: nsn, Address: ip, Port: port, Labels: labels},
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

func (p *mockProfile) Run(_ context.Context, _ *scheduling.InferenceRequest, _ []scheduling.Endpoint) (*scheduling.ProfileRunResult, error) {
	return &scheduling.ProfileRunResult{}, nil
}

func profileNames(m map[string]scheduling.SchedulerProfile) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// completionsRequest builds a text-only InferenceRequest. The tokenized prompt
// carries len(prompt)/averageCharactersPerToken token IDs, which the decider reads
// as the input token count.
func completionsRequest(prompt string) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Completions:     &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}},
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{make([]uint32, len(prompt)/averageCharactersPerToken)}},
		},
	}
}

// chatRequest builds a chat-completions InferenceRequest, populating the
// tokenized prompt with one multimodal feature per requested modality so
// that multimodal detection (which reads TokenizedPrompt) is exercised.
func chatRequest(hasImage, hasVideo, hasAudio bool) *scheduling.InferenceRequest {
	blocks := []fwkrh.ContentBlock{{Type: "text", Text: "describe this"}}
	var features []fwkrh.MultiModalFeature
	if hasImage {
		blocks = append(blocks, fwkrh.ContentBlock{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/img.jpg"}})
		features = append(features, fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage})
	}
	if hasVideo {
		blocks = append(blocks, fwkrh.ContentBlock{Type: "video_url"})
		features = append(features, fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage})
	}
	if hasAudio {
		blocks = append(blocks, fwkrh.ContentBlock{Type: "input_audio"})
		features = append(features, fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage})
	}
	body := &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Structured: blocks}}},
		},
	}
	if len(features) > 0 {
		body.TokenizedPrompt = &fwkrh.TokenizedPrompt{MultiModalFeatures: features}
	}
	return &scheduling.InferenceRequest{Body: body}
}

// withPrompt adds a completions body to a chat request and sets the input token
// count (len(prompt)/averageCharactersPerToken) on the tokenized prompt, preserving
// any existing multimodal features.
func withPrompt(req *scheduling.InferenceRequest, prompt string) *scheduling.InferenceRequest {
	req.Body.Completions = &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}}
	if req.Body.TokenizedPrompt == nil {
		req.Body.TokenizedPrompt = &fwkrh.TokenizedPrompt{}
	}
	req.Body.TokenizedPrompt.PerPromptTokens = [][]uint32{make([]uint32, len(prompt)/averageCharactersPerToken)}
	return req
}

// handleWithDeciders creates a plugin handle pre-loaded with all decider types.
func handleWithDeciders(ctx context.Context) plugin.Handle {
	h := plugin.NewEppHandle(ctx, nil, plugin.WithMetricsRecorder(prometheus.NewRegistry()))
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

type mockPDDecider struct {
	allow bool
}

func (m *mockPDDecider) TypedName() plugin.TypedName { return plugin.TypedName{} }

func (m *mockPDDecider) disaggregate(_ context.Context, _ *scheduling.InferenceRequest, _ scheduling.Endpoint) bool {
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
		{"nil tokenized prompt", &scheduling.InferenceRequest{Body: &fwkrh.InferenceRequestBody{}}, false},
		{"empty multimodal features", &scheduling.InferenceRequest{
			Body: &fwkrh.InferenceRequestBody{TokenizedPrompt: &fwkrh.TokenizedPrompt{}},
		}, false},
		{"text only", chatRequest(false, false, false), false},
		{"image", chatRequest(true, false, false), true},
		{"video", chatRequest(false, true, false), true},
		{"audio", chatRequest(false, false, true), true},
		{"feature present", &scheduling.InferenceRequest{
			Body: &fwkrh.InferenceRequestBody{
				TokenizedPrompt: &fwkrh.TokenizedPrompt{
					MultiModalFeatures: []fwkrh.MultiModalFeature{{Modality: fwkrh.ModalityImage}},
				},
			},
		}, true},
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
			p, err := HandlerFactory("h", plugin.StrictDecoder(b), handle)
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
		{"nested format with unknown extra fields is rejected", map[string]any{
			"profiles":     map[string]any{"decode": "decode"},
			"unknownField": "ignored",
		}, true},
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
			p, err := HandlerFactory("h", plugin.StrictDecoder(b), handle)
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
		{"pd-profile-handler with unknown fields is rejected", map[string]any{
			"decodeProfile":     "decode",
			"prefillProfile":    "prefill",
			"deciderPluginName": PrefixBasedPDDeciderPluginType,
			"prefixPluginType":  "prefix-cache-scorer", // unknown to both schemas (#1068)
			"prefixPluginName":  "prefix-cache-scorer",
			"primaryPort":       8080,
		}, true},
		{"pd-profile-handler unknown deciderPluginName", map[string]any{
			"deciderPluginName": "INVALID",
		}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, _ := json.Marshal(tt.params)
			p, err := HandlerFactory("h", plugin.StrictDecoder(b), handle)
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
		p, err := HandlerFactory("h", plugin.StrictDecoder(json.RawMessage(raw)), handle)
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
		name           string
		allow          bool
		profileResults map[string]*scheduling.ProfileRunResult
		want           []string
	}{
		{
			name:           "prefill not run, decider approves → run prefill",
			allow:          true,
			profileResults: map[string]*scheduling.ProfileRunResult{},
			want:           []string{defaultPrefillProfile},
		},
		{
			name:           "prefill not run, decider rejects → skip prefill, run decode",
			allow:          false,
			profileResults: map[string]*scheduling.ProfileRunResult{},
			want:           []string{defaultDecodeProfile},
		},
		{
			name:  "prefill done → run decode",
			allow: true,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
			},
			want: []string{defaultDecodeProfile},
		},
		{
			name:  "prefill failed (nil result) → run decode",
			allow: true,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: nil,
			},
			want: []string{defaultDecodeProfile},
		},
		{
			name:  "all profiles done → done",
			allow: true,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
				defaultDecodeProfile:  makeProfileRunResult("pod2"),
			},
			want: []string{},
		},
		{
			name:  "decode failed → done",
			allow: true,
			profileResults: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
				defaultDecodeProfile:  nil,
			},
			want: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(defaultDecodeProfile, defaultPrefillProfile, "",
				&mockPDDecider{allow: tt.allow}, nil)

			got := h.Pick(ctx, req, profiles, tt.profileResults)
			assert.ElementsMatch(t, tt.want, profileNames(got))
		})
	}
}

func TestHandler_Pick_PD_Series(t *testing.T) {
	ctx := context.Background()
	req := completionsRequest("hello world, hello world!")

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  &mockProfile{},
		defaultPrefillProfile: &mockProfile{},
	}
	tests := []struct {
		name  string
		allow bool
		steps []struct {
			results map[string]*scheduling.ProfileRunResult
			want    []string
		}
	}{
		{
			name:  "decider approves: prefill runs first, then decode, then done",
			allow: true,
			steps: []struct {
				results map[string]*scheduling.ProfileRunResult
				want    []string
			}{
				{map[string]*scheduling.ProfileRunResult{}, []string{defaultPrefillProfile}},
				{map[string]*scheduling.ProfileRunResult{defaultPrefillProfile: makeProfileRunResult("pod1")}, []string{defaultDecodeProfile}},
				{map[string]*scheduling.ProfileRunResult{defaultPrefillProfile: makeProfileRunResult("pod1"), defaultDecodeProfile: makeProfileRunResult("pod2")}, []string{}},
			},
		},
		{
			name:  "decider rejects: prefill skipped, decode runs first, then done",
			allow: false,
			steps: []struct {
				results map[string]*scheduling.ProfileRunResult
				want    []string
			}{
				{map[string]*scheduling.ProfileRunResult{}, []string{defaultDecodeProfile}},
				{map[string]*scheduling.ProfileRunResult{defaultPrefillProfile: nil, defaultDecodeProfile: makeProfileRunResult("pod2")}, []string{}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(defaultDecodeProfile, defaultPrefillProfile, "",
				&mockPDDecider{allow: tt.allow}, nil)

			for _, step := range tt.steps {
				got := h.Pick(ctx, req, profiles, step.results)
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
			res, err := h.ProcessResults(context.Background(), req, tt.results)
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
	_, err := h.ProcessResults(context.Background(), nil, results)
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

	h := NewDisaggProfileHandler(
		customDecodeProfile, customPrefillProfile, customEncodeProfile,
		newAlwaysDisaggPDDecider(), newAlwaysDisaggEncodeDecider(),
	)

	// Stage 1: prefill not run → run prefill
	got := h.Pick(ctx, chatRequest(true, false, false), profiles, map[string]*scheduling.ProfileRunResult{})
	assert.ElementsMatch(t, []string{customPrefillProfile}, profileNames(got))

	// Stage 2: prefill done, multimodal → run encode
	results := map[string]*scheduling.ProfileRunResult{
		customPrefillProfile: makeProfileRunResult("pod1"),
	}
	got = h.Pick(ctx, chatRequest(true, false, false), profiles, results)
	assert.ElementsMatch(t, []string{customEncodeProfile}, profileNames(got))

	// Stage 3: encode done → run decode
	results[customEncodeProfile] = makeProfileRunResult("pod2")
	got = h.Pick(ctx, chatRequest(true, false, false), profiles, results)
	assert.ElementsMatch(t, []string{customDecodeProfile}, profileNames(got))

	// Stage 4: decode done → done
	results[customDecodeProfile] = makeProfileRunResult("pod3")
	got = h.Pick(ctx, chatRequest(true, false, false), profiles, results)
	assert.Empty(t, got)
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
	res, err := h.ProcessResults(context.Background(), req, results)
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
			name:    "encode not run, multimodal → run encode",
			req:     chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{},
			want:    []string{defaultEncodeProfile},
		},
		{
			name:    "no multimodal → skip encode, run decode",
			req:     chatRequest(false, false, false),
			results: map[string]*scheduling.ProfileRunResult{},
			want:    []string{defaultDecodeProfile},
		},
		{
			name:    "image → run encode",
			req:     chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{},
			want:    []string{defaultEncodeProfile},
		},
		{
			name:    "video → run encode",
			req:     chatRequest(false, true, false),
			results: map[string]*scheduling.ProfileRunResult{},
			want:    []string{defaultEncodeProfile},
		},
		{
			name:    "audio → run encode",
			req:     chatRequest(false, false, true),
			results: map[string]*scheduling.ProfileRunResult{},
			want:    []string{defaultEncodeProfile},
		},
		{
			name: "encode done → run decode",
			req:  chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{
				defaultEncodeProfile: makeProfileRunResult("pod1"),
			},
			want: []string{defaultDecodeProfile},
		},
		{
			name: "encode failed → fall through, run decode",
			req:  chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{
				defaultEncodeProfile: nil,
			},
			want: []string{defaultDecodeProfile},
		},
		{
			name: "all profiles done → done",
			req:  chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{
				defaultEncodeProfile: makeProfileRunResult("pod1"),
				defaultDecodeProfile: makeProfileRunResult("pod2"),
			},
			want: []string{},
		},
		{
			name: "decode failed → done",
			req:  chatRequest(true, false, false),
			results: map[string]*scheduling.ProfileRunResult{
				defaultEncodeProfile: makeProfileRunResult("pod1"),
				defaultDecodeProfile: nil,
			},
			want: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(defaultDecodeProfile, "", defaultEncodeProfile, nil, newAlwaysDisaggEncodeDecider())
			got := h.Pick(ctx, tt.req, profiles, tt.results)
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
	results := map[string]*scheduling.ProfileRunResult{}

	tests := []struct {
		name  string
		allow bool
		want  []string
	}{
		{"decider approves → run encode", true, []string{defaultEncodeProfile}},
		{"decider rejects → skip encode, run decode", false, []string{defaultDecodeProfile}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(defaultDecodeProfile, "", defaultEncodeProfile,
				nil, &mockEncodeDecider{allow: tt.allow})
			got := h.Pick(ctx, chatRequest(true, false, false), profiles, results)
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
		{
			name: "encode ran but returned 0 endpoints - included in results",
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile: {TargetEndpoints: []scheduling.Endpoint{}},
			},
			check: func(t *testing.T, res *scheduling.SchedulingResult) {
				assert.Contains(t, res.ProfileResults, defaultDecodeProfile)
				assert.Contains(t, res.ProfileResults, defaultEncodeProfile)
				assert.Empty(t, res.ProfileResults[defaultEncodeProfile].TargetEndpoints)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(defaultDecodeProfile, "", defaultEncodeProfile, nil, newAlwaysDisaggEncodeDecider())
			res, err := h.ProcessResults(context.Background(), &scheduling.InferenceRequest{}, tt.results)
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
		name      string
		req       *scheduling.InferenceRequest
		pdDecider deciderPlugin
		results   map[string]*scheduling.ProfileRunResult
		want      []string
	}{
		{
			name:      "prefill not run → run prefill",
			req:       multimodalLong,
			pdDecider: newAlwaysDisaggPDDecider(),
			results:   map[string]*scheduling.ProfileRunResult{},
			want:      []string{defaultPrefillProfile},
		},
		{
			name:      "prefill done, multimodal → run encode next",
			req:       multimodalLong,
			pdDecider: newAlwaysDisaggPDDecider(),
			results: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
			},
			want: []string{defaultEncodeProfile},
		},
		{
			name:      "prefill done, text-only → skip encode, run decode",
			req:       completionsRequest(testLongPrompt),
			pdDecider: newAlwaysDisaggPDDecider(),
			results: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
			},
			want: []string{defaultDecodeProfile},
		},
		{
			name:      "prefill rejected, multimodal → skip prefill, run encode",
			req:       multimodalLong,
			pdDecider: &mockPDDecider{allow: false},
			results:   map[string]*scheduling.ProfileRunResult{},
			want:      []string{defaultEncodeProfile},
		},
		{
			name:      "prefill rejected, text-only → skip prefill, skip encode, run decode",
			req:       completionsRequest(testLongPrompt),
			pdDecider: &mockPDDecider{allow: false},
			results:   map[string]*scheduling.ProfileRunResult{},
			want:      []string{defaultDecodeProfile},
		},
		{
			name:      "prefill done, encode failed → fall through, run decode",
			req:       multimodalLong,
			pdDecider: newAlwaysDisaggPDDecider(),
			results: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile:  nil,
			},
			want: []string{defaultDecodeProfile},
		},
		{
			name:      "prefill done, encode done → run decode",
			req:       multimodalLong,
			pdDecider: newAlwaysDisaggPDDecider(),
			results: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile:  makeProfileRunResult("pod2"),
			},
			want: []string{defaultDecodeProfile},
		},
		{
			name:      "all three done → done",
			req:       multimodalLong,
			pdDecider: newAlwaysDisaggPDDecider(),
			results: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile:  makeProfileRunResult("pod2"),
				defaultDecodeProfile:  makeProfileRunResult("pod3"),
			},
			want: []string{},
		},
		{
			name:      "decode failed → done",
			req:       multimodalLong,
			pdDecider: newAlwaysDisaggPDDecider(),
			results: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
				defaultEncodeProfile:  makeProfileRunResult("pod2"),
				defaultDecodeProfile:  nil,
			},
			want: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(
				defaultDecodeProfile, defaultPrefillProfile, defaultEncodeProfile,
				tt.pdDecider, newAlwaysDisaggEncodeDecider(),
			)

			got := h.Pick(ctx, tt.req, profiles, tt.results)
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
		wantNext []string // expected next profile from Pick (prefill already run)
	}{
		{"decider approves → run encode next", true, []string{defaultEncodeProfile}},
		{"decider rejects → skip encode, run decode next", false, []string{defaultDecodeProfile}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(
				defaultDecodeProfile, defaultPrefillProfile, defaultEncodeProfile,
				newAlwaysDisaggPDDecider(), &mockEncodeDecider{allow: tt.allow},
			)

			results := map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
			}

			got := h.Pick(ctx, multimodalLong, profiles, results)
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
			res, err := h.ProcessResults(context.Background(), &scheduling.InferenceRequest{}, tt.results)
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
			name:          "both deciders nil, nothing run → skip prefill and encode, run decode",
			pdDecider:     nil,
			encodeDecider: nil,
			req:           multimodalLong,
			results:       map[string]*scheduling.ProfileRunResult{},
			want:          []string{defaultDecodeProfile},
			description:   "With nil deciders, both prefill and encode should be skipped and decode should run",
		},
		{
			name:          "both deciders nil, decode done → done",
			pdDecider:     nil,
			encodeDecider: nil,
			req:           multimodalLong,
			results: map[string]*scheduling.ProfileRunResult{
				defaultDecodeProfile: makeProfileRunResult("pod1"),
			},
			want:        []string{},
			description: "With nil deciders, when decode is done scheduling is complete",
		},
		{
			name:          "pdDecider nil, encodeDecider present, nothing run, multimodal → skip prefill, run encode",
			pdDecider:     nil,
			encodeDecider: newAlwaysDisaggEncodeDecider(),
			req:           multimodalLong,
			results:       map[string]*scheduling.ProfileRunResult{},
			want:          []string{defaultEncodeProfile},
			description:   "Nil pdDecider skips prefill and runs encode for multimodal",
		},
		{
			name:          "pdDecider nil, encodeDecider present, encode done → run decode",
			pdDecider:     nil,
			encodeDecider: newAlwaysDisaggEncodeDecider(),
			req:           multimodalLong,
			results: map[string]*scheduling.ProfileRunResult{
				defaultEncodeProfile: makeProfileRunResult("pod1"),
			},
			want:        []string{defaultDecodeProfile},
			description: "Encode done, runs decode",
		},
		{
			name:          "encodeDecider nil, pdDecider present, nothing run → run prefill",
			pdDecider:     newAlwaysDisaggPDDecider(),
			encodeDecider: nil,
			req:           completionsRequest(testLongPrompt),
			results:       map[string]*scheduling.ProfileRunResult{},
			want:          []string{defaultPrefillProfile},
			description:   "pdDecider present runs prefill",
		},
		{
			name:          "encodeDecider nil, pdDecider present, prefill done → skip encode, run decode",
			pdDecider:     newAlwaysDisaggPDDecider(),
			encodeDecider: nil,
			req:           multimodalLong,
			results: map[string]*scheduling.ProfileRunResult{
				defaultPrefillProfile: makeProfileRunResult("pod1"),
			},
			want:        []string{defaultDecodeProfile},
			description: "Prefill done and encodeDecider nil, runs decode",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewDisaggProfileHandler(
				defaultDecodeProfile, defaultPrefillProfile, defaultEncodeProfile,
				tt.pdDecider, tt.encodeDecider,
			)

			got := h.Pick(ctx, tt.req, profiles, tt.results)
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

			res, err := h.ProcessResults(context.Background(), &scheduling.InferenceRequest{}, tt.results)
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
			p, err := HandlerFactory("h", plugin.StrictDecoder(b), handle)
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
			ID:      k8stypes.NamespacedName{Namespace: "default", Name: "prefill-pod"},
			Address: podAddr,
			Port:    podPort,
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

	_ = profileHandler.PreRequest(ctx, request, result)
	_ = headersHandler.PreRequest(ctx, request, result)

	expected := net.JoinHostPort(podAddr, podPort)
	assert.Equal(t, expected, request.Headers[routing.PrefillEndpointHeader],
		"both handlers set the same prefill header — redundant but no conflict")
}

func TestHandler_PreRequest_EncodeMultipleEndpoints(t *testing.T) {
	ctx := utils.NewTestContext(t)
	h := NewDisaggProfileHandler("decode", "", "encode", nil, nil)

	eps := []scheduling.Endpoint{
		scheduling.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "8000"}, nil, nil),
		scheduling.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.2", Port: "8000"}, nil, nil),
	}
	request := &scheduling.InferenceRequest{Headers: map[string]string{}}
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"encode": {TargetEndpoints: eps},
		},
	}

	_ = h.PreRequest(ctx, request, result)

	want := net.JoinHostPort("10.0.0.1", "8000") + "," + net.JoinHostPort("10.0.0.2", "8000")
	assert.Equal(t, want, request.Headers[routing.EncoderEndpointsHeader])
}

func TestHandler_Pick_PD_StampsPeerEndpointBeforeDecode(t *testing.T) {
	ctx := utils.NewTestContext(t)
	req := completionsRequest(testLongPrompt)

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  &mockProfile{},
		defaultPrefillProfile: &mockProfile{},
	}

	prefillResult := makeProfileRunResult("pod1")
	profileResults := map[string]*scheduling.ProfileRunResult{defaultPrefillProfile: prefillResult}

	h := NewDisaggProfileHandler(defaultDecodeProfile, defaultPrefillProfile, "", newAlwaysDisaggPDDecider(), nil)

	got := h.Pick(ctx, req, profiles, profileResults)
	assert.ElementsMatch(t, []string{defaultDecodeProfile}, profileNames(got), "decode must run")

	peer, ok := scheduling.ReadRequestAttribute[scheduling.Endpoint](req, PeerEndpointAttributeKey)
	assert.True(t, ok, "peer endpoint attribute must be published before decode runs")
	assert.Equal(t, prefillResult.TargetEndpoints[0], peer)
}

func TestHandler_Pick_PD_NoPeerEndpointWhenPrefillSkipped(t *testing.T) {
	ctx := utils.NewTestContext(t)
	req := completionsRequest(testLongPrompt)

	profiles := map[string]scheduling.SchedulerProfile{
		defaultDecodeProfile:  &mockProfile{},
		defaultPrefillProfile: &mockProfile{},
	}

	profileResults := map[string]*scheduling.ProfileRunResult{defaultPrefillProfile: nil}

	h := NewDisaggProfileHandler(defaultDecodeProfile, defaultPrefillProfile, "", &mockPDDecider{allow: false}, nil)

	got := h.Pick(ctx, req, profiles, profileResults)
	assert.ElementsMatch(t, []string{defaultDecodeProfile}, profileNames(got), "decode must run")

	_, ok := scheduling.ReadRequestAttribute[scheduling.Endpoint](req, PeerEndpointAttributeKey)
	assert.False(t, ok, "peer endpoint attribute must not be published when prefill is skipped")
}
