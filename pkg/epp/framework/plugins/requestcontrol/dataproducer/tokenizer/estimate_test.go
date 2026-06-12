/*
Copyright 2026 The llm-d Authors.

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

package tokenizer

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"
	"unsafe"

	"github.com/cespare/xxhash/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// hashTokens hashes a token block the way the scorer's HashBlock does: uint32s
// reinterpreted as little-endian bytes.
func hashTokens(t []uint32) uint64 {
	if len(t) == 0 {
		return 0
	}
	return xxhash.Sum64(unsafe.Slice((*byte)(unsafe.Pointer(&t[0])), len(t)*4))
}

// TestPackBytes_KeyPreserving asserts packed-token hashing matches raw-byte
// hashing, so the scorer's cache keys are unchanged.
func TestPackBytes_KeyPreserving(t *testing.T) {
	raw := []byte("the quick brown fox jumps over!!") // len 32, 4-byte aligned
	require.Zero(t, len(raw)%bytesPerToken, "fixture must be %d-byte aligned, got len %d", bytesPerToken, len(raw))
	tokens := packBytes(raw)
	require.Len(t, tokens, len(raw)/bytesPerToken)
	assert.Equal(t, xxhash.Sum64(raw), hashTokens(tokens), "packed-token hash != raw-byte hash; estimate path is not key-preserving")
}

// TestEstimateBackend_GeneratePassthrough asserts pre-tokenized input is kept
// as real tokens, not re-estimated.
func TestEstimateBackend_GeneratePassthrough(t *testing.T) {
	in := []uint32{7, 8, 9}
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Generate: &fwkrh.GenerateRequest{TokenIDs: in},
	})
	require.NoError(t, err)
	assert.Equal(t, in, tp.PerPromptTokens[0])
}

// TestEstimateBackend_CompletionsTokenIDsPassthrough asserts token-ID completions
// input is passed through as real tokens, not byte-estimated.
func TestEstimateBackend_CompletionsTokenIDsPassthrough(t *testing.T) {
	in := []uint32{11, 22, 33}
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{TokenIDs: in}},
	})
	require.NoError(t, err)
	assert.Equal(t, in, tp.PerPromptTokens[0], "token IDs must pass through, not be byte-estimated")
}

// TestEstimateBackend_EmbeddingsTokenIDsPassthrough asserts token-ID embeddings
// input is passed through as real tokens.
func TestEstimateBackend_EmbeddingsTokenIDsPassthrough(t *testing.T) {
	in := []uint32{4, 5}
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Embeddings: &fwkrh.EmbeddingsRequest{Input: fwkrh.EmbeddingsInput{TokenIDs: in}},
	})
	require.NoError(t, err)
	assert.Equal(t, in, tp.PerPromptTokens[0])
}

// TestEstimateBackend_CompletionsDeterministic asserts the same prompt produces
// the same tokens (locality precondition) and that distinct prompts differ.
func TestEstimateBackend_CompletionsDeterministic(t *testing.T) {
	body := func(s string) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: s}}}
	}
	a, err := estimateBackend{}.produce(context.Background(), body("hello world"))
	require.NoError(t, err)
	b, err := estimateBackend{}.produce(context.Background(), body("hello world"))
	require.NoError(t, err)
	assert.Equal(t, hashTokens(a.PerPromptTokens[0]), hashTokens(b.PerPromptTokens[0]), "same prompt produced different tokens")
	c, err := estimateBackend{}.produce(context.Background(), body("hello there"))
	require.NoError(t, err)
	assert.NotEqual(t, hashTokens(a.PerPromptTokens[0]), hashTokens(c.PerPromptTokens[0]), "distinct prompts produced identical tokens")
}

// pngBase64Raw is a 64x32 RGBA PNG (bare base64 payload), yielding
// 64*32/imageTokenFactor = 2 placeholder tokens under the dynamic estimator.
const pngBase64Raw = "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAgCAIAAAAt/+nTAAAARUlEQVR4nOzP0QnAUAzDwBSy/8zlTSECdxj/a2fmu7x9d5mAmoCagJqAmoCagJqAmoCagJqAmoCagJqAmoCagJqAmoCagNofAAD//57WAN8yR4QZAAAAAElFTkSuQmCC"
const pngBase64DataURL = "data:image/png;base64," + pngBase64Raw

// TestEstimateBackend_ChatImageFeature asserts a chat image emits a multimodal
// feature with the image modality and the URL content hash, occupies more than
// one placeholder pseudo-token (weighting), and points within the token stream.
func TestEstimateBackend_ChatImageFeature(t *testing.T) {
	body := &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{{
				Role: "user",
				Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
					{Type: "text", Text: "describe this"},
					{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: pngBase64DataURL}},
				}},
			}},
		},
	}
	tp, err := estimateBackend{}.produce(context.Background(), body)
	require.NoError(t, err)
	require.Len(t, tp.MultiModalFeatures, 1)
	require.Len(t, tp.MultiModalFeatures[0], 1)
	f := tp.MultiModalFeatures[0][0]
	assert.Equal(t, fwkrh.ModalityImage, f.Modality)
	assert.Equal(t, strconv.FormatUint(xxhash.Sum64String(pngBase64DataURL), 16), f.Hash)
	assert.Greater(t, f.Length, 1, "image length must be > 1 (placeholder weighting)")
	assert.GreaterOrEqual(t, f.Offset, 0)
	tokens := tp.PerPromptTokens[0]
	assert.LessOrEqual(t, f.Offset+f.Length, len(tokens), "feature span [%d,%d) outside token stream of len %d", f.Offset, f.Offset+f.Length, len(tokens))
	// Placeholder tokens are the URL hash repeated; verify the span carries weight.
	for i := f.Offset; i < f.Offset+f.Length; i++ {
		assert.Equal(t, uint32(xxhash.Sum64String(pngBase64DataURL)), tokens[i], "token %d: got %d, want image placeholder token", i, tokens[i])
	}
}

func TestEstimateBackend_ChatModalityLabels(t *testing.T) {
	chat := func(block fwkrh.ContentBlock) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{block}}}},
		}}
	}
	for _, tc := range []struct {
		name  string
		block fwkrh.ContentBlock
		want  fwkrh.Modality
	}{
		{"image", fwkrh.ContentBlock{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/a.png"}}, fwkrh.ModalityImage},
		{"audio", fwkrh.ContentBlock{Type: "input_audio", InputAudio: fwkrh.AudioBlock{Data: "AAAA", Format: "wav"}}, fwkrh.ModalityAudio},
		{"video", fwkrh.ContentBlock{Type: "video_url", VideoURL: fwkrh.VideoBlock{URL: "https://example.com/clip.mp4"}}, fwkrh.ModalityVideo},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tp, err := estimateBackend{}.produce(context.Background(), chat(tc.block))
			require.NoError(t, err)
			require.Len(t, tp.MultiModalFeatures, 1)
			require.Equal(t, tc.want, tp.MultiModalFeatures[0].Modality)
		})
	}
}

// TestEstimateBackend_ChatImageWeightingDistinct asserts two images with
// different placeholder counts produce different token streams, so image
// weighting affects locality keys.
func TestEstimateBackend_ChatImageWeightingDistinct(t *testing.T) {
	chat := func(url string) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
				{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: url}},
			}}}},
		}}
	}
	// Non-decodable URL falls back to the default 640x360 resolution.
	def, err := estimateBackend{}.produce(context.Background(), chat("https://example.com/a.png"))
	require.NoError(t, err)
	assert.Equal(t, (defaultImageWidth*defaultImageHeight)/imageTokenFactor, def.MultiModalFeatures[0][0].Length, "default image length")
	small, err := estimateBackend{}.produce(context.Background(), chat(pngBase64DataURL))
	require.NoError(t, err)
	assert.NotEqual(t, def.MultiModalFeatures[0][0].Length, small.MultiModalFeatures[0][0].Length, "different images yielded identical placeholder counts")
}

// chatImageBody builds a chat request carrying a single image_url block.
func chatImageBody(url string) *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
		Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
			{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: url}},
		}}}},
	}}
}

// TestImageEstimator_StaticMode asserts static mode emits a constant placeholder
// count regardless of image dimensions.
func TestImageEstimator_StaticMode(t *testing.T) {
	b := estimateBackend{img: newImageEstimator(&estimateConfig{Image: &imageEstimateConfig{Mode: imageModeStatic, Static: &staticImageConfig{StaticToken: 7}}})}
	tp, err := b.produce(context.Background(), chatImageBody(pngBase64DataURL))
	require.NoError(t, err)
	require.Len(t, tp.MultiModalFeatures, 1)
	require.Len(t, tp.MultiModalFeatures[0], 1)
	assert.Equal(t, 7, tp.MultiModalFeatures[0][0].Length, "static image length")
}

// TestImageEstimator_CustomFactor asserts the dynamic factor knob changes the
// placeholder count for the default resolution.
func TestImageEstimator_CustomFactor(t *testing.T) {
	b := estimateBackend{img: newImageEstimator(&estimateConfig{Image: &imageEstimateConfig{Dynamic: &dynamicImageConfig{Factor: 2048}}})}
	// Non-decodable URL falls back to the default 640x360 resolution.
	tp, err := b.produce(context.Background(), chatImageBody("https://example.com/a.png"))
	require.NoError(t, err)
	assert.Equal(t, (defaultImageWidth*defaultImageHeight)/2048, tp.MultiModalFeatures[0][0].Length, "custom-factor image length")
}

// TestImageEstimator_CustomDefaultResolution asserts the default-resolution knob
// is used when an image's dimensions cannot be decoded.
func TestImageEstimator_CustomDefaultResolution(t *testing.T) {
	b := estimateBackend{img: newImageEstimator(&estimateConfig{Image: &imageEstimateConfig{
		DefaultResolution: &resolution{Width: 1024, Height: 1024},
	}})}
	tp, err := b.produce(context.Background(), chatImageBody("https://example.com/a.png"))
	require.NoError(t, err)
	assert.Equal(t, (1024*1024)/imageTokenFactor, tp.MultiModalFeatures[0][0].Length, "custom default-resolution length")
}

// TestEstimateBackend_MessagesImageFeature asserts an Anthropic messages image
// emits a multimodal feature with image modality, a content-derived hash, and
// span inside the token stream. The base64 source must hash by its raw payload.
func TestEstimateBackend_MessagesImageFeature(t *testing.T) {
	body := &fwkrh.InferenceRequestBody{
		Messages: &fwkrh.MessagesRequest{
			Messages: []fwkrh.AnthropicMessage{{
				Role: "user",
				Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
					{Type: "text", Text: "describe this"},
					{Type: "image", Source: &fwkrh.AnthropicImageSource{
						Type: "base64", MediaType: "image/png", Data: pngBase64Raw,
					}},
				}},
			}},
		},
	}
	tp, err := estimateBackend{}.produce(context.Background(), body)
	require.NoError(t, err)
	require.Len(t, tp.MultiModalFeatures, 1)
	require.Len(t, tp.MultiModalFeatures[0], 1)
	f := tp.MultiModalFeatures[0][0]
	assert.Equal(t, fwkrh.ModalityImage, f.Modality)
	assert.Equal(t, strconv.FormatUint(xxhash.Sum64String(pngBase64Raw), 16), f.Hash, "base64 source must hash by its raw payload")
	assert.Greater(t, f.Length, 1, "image length must be > 1 (placeholder weighting)")
	assert.GreaterOrEqual(t, f.Offset, 0)
	assert.LessOrEqual(t, f.Offset+f.Length, tp.TokenCount(), "feature span [%d,%d) outside token stream of len %d", f.Offset, f.Offset+f.Length, tp.TokenCount())
}

// TestEstimateBackend_MessagesURLImageKey asserts a url-typed source is hashed
// by its URL unchanged (no synthesized data-URL prefix).
func TestEstimateBackend_MessagesURLImageKey(t *testing.T) {
	const url = "https://example.com/a.png"
	body := &fwkrh.InferenceRequestBody{
		Messages: &fwkrh.MessagesRequest{
			Messages: []fwkrh.AnthropicMessage{{
				Role: "user",
				Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
					{Type: "image", Source: &fwkrh.AnthropicImageSource{Type: "url", URL: url}},
				}},
			}},
		},
	}
	tp, err := estimateBackend{}.produce(context.Background(), body)
	require.NoError(t, err)
	require.Len(t, tp.MultiModalFeatures, 1)
	require.Len(t, tp.MultiModalFeatures[0], 1)
	assert.Equal(t, strconv.FormatUint(xxhash.Sum64String(url), 16), tp.MultiModalFeatures[0][0].Hash)
}

// TestEstimateBackend_MessagesDeterministic asserts identical requests produce
// identical tokens and that changing the system prompt changes the stream.
// CacheSalt is intentionally NOT tested -- the approximateprefix layer mixes it
// into the seed, not this estimator.
func TestEstimateBackend_MessagesDeterministic(t *testing.T) {
	build := func(system, userText string) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
			System: fwkrh.AnthropicContent{Raw: system},
			Messages: []fwkrh.AnthropicMessage{
				{Role: "user", Content: fwkrh.AnthropicContent{Raw: userText}},
			},
		}}
	}
	a, err := estimateBackend{}.produce(context.Background(), build("you are helpful", "hello world"))
	require.NoError(t, err)
	b, err := estimateBackend{}.produce(context.Background(), build("you are helpful", "hello world"))
	require.NoError(t, err)
	assert.Equal(t, hashTokens(a.PerPromptTokens[0]), hashTokens(b.PerPromptTokens[0]), "identical messages requests produced different tokens")
	c, err := estimateBackend{}.produce(context.Background(), build("you are concise", "hello world"))
	require.NoError(t, err)
	assert.NotEqual(t, hashTokens(a.PerPromptTokens[0]), hashTokens(c.PerPromptTokens[0]), "different system prompts produced identical tokens")
}

// TestEstimateBackend_ChatToolsBeforeSystem asserts the tools list is emitted
// before the system message, so requests sharing tools but differing in system
// share their leading tokens.
func TestEstimateBackend_ChatToolsBeforeSystem(t *testing.T) {
	tools := []any{map[string]any{"name": "search_for_long_enough_byte_segment_for_this_ordering_test"}}
	toolsJSON, err := json.Marshal(tools)
	require.NoError(t, err)
	// -1 skips the token straddling the tools/system byte boundary.
	sharedTokens := len(toolsJSON)/bytesPerToken - 1
	chat := func(systemContent string) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{
				{Role: "system", Content: fwkrh.Content{Raw: systemContent}},
				{Role: "user", Content: fwkrh.Content{Raw: "hi"}},
			},
			Tools: tools,
		}}
	}
	a, err := estimateBackend{}.produce(context.Background(), chat("you are a helpful assistant"))
	require.NoError(t, err)
	b, err := estimateBackend{}.produce(context.Background(), chat("you are a concise assistant"))
	require.NoError(t, err)
	require.NotEqual(t, hashTokens(a.PerPromptTokens[0]), hashTokens(b.PerPromptTokens[0]), "streams identical, system was not applied")
	for i := 0; i < sharedTokens; i++ {
		assert.Equal(t, a.PerPromptTokens[0][i], b.PerPromptTokens[0][i], "token %d differs: tools should seed the prefix before system", i)
	}
}

// TestEstimateBackend_MessagesToolsBeforeSystem is the /v1/messages analog of
// TestEstimateBackend_ChatToolsBeforeSystem.
func TestEstimateBackend_MessagesToolsBeforeSystem(t *testing.T) {
	tools := []any{map[string]any{
		"name":         "search_for_long_enough_byte_segment_for_this_ordering_test",
		"description":  "ensures stable byte length",
		"input_schema": map[string]any{"type": "object"},
	}}
	toolsJSON, err := json.Marshal(tools)
	require.NoError(t, err)
	// -1 skips the token straddling the tools/system byte boundary.
	sharedTokens := len(toolsJSON)/bytesPerToken - 1
	build := func(systemContent string) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
			System: fwkrh.AnthropicContent{Raw: systemContent},
			Messages: []fwkrh.AnthropicMessage{
				{Role: "user", Content: fwkrh.AnthropicContent{Raw: "hi"}},
			},
			Tools: tools,
		}}
	}
	a, err := estimateBackend{}.produce(context.Background(), build("you are a helpful assistant"))
	require.NoError(t, err)
	b, err := estimateBackend{}.produce(context.Background(), build("you are a concise assistant"))
	require.NoError(t, err)
	require.NotEqual(t, hashTokens(a.PerPromptTokens[0]), hashTokens(b.PerPromptTokens[0]), "streams identical, system was not applied")
	for i := 0; i < sharedTokens; i++ {
		assert.Equal(t, a.PerPromptTokens[0][i], b.PerPromptTokens[0][i], "token %d differs: tools should seed the prefix before system", i)
	}
}

// TestEstimateBackend_ChatToolsAffectPrefix asserts the tools list participates
// in the prefix stream so distinct tool sets do not collide on the same key.
func TestEstimateBackend_ChatToolsAffectPrefix(t *testing.T) {
	chat := func(tools []any) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hello world"}}},
			Tools:    tools,
		}}
	}
	noTools, err := estimateBackend{}.produce(context.Background(), chat(nil))
	require.NoError(t, err)
	weather := []any{map[string]any{
		"type":     "function",
		"function": map[string]any{"name": "get_weather"},
	}}
	withTools, err := estimateBackend{}.produce(context.Background(), chat(weather))
	require.NoError(t, err)
	assert.NotEqual(t, hashTokens(noTools.PerPromptTokens[0]), hashTokens(withTools.PerPromptTokens[0]), "tools list was ignored by the prefix estimator")
	stock := []any{map[string]any{
		"type":     "function",
		"function": map[string]any{"name": "get_stock_price"},
	}}
	otherTools, err := estimateBackend{}.produce(context.Background(), chat(stock))
	require.NoError(t, err)
	assert.NotEqual(t, hashTokens(withTools.PerPromptTokens[0]), hashTokens(otherTools.PerPromptTokens[0]), "different tools lists produced identical tokens")
}

// TestEstimateBackend_MessagesToolsAffectPrefix is the /v1/messages analog of
// TestEstimateBackend_ChatToolsAffectPrefix.
func TestEstimateBackend_MessagesToolsAffectPrefix(t *testing.T) {
	build := func(tools []any) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
			Messages: []fwkrh.AnthropicMessage{
				{Role: "user", Content: fwkrh.AnthropicContent{Raw: "hello world"}},
			},
			Tools: tools,
		}}
	}
	noTools, err := estimateBackend{}.produce(context.Background(), build(nil))
	require.NoError(t, err)
	weather := []any{map[string]any{
		"name":         "get_weather",
		"description":  "Get the current weather",
		"input_schema": map[string]any{"type": "object"},
	}}
	withTools, err := estimateBackend{}.produce(context.Background(), build(weather))
	require.NoError(t, err)
	assert.NotEqual(t, hashTokens(noTools.PerPromptTokens[0]), hashTokens(withTools.PerPromptTokens[0]), "tools list was ignored by the messages prefix estimator")
	stock := []any{map[string]any{
		"name":         "get_stock_price",
		"description":  "Get a stock price",
		"input_schema": map[string]any{"type": "object"},
	}}
	otherTools, err := estimateBackend{}.produce(context.Background(), build(stock))
	require.NoError(t, err)
	assert.NotEqual(t, hashTokens(withTools.PerPromptTokens[0]), hashTokens(otherTools.PerPromptTokens[0]), "different tools lists produced identical tokens")
}

// TestEstimateBackend_NonChatNoFeatures asserts non-chat protocols carry no
// multimodal features.
func TestEstimateBackend_NonChatNoFeatures(t *testing.T) {
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}},
	})
	require.NoError(t, err)
	assert.Nil(t, tp.MultiModalFeatures, "non-chat features should be nil")
}

// TestEstimateBackend_MultiStringCompletionsPopulatesPerPromptTokens asserts
// that a multi-string completions prompt estimates each string independently
// and populates PerPromptTokens.
func TestEstimateBackend_MultiStringCompletionsPopulatesPerPromptTokens(t *testing.T) {
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{
			Prompt: fwkrh.Prompt{Strings: []string{"hello world", "foo bar"}},
		},
	})
	if err != nil {
		t.Fatalf("produce: %v", err)
	}
	if len(tp.PerPromptTokens) != 2 {
		t.Fatalf("PerPromptTokens: got %d prompts, want 2", len(tp.PerPromptTokens))
	}
	if tp.TokenCount() != len(tp.PerPromptTokens[0])+len(tp.PerPromptTokens[1]) {
		t.Errorf("flat TokenIDs length %d != sum of per-prompt lengths %d+%d",
			tp.TokenCount(), len(tp.PerPromptTokens[0]), len(tp.PerPromptTokens[1]))
	}
}

// TestEstimateBackend_SingleStringCompletionsSetsPerPromptTokens asserts that a
// single-element string array uses a length-1 PerPromptTokens slice.
func TestEstimateBackend_SingleStringCompletionsSetsPerPromptTokens(t *testing.T) {
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{
			Prompt: fwkrh.Prompt{Strings: []string{"hello world"}},
		},
	})
	if err != nil {
		t.Fatalf("produce: %v", err)
	}
	if len(tp.PerPromptTokens) != 1 {
		t.Errorf("single-string prompt should set length-1 PerPromptTokens, got %d", len(tp.PerPromptTokens))
	}
}
