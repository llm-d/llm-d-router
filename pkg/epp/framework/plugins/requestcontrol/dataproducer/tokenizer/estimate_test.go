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
	"strconv"
	"testing"
	"unsafe"

	"github.com/cespare/xxhash/v2"

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
	if len(raw)%bytesPerToken != 0 {
		t.Fatalf("fixture must be %d-byte aligned, got len %d", bytesPerToken, len(raw))
	}
	tokens := packBytes(raw)
	if got, want := len(tokens), len(raw)/bytesPerToken; got != want {
		t.Fatalf("token count: got %d, want %d", got, want)
	}
	if hashTokens(tokens) != xxhash.Sum64(raw) {
		t.Errorf("packed-token hash != raw-byte hash; estimate path is not key-preserving")
	}
}

// TestEstimateBackend_GeneratePassthrough asserts pre-tokenized input is kept
// as real tokens, not re-estimated.
func TestEstimateBackend_GeneratePassthrough(t *testing.T) {
	in := []uint32{7, 8, 9}
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Generate: &fwkrh.GenerateRequest{TokenIDs: in},
	})
	if err != nil {
		t.Fatalf("produce: %v", err)
	}
	if len(tp.TokenIDs) != len(in) {
		t.Fatalf("got %d tokens, want %d", len(tp.TokenIDs), len(in))
	}
	for i := range in {
		if tp.TokenIDs[i] != in[i] {
			t.Errorf("token %d: got %d, want %d", i, tp.TokenIDs[i], in[i])
		}
	}
}

// TestEstimateBackend_CompletionsDeterministic asserts the same prompt produces
// the same tokens (locality precondition) and that distinct prompts differ.
func TestEstimateBackend_CompletionsDeterministic(t *testing.T) {
	body := func(s string) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: s}}}
	}
	a, err := estimateBackend{}.produce(context.Background(), body("hello world"))
	if err != nil {
		t.Fatalf("produce a: %v", err)
	}
	b, err := estimateBackend{}.produce(context.Background(), body("hello world"))
	if err != nil {
		t.Fatalf("produce b: %v", err)
	}
	if hashTokens(a.TokenIDs) != hashTokens(b.TokenIDs) {
		t.Error("same prompt produced different tokens")
	}
	c, err := estimateBackend{}.produce(context.Background(), body("hello there"))
	if err != nil {
		t.Fatalf("produce c: %v", err)
	}
	if hashTokens(a.TokenIDs) == hashTokens(c.TokenIDs) {
		t.Error("distinct prompts produced identical tokens")
	}
}

// pngBase64DataURL is a 64x32 RGBA PNG, yielding 64*32/imageTokenFactor = 2
// placeholder tokens under the dynamic estimator.
const pngBase64DataURL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAgCAIAAAAt/+nTAAAARUlEQVR4nOzP0QnAUAzDwBSy/8zlTSECdxj/a2fmu7x9d5mAmoCagJqAmoCagJqAmoCagJqAmoCagJqAmoCagNofAAD//57WAN8yR4QZAAAAAElFTkSuQmCC"

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
	if err != nil {
		t.Fatalf("produce: %v", err)
	}
	if len(tp.MultiModalFeatures) != 1 {
		t.Fatalf("got %d features, want 1", len(tp.MultiModalFeatures))
	}
	f := tp.MultiModalFeatures[0]
	if f.Modality != fwkrh.ModalityImage {
		t.Errorf("modality: got %q, want %q", f.Modality, fwkrh.ModalityImage)
	}
	if want := strconv.FormatUint(xxhash.Sum64String(pngBase64DataURL), 16); f.Hash != want {
		t.Errorf("hash: got %q, want %q", f.Hash, want)
	}
	if f.Length <= 1 {
		t.Errorf("image length: got %d, want > 1 (placeholder weighting)", f.Length)
	}
	if f.Offset < 0 || f.Offset+f.Length > len(tp.TokenIDs) {
		t.Errorf("feature span [%d,%d) outside token stream of len %d", f.Offset, f.Offset+f.Length, len(tp.TokenIDs))
	}
	// Placeholder tokens are the URL hash repeated; verify the span carries weight.
	for i := f.Offset; i < f.Offset+f.Length; i++ {
		if tp.TokenIDs[i] != uint32(xxhash.Sum64String(pngBase64DataURL)) {
			t.Errorf("token %d: got %d, want image placeholder token", i, tp.TokenIDs[i])
		}
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
	if err != nil {
		t.Fatalf("produce default: %v", err)
	}
	if got, want := def.MultiModalFeatures[0].Length, (defaultImageWidth*defaultImageHeight)/imageTokenFactor; got != want {
		t.Errorf("default image length: got %d, want %d", got, want)
	}
	small, err := estimateBackend{}.produce(context.Background(), chat(pngBase64DataURL))
	if err != nil {
		t.Fatalf("produce small: %v", err)
	}
	if def.MultiModalFeatures[0].Length == small.MultiModalFeatures[0].Length {
		t.Error("different images yielded identical placeholder counts")
	}
}

// TestEstimateBackend_NonChatNoFeatures asserts non-chat protocols carry no
// multimodal features.
func TestEstimateBackend_NonChatNoFeatures(t *testing.T) {
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}},
	})
	if err != nil {
		t.Fatalf("produce: %v", err)
	}
	if tp.MultiModalFeatures != nil {
		t.Errorf("non-chat features: got %v, want nil", tp.MultiModalFeatures)
	}
}
