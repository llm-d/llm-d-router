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
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"hash/crc32"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// pngHeaderBase64 returns a 33-byte PNG (signature and IHDR chunk, no image
// data) declaring width x height, base64-encoded. image.DecodeConfig accepts
// it, so the estimator reads the declared size without decoding any pixels.
func pngHeaderBase64(width, height uint32) string {
	var b bytes.Buffer
	b.WriteString("\x89PNG\r\n\x1a\n")
	chunk := make([]byte, 4+13)
	copy(chunk, "IHDR")
	binary.BigEndian.PutUint32(chunk[4:], width)
	binary.BigEndian.PutUint32(chunk[8:], height)
	chunk[12], chunk[13] = 8, 2 // 8-bit truecolor
	_ = binary.Write(&b, binary.BigEndian, uint32(13))
	b.Write(chunk)
	_ = binary.Write(&b, binary.BigEndian, crc32.ChecksumIEEE(chunk))
	return base64.StdEncoding.EncodeToString(b.Bytes())
}

// hugePNG declares 2^30 x 2^29 pixels, about 5.6e14 placeholder tokens under
// the default dynamic image estimator.
var hugePNG = pngHeaderBase64(1<<30, 1<<29)

func chatBody(blocks ...fwkrh.ContentBlock) *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
		Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Structured: blocks}}},
	}}
}

func hugeImageBlock() fwkrh.ContentBlock {
	return fwkrh.ContentBlock{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "data:image/png;base64," + hugePNG}}
}

// TestEstimateBackend_CapsClientDeclaredPlaceholders checks that placeholder
// counts derived from client-declared sizes and durations cannot grow the token
// stream past maxEstimatedPromptTokens, on every API and modality.
func TestEstimateBackend_CapsClientDeclaredPlaceholders(t *testing.T) {
	tests := []struct {
		name string
		ctx  context.Context
		body *fwkrh.InferenceRequestBody
	}{
		{
			name: "chat image header declares a huge size",
			ctx:  context.Background(),
			body: chatBody(fwkrh.ContentBlock{Type: "text", Text: "describe"}, hugeImageBlock()),
		},
		{
			name: "anthropic base64 image declares a huge size",
			ctx:  context.Background(),
			body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
				Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{
					Structured: []fwkrh.AnthropicContentBlock{{Type: "image", Source: imageSource(fwkrh.AnthropicImageSource{
						Type: "base64", MediaType: "image/png", Data: hugePNG,
					})}},
				}}},
			}},
		},
		{
			name: "responses image declares a huge size",
			ctx:  context.Background(),
			body: &fwkrh.InferenceRequestBody{Responses: &fwkrh.ResponsesRequest{
				Input: []any{map[string]any{
					"role":    "user",
					"content": []any{map[string]any{"type": "input_image", "image_url": "data:image/png;base64," + hugePNG}},
				}},
			}},
		},
		{
			name: "video duration header is huge",
			ctx:  videoCtx(videoMetadata{duration: 1e9}),
			body: chatBody(fwkrh.ContentBlock{Type: "video_url", VideoURL: fwkrh.VideoBlock{URL: "https://example.com/v.mp4"}}),
		},
		{
			name: "video resolution header is huge",
			ctx:  videoCtx(videoMetadata{width: 1 << 30, height: 1 << 30}),
			body: chatBody(fwkrh.ContentBlock{Type: "video_url", VideoURL: fwkrh.VideoBlock{URL: "https://example.com/v.mp4"}}),
		},
		{
			name: "audio duration header is huge",
			ctx:  withMMMetadata(context.Background(), mmMetadata{audio: audioMetadata{duration: 1e12}}),
			body: chatBody(fwkrh.ContentBlock{Type: "audio_url", AudioURL: fwkrh.AudioURLBlock{URL: "https://example.com/a.wav"}}),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tp, err := estimateBackend{}.produce(tc.ctx, tc.body)
			require.NoError(t, err)
			require.Len(t, tp.Prompts, 1)
			require.Len(t, tp.Prompts[0].MultiModalFeatures, 1)
			f := tp.Prompts[0].MultiModalFeatures[0]
			assert.Equal(t, maxEstimatedPromptTokens-f.Offset, f.Length, "placeholders must fill exactly the remaining budget")
			assert.Equal(t, maxEstimatedPromptTokens, tp.TokenCount())
		})
	}
}

// TestEstimateBackend_PlaceholderBudgetSpansAssets checks that the budget is
// per request, not per asset: once one asset exhausts it, later assets keep a
// single placeholder each, so many oversized assets cannot multiply the cap.
func TestEstimateBackend_PlaceholderBudgetSpansAssets(t *testing.T) {
	tp, err := estimateBackend{}.produce(context.Background(),
		chatBody(hugeImageBlock(), hugeImageBlock(), hugeImageBlock()))
	require.NoError(t, err)
	features := tp.Prompts[0].MultiModalFeatures
	require.Len(t, features, 3)
	assert.Equal(t, maxEstimatedPromptTokens-features[0].Offset, features[0].Length)
	for _, f := range features[1:] {
		assert.Equal(t, 1, f.Length, "assets past the budget keep one placeholder")
		assert.Equal(t, reqcommon.ModalityImage, f.Modality)
	}
	assert.Equal(t, maxEstimatedPromptTokens+2, tp.TokenCount())
}

func TestAppendMMAsset_ClampsCount(t *testing.T) {
	tests := []struct {
		name   string
		prefix int // tokens already in the stream
		count  int
		want   int
	}{
		{name: "within budget is unchanged", count: 225, want: 225},
		{name: "over budget is clamped", count: maxEstimatedPromptTokens + 1, want: maxEstimatedPromptTokens},
		{name: "clamped to what remains", prefix: 10, count: maxEstimatedPromptTokens, want: maxEstimatedPromptTokens - 10},
		{name: "exhausted budget keeps one", prefix: maxEstimatedPromptTokens, count: 5, want: 1},
		{name: "non-positive count keeps one", count: -7, want: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out := make([]byte, tc.prefix*fwkrh.BytesPerToken)
			out, features := appendMMAsset(out, nil, reqcommon.ModalityImage, "x", tc.count)
			require.Len(t, features, 1)
			assert.Equal(t, tc.want, features[0].Length)
			assert.Equal(t, tc.prefix, features[0].Offset)
			assert.Len(t, out, (tc.prefix+tc.want)*fwkrh.BytesPerToken)
		})
	}
}
