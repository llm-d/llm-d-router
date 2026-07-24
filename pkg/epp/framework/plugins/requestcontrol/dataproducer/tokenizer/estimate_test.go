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
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
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
	assert.Equal(t, in, tp.Prompts[0].TokenIDs)
}

// TestEstimateBackend_CompletionsTokenIDsPassthrough asserts token-ID completions
// input is passed through as real tokens, not byte-estimated.
func TestEstimateBackend_CompletionsTokenIDsPassthrough(t *testing.T) {
	in := []uint32{11, 22, 33}
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{TokenIDs: in}},
	})
	require.NoError(t, err)
	assert.Equal(t, in, tp.Prompts[0].TokenIDs, "token IDs must pass through, not be byte-estimated")
}

// TestEstimateBackend_EmbeddingsTokenIDsPassthrough asserts token-ID embeddings
// input is passed through as real tokens.
func TestEstimateBackend_EmbeddingsTokenIDsPassthrough(t *testing.T) {
	in := []uint32{4, 5}
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Embeddings: &fwkrh.EmbeddingsRequest{Input: fwkrh.EmbeddingsInput{TokenIDs: in}},
	})
	require.NoError(t, err)
	assert.Equal(t, in, tp.Prompts[0].TokenIDs)
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
	assert.Equal(t, hashTokens(a.Prompts[0].TokenIDs), hashTokens(b.Prompts[0].TokenIDs), "same prompt produced different tokens")
	c, err := estimateBackend{}.produce(context.Background(), body("hello there"))
	require.NoError(t, err)
	assert.NotEqual(t, hashTokens(a.Prompts[0].TokenIDs), hashTokens(c.Prompts[0].TokenIDs), "distinct prompts produced identical tokens")
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
	require.Len(t, tp.Prompts, 1)
	require.Len(t, tp.Prompts[0].MultiModalFeatures, 1)
	f := tp.Prompts[0].MultiModalFeatures[0]
	assert.Equal(t, fwkrh.ModalityImage, f.Modality)
	assert.Equal(t, strconv.FormatUint(xxhash.Sum64String(pngBase64DataURL), 16), f.Hash)
	assert.Greater(t, f.Length, 1, "image length must be > 1 (placeholder weighting)")
	assert.GreaterOrEqual(t, f.Offset, 0)
	tokens := tp.Prompts[0].TokenIDs
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
			require.Len(t, tp.Prompts[0].MultiModalFeatures, 1)
			require.Equal(t, tc.want, tp.Prompts[0].MultiModalFeatures[0].Modality)
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
	assert.Equal(t, (defaultImageWidth*defaultImageHeight)/imageTokenFactor, def.Prompts[0].MultiModalFeatures[0].Length, "default image length")
	small, err := estimateBackend{}.produce(context.Background(), chat(pngBase64DataURL))
	require.NoError(t, err)
	assert.NotEqual(t, def.Prompts[0].MultiModalFeatures[0].Length, small.Prompts[0].MultiModalFeatures[0].Length, "different images yielded identical placeholder counts")
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
	require.Len(t, tp.Prompts, 1)
	require.Len(t, tp.Prompts[0].MultiModalFeatures, 1)
	assert.Equal(t, 7, tp.Prompts[0].MultiModalFeatures[0].Length, "static image length")
}

// TestImageEstimator_CustomFactor asserts the dynamic factor knob changes the
// placeholder count for the default resolution.
func TestImageEstimator_CustomFactor(t *testing.T) {
	b := estimateBackend{img: newImageEstimator(&estimateConfig{Image: &imageEstimateConfig{Dynamic: &dynamicImageConfig{Factor: 2048}}})}
	// Non-decodable URL falls back to the default 640x360 resolution.
	tp, err := b.produce(context.Background(), chatImageBody("https://example.com/a.png"))
	require.NoError(t, err)
	assert.Equal(t, (defaultImageWidth*defaultImageHeight)/2048, tp.Prompts[0].MultiModalFeatures[0].Length, "custom-factor image length")
}

// TestImageEstimator_CustomDefaultResolution asserts the default-resolution knob
// is used when an image's dimensions cannot be decoded.
func TestImageEstimator_CustomDefaultResolution(t *testing.T) {
	b := estimateBackend{img: newImageEstimator(&estimateConfig{Image: &imageEstimateConfig{
		DefaultResolution: &resolution{Width: 1024, Height: 1024},
	}})}
	tp, err := b.produce(context.Background(), chatImageBody("https://example.com/a.png"))
	require.NoError(t, err)
	assert.Equal(t, (1024*1024)/imageTokenFactor, tp.Prompts[0].MultiModalFeatures[0].Length, "custom default-resolution length")
}

// chatVideoBody builds a chat request carrying a single video_url block.
func chatVideoBody(url string) *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
		Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
			{Type: "video_url", VideoURL: fwkrh.VideoBlock{URL: url}},
		}}}},
	}}
}

// TestVideoEstimator_Default asserts the zero-config estimator uses sampled
// frames (duration*sampleFPS) and dynamic tokens-per-frame (w*h/factor).
func TestVideoEstimator_Default(t *testing.T) {
	tp, err := estimateBackend{}.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	require.Len(t, tp.Prompts[0].MultiModalFeatures, 1)
	frames := defaultVideoDuration * defaultVideoSampleFPS
	tpf := (defaultVideoWidth * defaultVideoHeight) / videoTokenFactor
	assert.Equal(t, frames*tpf, tp.Prompts[0].MultiModalFeatures[0].Length, "default video length")
}

// TestVideoEstimator_StaticTokensPerFrame asserts static mode emits a constant
// per-frame count.
func TestVideoEstimator_StaticTokensPerFrame(t *testing.T) {
	b := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		TokensPerFrame: &tokensPerFrameConfig{Mode: videoTPFModeStatic, Static: &tokensPerFrameStaticMode{NumTokensPerFrame: 100}},
	}})}
	tp, err := b.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	frames := defaultVideoDuration * defaultVideoSampleFPS
	assert.Equal(t, frames*100, tp.Prompts[0].MultiModalFeatures[0].Length, "static tokens-per-frame video length")
}

// TestVideoEstimator_DynamicFactor asserts the dynamic factor knob changes the
// per-frame count for the default resolution.
func TestVideoEstimator_DynamicFactor(t *testing.T) {
	b := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		TokensPerFrame: &tokensPerFrameConfig{Mode: videoTPFModeDynamic, Dynamic: &tokensPerFrameDynamicMode{Factor: 2048}},
	}})}
	tp, err := b.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	frames := defaultVideoDuration * defaultVideoSampleFPS
	tpf := (defaultVideoWidth * defaultVideoHeight) / 2048
	assert.Equal(t, frames*tpf, tp.Prompts[0].MultiModalFeatures[0].Length, "custom-factor video length")
}

// TestVideoEstimator_SampledFrames asserts sampled frames scale with sampleFPS
// and defaultDuration.
func TestVideoEstimator_SampledFrames(t *testing.T) {
	b := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		DefaultDuration: 8,
		Frames:          &framesConfig{Mode: videoFramesModeSampled, Sampled: &framesSampledMode{SampleFPS: 2}},
		TokensPerFrame:  &tokensPerFrameConfig{Mode: videoTPFModeStatic, Static: &tokensPerFrameStaticMode{NumTokensPerFrame: 10}},
	}})}
	tp, err := b.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	assert.Equal(t, 8*2*10, tp.Prompts[0].MultiModalFeatures[0].Length, "sampled-frames video length")
}

// TestVideoEstimator_StridedFramesCapped asserts strided frames apply the
// stride divisor and the maxFrames cap.
func TestVideoEstimator_StridedFramesCapped(t *testing.T) {
	b := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		DefaultDuration: 10,
		Frames:          &framesConfig{Mode: videoFramesModeStrided, MaxFrames: 16, Strided: &framesStridedMode{DefaultSourceFPS: 24, FrameStride: 4}},
		TokensPerFrame:  &tokensPerFrameConfig{Mode: videoTPFModeStatic, Static: &tokensPerFrameStaticMode{NumTokensPerFrame: 100}},
	}})}
	tp, err := b.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	// duration*sourceFPS/stride = 10*24/4 = 60, capped to 16; 16*100 tokens.
	assert.Equal(t, 16*100, tp.Prompts[0].MultiModalFeatures[0].Length, "strided-frames video length")
}

// TestVideoEstimator_StridedFramesFloored asserts strided frames apply the
// minFrames floor when the strided count falls below it.
func TestVideoEstimator_StridedFramesFloored(t *testing.T) {
	b := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		DefaultDuration: 1,
		Frames:          &framesConfig{Mode: videoFramesModeStrided, MinFrames: 8, Strided: &framesStridedMode{DefaultSourceFPS: 24, FrameStride: 4}},
		TokensPerFrame:  &tokensPerFrameConfig{Mode: videoTPFModeStatic, Static: &tokensPerFrameStaticMode{NumTokensPerFrame: 100}},
	}})}
	tp, err := b.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	// duration*sourceFPS/stride = 1*24/4 = 6, floored to 8; 8*100 tokens.
	assert.Equal(t, 8*100, tp.Prompts[0].MultiModalFeatures[0].Length, "strided-frames floored video length")
}

// TestVideoEstimator_MaxVideoTokens asserts the overall cap bounds the total
// placeholder count.
func TestVideoEstimator_MaxVideoTokens(t *testing.T) {
	b := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		MaxVideoTokens: 500,
	}})}
	tp, err := b.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	// Default frames*tpf = 10*225 = 2250, capped to 500.
	assert.Equal(t, 500, tp.Prompts[0].MultiModalFeatures[0].Length, "max-video-tokens cap")
}

// TestVideoEstimator_Qwen3AndGemma4 asserts the two documented model shapes
// compute the expected placeholder counts.
func TestVideoEstimator_Qwen3AndGemma4(t *testing.T) {
	qwen3 := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		DefaultResolution: &resolution{Width: 640, Height: 480},
		DefaultDuration:   10,
		TokensPerFrame:    &tokensPerFrameConfig{Mode: videoTPFModeDynamic, Dynamic: &tokensPerFrameDynamicMode{Factor: 1024}},
		Frames:            &framesConfig{Mode: videoFramesModeSampled, Sampled: &framesSampledMode{SampleFPS: 2}},
		MaxVideoTokens:    100000,
	}})}
	tp, err := qwen3.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	// frames = 10*2 = 20, tpf = 640*480/1024 = 300, tokens = 6000.
	assert.Equal(t, 20*((640*480)/1024), tp.Prompts[0].MultiModalFeatures[0].Length, "qwen3-shaped video length")

	gemma4 := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		DefaultDuration: 10,
		TokensPerFrame:  &tokensPerFrameConfig{Mode: videoTPFModeStatic, Static: &tokensPerFrameStaticMode{NumTokensPerFrame: 256}},
		Frames:          &framesConfig{Mode: videoFramesModeStrided, MaxFrames: 16, Strided: &framesStridedMode{DefaultSourceFPS: 24, FrameStride: 4}},
	}})}
	tp, err = gemma4.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	// frames = min(10*24/4, 16) = 16, tokens = 16*256.
	assert.Equal(t, 16*256, tp.Prompts[0].MultiModalFeatures[0].Length, "gemma4-shaped video length")
}

// TestParseVideoMetadataHeaders covers full, partial, missing, and malformed
// header sets, and the accepted resolution formats.
func TestParseVideoMetadataHeaders(t *testing.T) {
	for _, tc := range []struct {
		name    string
		headers map[string]string
		want    videoMetadata
	}{
		{
			name: "all present",
			headers: map[string]string{
				metadata.VideoFPSHeaderKey:        "30",
				metadata.VideoDurationHeaderKey:   "12.5",
				metadata.VideoResolutionHeaderKey: "1920x1080",
			},
			want: videoMetadata{width: 1920, height: 1080, duration: 12.5, fps: 30},
		},
		{
			name:    "resolution only",
			headers: map[string]string{metadata.VideoResolutionHeaderKey: "640x360"},
			want:    videoMetadata{width: 640, height: 360},
		},
		{
			name:    "uppercase separator",
			headers: map[string]string{metadata.VideoResolutionHeaderKey: "1280X720"},
			want:    videoMetadata{width: 1280, height: 720},
		},
		{
			name:    "spaces trimmed",
			headers: map[string]string{metadata.VideoResolutionHeaderKey: " 800 x 600 "},
			want:    videoMetadata{width: 800, height: 600},
		},
		{
			name:    "empty",
			headers: map[string]string{},
			want:    videoMetadata{},
		},
		{
			name: "malformed values ignored",
			headers: map[string]string{
				metadata.VideoFPSHeaderKey:        "abc",
				metadata.VideoDurationHeaderKey:   "",
				metadata.VideoResolutionHeaderKey: "not-a-res",
			},
			want: videoMetadata{},
		},
		{
			name: "non-positive ignored",
			headers: map[string]string{
				metadata.VideoFPSHeaderKey:        "-1",
				metadata.VideoDurationHeaderKey:   "0",
				metadata.VideoResolutionHeaderKey: "0x0",
			},
			want: videoMetadata{},
		},
		{
			name:    "partial resolution ignored",
			headers: map[string]string{metadata.VideoResolutionHeaderKey: "1920x"},
			want:    videoMetadata{},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, parseVideoMetadataHeaders(tc.headers))
		})
	}
}

// videoCtx returns a context carrying the given video metadata, as the tokenizer
// plugin sets from the x-llm-d-video-* request headers.
func videoCtx(v videoMetadata) context.Context {
	return withMMMetadata(context.Background(), mmMetadata{video: v})
}

// TestMMMetadataContextRoundTrip asserts the metadata survives the context hop and
// that an unset context yields the zero value.
func TestMMMetadataContextRoundTrip(t *testing.T) {
	meta := mmMetadata{video: videoMetadata{width: 1280, height: 720, duration: 4, fps: 25}}
	assert.Equal(t, meta, mmMetadataFromContext(withMMMetadata(context.Background(), meta)))
	assert.Equal(t, mmMetadata{}, mmMetadataFromContext(context.Background()), "unset context must yield the zero value")
}

// TestVideoEstimator_HeaderMetadataOverridesDefaults asserts header-provided
// duration and resolution override the defaults in the zero-config estimator.
func TestVideoEstimator_HeaderMetadataOverridesDefaults(t *testing.T) {
	ctx := videoCtx(videoMetadata{width: 320, height: 240, duration: 3})
	withMeta, err := estimateBackend{}.produce(ctx, chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	// sampled frames = duration(3)*sampleFPS(2) = 6; dynamic tpf = 320*240/1024 = 75.
	assert.Equal(t, 6*((320*240)/videoTokenFactor), withMeta.Prompts[0].MultiModalFeatures[0].Length)

	def, err := estimateBackend{}.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	assert.NotEqual(t, def.Prompts[0].MultiModalFeatures[0].Length, withMeta.Prompts[0].MultiModalFeatures[0].Length, "header metadata must change the count")
}

// TestVideoEstimator_HeaderFPSStridedMode asserts header source FPS and duration
// drive strided frame counting.
func TestVideoEstimator_HeaderFPSStridedMode(t *testing.T) {
	b := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		Frames:         &framesConfig{Mode: videoFramesModeStrided, Strided: &framesStridedMode{FrameStride: 2}},
		TokensPerFrame: &tokensPerFrameConfig{Mode: videoTPFModeStatic, Static: &tokensPerFrameStaticMode{NumTokensPerFrame: 1}},
	}})}
	// strided frames = int(duration(3)*fps(30))/2 = 45.
	tp, err := b.produce(videoCtx(videoMetadata{duration: 3, fps: 30}), chatVideoBody("https://cdn.example.com/movie.mp4"))
	require.NoError(t, err)
	assert.Equal(t, 45, tp.Prompts[0].MultiModalFeatures[0].Length)
}

// TestVideoEstimator_SampledIgnoresHeaderFPS asserts sampled mode honors the
// header duration but not the header source FPS (sampleFPS is authoritative).
func TestVideoEstimator_SampledIgnoresHeaderFPS(t *testing.T) {
	b := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		Frames:         &framesConfig{Mode: videoFramesModeSampled, Sampled: &framesSampledMode{SampleFPS: 2}},
		TokensPerFrame: &tokensPerFrameConfig{Mode: videoTPFModeStatic, Static: &tokensPerFrameStaticMode{NumTokensPerFrame: 1}},
	}})}
	// Same 3s duration, different source fps: both must yield 3*2 = 6.
	fps30, err := b.produce(videoCtx(videoMetadata{duration: 3, fps: 30}), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	fps60, err := b.produce(videoCtx(videoMetadata{duration: 3, fps: 60}), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	assert.Equal(t, 6, fps30.Prompts[0].MultiModalFeatures[0].Length)
	assert.Equal(t, fps30.Prompts[0].MultiModalFeatures[0].Length, fps60.Prompts[0].MultiModalFeatures[0].Length, "sampled mode must ignore header source fps")
}

// TestVideoEstimator_NoHeadersUseDefaults asserts that without header metadata the
// estimator falls back to the built-in defaults.
func TestVideoEstimator_NoHeadersUseDefaults(t *testing.T) {
	tp, err := estimateBackend{}.produce(context.Background(), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	frames := defaultVideoDuration * defaultVideoSampleFPS
	tpf := (defaultVideoWidth * defaultVideoHeight) / videoTokenFactor
	assert.Equal(t, frames*tpf, tp.Prompts[0].MultiModalFeatures[0].Length, "default video length")
}

// TestVideoEstimator_TemporalMergeAndMinFrames asserts sampled frames are floored
// by minFrames and merged by temporalPatchSize (qwen3-vl shape). Static tpf=1
// isolates the frame-group count.
func TestVideoEstimator_TemporalMergeAndMinFrames(t *testing.T) {
	b := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{
		Frames:         &framesConfig{Mode: videoFramesModeSampled, MinFrames: 4, Sampled: &framesSampledMode{SampleFPS: 2, TemporalPatchSize: 2}},
		TokensPerFrame: &tokensPerFrameConfig{Mode: videoTPFModeStatic, Static: &tokensPerFrameStaticMode{NumTokensPerFrame: 1}},
	}})}
	// 10s: 10*2 = 20 sampled frames, /2 temporal merge = 10 groups.
	long, err := b.produce(videoCtx(videoMetadata{duration: 10}), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	assert.Equal(t, 10, long.Prompts[0].MultiModalFeatures[0].Length)
	// 1s: 2 sampled frames floored to minFrames 4, /2 merge = 2 groups.
	short, err := b.produce(videoCtx(videoMetadata{duration: 1}), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	assert.Equal(t, 2, short.Prompts[0].MultiModalFeatures[0].Length)
}

// TestVideoEstimator_HeaderRespectsMaxVideoTokens asserts the overall cap still
// bounds a count derived from header metadata.
func TestVideoEstimator_HeaderRespectsMaxVideoTokens(t *testing.T) {
	b := estimateBackend{vid: newVideoEstimator(&estimateConfig{Video: &videoEstimateConfig{MaxVideoTokens: 100}})}
	// Uncapped would be 6*75 = 450; capped to 100.
	tp, err := b.produce(videoCtx(videoMetadata{width: 320, height: 240, duration: 3}), chatVideoBody("https://example.com/clip.mp4"))
	require.NoError(t, err)
	assert.Equal(t, 100, tp.Prompts[0].MultiModalFeatures[0].Length)
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
	require.Len(t, tp.Prompts, 1)
	require.Len(t, tp.Prompts[0].MultiModalFeatures, 1)
	f := tp.Prompts[0].MultiModalFeatures[0]
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
	require.Len(t, tp.Prompts, 1)
	require.Len(t, tp.Prompts[0].MultiModalFeatures, 1)
	assert.Equal(t, strconv.FormatUint(xxhash.Sum64String(url), 16), tp.Prompts[0].MultiModalFeatures[0].Hash)
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
	assert.Equal(t, hashTokens(a.Prompts[0].TokenIDs), hashTokens(b.Prompts[0].TokenIDs), "identical messages requests produced different tokens")
	c, err := estimateBackend{}.produce(context.Background(), build("you are concise", "hello world"))
	require.NoError(t, err)
	assert.NotEqual(t, hashTokens(a.Prompts[0].TokenIDs), hashTokens(c.Prompts[0].TokenIDs), "different system prompts produced identical tokens")
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
	require.NotEqual(t, hashTokens(a.Prompts[0].TokenIDs), hashTokens(b.Prompts[0].TokenIDs), "streams identical, system was not applied")
	for i := 0; i < sharedTokens; i++ {
		assert.Equal(t, a.Prompts[0].TokenIDs[i], b.Prompts[0].TokenIDs[i], "token %d differs: tools should seed the prefix before system", i)
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
	require.NotEqual(t, hashTokens(a.Prompts[0].TokenIDs), hashTokens(b.Prompts[0].TokenIDs), "streams identical, system was not applied")
	for i := 0; i < sharedTokens; i++ {
		assert.Equal(t, a.Prompts[0].TokenIDs[i], b.Prompts[0].TokenIDs[i], "token %d differs: tools should seed the prefix before system", i)
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
	assert.NotEqual(t, hashTokens(noTools.Prompts[0].TokenIDs), hashTokens(withTools.Prompts[0].TokenIDs), "tools list was ignored by the prefix estimator")
	stock := []any{map[string]any{
		"type":     "function",
		"function": map[string]any{"name": "get_stock_price"},
	}}
	otherTools, err := estimateBackend{}.produce(context.Background(), chat(stock))
	require.NoError(t, err)
	assert.NotEqual(t, hashTokens(withTools.Prompts[0].TokenIDs), hashTokens(otherTools.Prompts[0].TokenIDs), "different tools lists produced identical tokens")
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
	assert.NotEqual(t, hashTokens(noTools.Prompts[0].TokenIDs), hashTokens(withTools.Prompts[0].TokenIDs), "tools list was ignored by the messages prefix estimator")
	stock := []any{map[string]any{
		"name":         "get_stock_price",
		"description":  "Get a stock price",
		"input_schema": map[string]any{"type": "object"},
	}}
	otherTools, err := estimateBackend{}.produce(context.Background(), build(stock))
	require.NoError(t, err)
	assert.NotEqual(t, hashTokens(withTools.Prompts[0].TokenIDs), hashTokens(otherTools.Prompts[0].TokenIDs), "different tools lists produced identical tokens")
}

// TestEstimateBackend_NonChatNoFeatures asserts non-chat protocols carry no
// multimodal features.
func TestEstimateBackend_NonChatNoFeatures(t *testing.T) {
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}},
	})
	require.NoError(t, err)
	assert.Nil(t, tp.Prompts[0].MultiModalFeatures, "non-chat features should be nil")
}

// TestEstimateBackend_MultiStringCompletionsPopulatesPrompts asserts that a
// multi-string completions prompt estimates each string independently and
// populates Prompts.
func TestEstimateBackend_MultiStringCompletionsPopulatesPrompts(t *testing.T) {
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{
			Prompt: fwkrh.Prompt{Strings: []string{"hello world", "foo bar"}},
		},
	})
	if err != nil {
		t.Fatalf("produce: %v", err)
	}
	if len(tp.Prompts) != 2 {
		t.Fatalf("Prompts: got %d prompts, want 2", len(tp.Prompts))
	}
	if tp.TokenCount() != len(tp.Prompts[0].TokenIDs)+len(tp.Prompts[1].TokenIDs) {
		t.Errorf("flat TokenIDs length %d != sum of per-prompt lengths %d+%d",
			tp.TokenCount(), len(tp.Prompts[0].TokenIDs), len(tp.Prompts[1].TokenIDs))
	}
}

// TestEstimateBackend_SingleStringCompletionsSetsPrompts asserts that a
// single-element string array uses a length-1 Prompts slice.
func TestEstimateBackend_SingleStringCompletionsSetsPrompts(t *testing.T) {
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{
			Prompt: fwkrh.Prompt{Strings: []string{"hello world"}},
		},
	})
	if err != nil {
		t.Fatalf("produce: %v", err)
	}
	if len(tp.Prompts) != 1 {
		t.Errorf("single-string prompt should set length-1 Prompts, got %d", len(tp.Prompts))
	}
}
