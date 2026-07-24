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
	"encoding/base64"
	"image"
	"strings"

	// Registers decoders so image.DecodeConfig can read dimensions.
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const (
	// Image estimation modes.
	imageModeDynamic = "dynamic"
	imageModeStatic  = "static"

	// defaultImageWidth and defaultImageHeight model a 360p image, used when an
	// image URL is not a decodable base64 payload.
	defaultImageWidth  = 640
	defaultImageHeight = 360
	// imageTokenFactor maps image pixels to placeholder tokens (width*height/factor).
	imageTokenFactor = 1024

	// Video tokens-per-frame modes.
	videoTPFModeDynamic = "dynamic"
	videoTPFModeStatic  = "static"
	// Video frame-count modes.
	videoFramesModeSampled = "sampled"
	videoFramesModeStrided = "strided"

	// Video estimation defaults, applied when a property is absent from both the
	// request headers and the config. Duration, resolution, and source FPS come
	// from the x-llm-d-video- request headers when provided; otherwise they fall
	// back to configuration and then these values.
	defaultVideoWidth     = 640
	defaultVideoHeight    = 360
	defaultVideoDuration  = 10 // seconds
	defaultVideoSampleFPS = 2  // sampled frames: duration*sampleFPS
	defaultVideoSourceFPS = 24 // strided frames: duration*sourceFPS/frameStride
	// videoTokenFactor maps a frame's pixels to placeholder tokens (width*height/factor).
	videoTokenFactor = 1024
)

// imageEstimator estimates an image's placeholder-token count from configured or
// default parameters. The zero value is valid and uses all built-in defaults.
type imageEstimator struct {
	mode        string
	defWidth    int
	defHeight   int
	factor      int
	staticToken int
}

// newImageEstimator resolves an estimateConfig into an imageEstimator, leaving
// unset fields zero so placeholderCount applies built-in defaults.
func newImageEstimator(cfg *estimateConfig) imageEstimator {
	if cfg == nil || cfg.Image == nil {
		return imageEstimator{}
	}
	img := cfg.Image
	est := imageEstimator{mode: img.Mode}
	if img.DefaultResolution != nil {
		est.defWidth, est.defHeight = img.DefaultResolution.Width, img.DefaultResolution.Height
	}
	if img.Dynamic != nil {
		est.factor = img.Dynamic.Factor
	}
	if img.Static != nil {
		est.staticToken = img.Static.StaticToken
	}
	return est
}

// placeholderCount estimates placeholder tokens for an image URL. Data URLs
// are decoded for dimensions; other URLs fall back to the default resolution.
func (e imageEstimator) placeholderCount(url string) int {
	w, h, ok := imageDimensionsFromBase64(url)
	return e.countFromDims(w, h, ok)
}

// placeholderForAnthropicImage returns the content (URL or raw base64) and
// placeholder count for an Anthropic image source. Empty content means skip.
func (e imageEstimator) placeholderForAnthropicImage(src *fwkrh.AnthropicImageSource) (content string, count int) {
	if src == nil {
		return "", 0
	}
	if src.URL != "" {
		return src.URL, e.placeholderCount(src.URL)
	}
	if src.Data != "" {
		w, h, ok := imageDimensionsFromBase64Payload(src.Data)
		return src.Data, e.countFromDims(w, h, ok)
	}
	return "", 0
}

// countFromDims returns the token count from decoded dimensions (decoded==true)
// or the configured defaults. Always >= 1 so every image carries weight.
func (e imageEstimator) countFromDims(decW, decH int, decoded bool) int {
	if e.mode == imageModeStatic {
		if e.staticToken > 0 {
			return e.staticToken
		}
		return 1
	}
	w, h := e.defWidth, e.defHeight
	if w <= 0 {
		w = defaultImageWidth
	}
	if h <= 0 {
		h = defaultImageHeight
	}
	if decoded {
		w, h = decW, decH
	}
	factor := e.factor
	if factor <= 0 {
		factor = imageTokenFactor
	}
	if n := (w * h) / factor; n > 0 {
		return n
	}
	return 1
}

// imageDimensionsFromBase64 decodes a data:image/...;base64 URL and returns its
// pixel dimensions. ok is false when the URL is not a decodable base64 image.
func imageDimensionsFromBase64(url string) (width, height int, ok bool) {
	if !strings.HasPrefix(url, "data:image/") || !strings.Contains(url, "base64,") {
		return 0, 0, false
	}
	idx := strings.Index(url, "base64,")
	return imageDimensionsFromBase64Payload(url[idx+len("base64,"):])
}

// imageDimensionsFromBase64Payload decodes a bare base64 image payload and
// returns its pixel dimensions.
func imageDimensionsFromBase64Payload(rawB64 string) (width, height int, ok bool) {
	decoded, err := base64.StdEncoding.DecodeString(rawB64)
	if err != nil {
		return 0, 0, false
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(decoded))
	if err != nil || cfg.Width <= 0 || cfg.Height <= 0 {
		return 0, 0, false
	}
	return cfg.Width, cfg.Height, true
}

// mmMetadata carries per-request multimodal properties parsed from the
// x-llm-d-* request headers. Only video is populated today; image and audio
// fields follow the same pattern when their headers are added.
type mmMetadata struct {
	video videoMetadata
}

// videoMetadata carries per-request video properties parsed from the
// x-llm-d-video- request headers (metadata.VideoFPSHeaderKey and siblings). A
// zero field means "not provided"; the estimator falls back per field to
// configuration and then built-in defaults.
type videoMetadata struct {
	width, height int
	duration      float64 // seconds
	fps           float64 // source frames per second
}

// videoEstimator estimates a video's placeholder-token count as
// min(frames * tokensPerFrame, maxVideoTokens). Frame count and per-frame token
// count are configured independently: qwen3 is sampled frames + dynamic
// tokens-per-frame, gemma4 is strided frames + static tokens-per-frame. Duration,
// resolution, and source FPS come from the request's videoMetadata when provided
// and take precedence over configuration; the config fields are fallbacks. The
// zero value is valid and uses all built-in defaults.
type videoEstimator struct {
	tpfMode     string
	factor      int
	staticToken int

	framesMode        string
	sampleFPS         float64
	sourceFPS         float64
	frameStride       int
	maxFrames         int
	minFrames         int
	temporalPatchSize int

	defWidth       int
	defHeight      int
	defDuration    float64
	maxVideoTokens int
}

// newVideoEstimator resolves an estimateConfig into a videoEstimator, leaving
// unset fields zero so placeholderCount applies built-in defaults.
func newVideoEstimator(cfg *estimateConfig) videoEstimator {
	if cfg == nil || cfg.Video == nil {
		return videoEstimator{}
	}
	vid := cfg.Video
	est := videoEstimator{
		defDuration:    vid.DefaultDuration,
		maxVideoTokens: vid.MaxVideoTokens,
	}
	if vid.DefaultResolution != nil {
		est.defWidth, est.defHeight = vid.DefaultResolution.Width, vid.DefaultResolution.Height
	}
	if vid.TokensPerFrame != nil {
		est.tpfMode = vid.TokensPerFrame.Mode
		if vid.TokensPerFrame.Dynamic != nil {
			est.factor = vid.TokensPerFrame.Dynamic.Factor
		}
		if vid.TokensPerFrame.Static != nil {
			est.staticToken = vid.TokensPerFrame.Static.NumTokensPerFrame
		}
	}
	if vid.Frames != nil {
		est.framesMode = vid.Frames.Mode
		est.minFrames = vid.Frames.MinFrames
		est.maxFrames = vid.Frames.MaxFrames
		if vid.Frames.Sampled != nil {
			est.sampleFPS = vid.Frames.Sampled.SampleFPS
			est.temporalPatchSize = vid.Frames.Sampled.TemporalPatchSize
		}
		if vid.Frames.Strided != nil {
			est.sourceFPS = vid.Frames.Strided.DefaultSourceFPS
			est.frameStride = vid.Frames.Strided.FrameStride
		}
	}
	return est
}

// placeholderCount estimates placeholder tokens for a video from its request
// metadata. Always >= 1 so every video carries weight.
func (e videoEstimator) placeholderCount(meta videoMetadata) int {
	tokens := e.frameCount(meta) * e.tokensPerFrame(meta)
	if e.maxVideoTokens > 0 && tokens > e.maxVideoTokens {
		tokens = e.maxVideoTokens
	}
	if tokens < 1 {
		tokens = 1
	}
	return tokens
}

// frameCount returns the number of frame token-groups. Both modes clamp the raw
// count to [minFrames, maxFrames]. Sampled mode samples duration*sampleFPS
// frames, then merges every temporalPatchSize frames into one group (models e.g.
// qwen3-vl, which samples ~2fps and merges frame pairs). Strided mode takes
// duration*sourceFPS/frameStride. A header-provided duration and source FPS take
// precedence over configuration. sampleFPS is a model sampling rate, not a source
// property, so it is never overridden.
func (e videoEstimator) frameCount(meta videoMetadata) int {
	duration := meta.duration
	if duration <= 0 {
		duration = e.defDuration
	}
	if duration <= 0 {
		duration = defaultVideoDuration
	}
	if e.framesMode == videoFramesModeStrided {
		fps := meta.fps
		if fps <= 0 {
			fps = e.sourceFPS
		}
		if fps <= 0 {
			fps = defaultVideoSourceFPS
		}
		stride := e.frameStride
		if stride < 1 {
			stride = 1
		}
		n := int(duration*fps) / stride
		if e.minFrames > 0 && n < e.minFrames {
			n = e.minFrames
		}
		if e.maxFrames > 0 && n > e.maxFrames {
			n = e.maxFrames
		}
		return n
	}
	fps := e.sampleFPS
	if fps <= 0 {
		fps = defaultVideoSampleFPS
	}
	n := int(duration * fps)
	if e.minFrames > 0 && n < e.minFrames {
		n = e.minFrames
	}
	if e.maxFrames > 0 && n > e.maxFrames {
		n = e.maxFrames
	}
	if e.temporalPatchSize > 1 {
		n /= e.temporalPatchSize
	}
	return n
}

// tokensPerFrame returns the per-frame placeholder count: a fixed constant in
// static mode, or width*height/factor in dynamic mode. A header-provided
// resolution takes precedence over configuration. Always >= 1.
func (e videoEstimator) tokensPerFrame(meta videoMetadata) int {
	if e.tpfMode == videoTPFModeStatic {
		if e.staticToken > 0 {
			return e.staticToken
		}
		return 1
	}
	w, h := e.defWidth, e.defHeight
	if meta.width > 0 && meta.height > 0 {
		w, h = meta.width, meta.height
	}
	if w <= 0 {
		w = defaultVideoWidth
	}
	if h <= 0 {
		h = defaultVideoHeight
	}
	factor := e.factor
	if factor <= 0 {
		factor = videoTokenFactor
	}
	if n := (w * h) / factor; n > 0 {
		return n
	}
	return 1
}
