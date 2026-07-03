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
	"math"
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

const (
	// Qwen3-VL / Gemma4 model defaults — also match vLLM VideoMediaIO defaults.
	// vLLM never feeds the model more than defaultVideoMaxFrames raw frames: a
	// video decoded to more frames than this is pre-sampled down to exactly
	// defaultVideoMaxFrames (see sampleFrames), so estimation must reproduce
	// that cap rather than the source frame count.
	defaultVideoMaxFrames = 32
	// defaultVideoSampleFPS is the resampling rate applied to videos that decode
	// to defaultVideoMaxFrames or fewer frames: vLLM re-derives the frame count
	// as roughly totalFrames/srcFPS*defaultVideoSampleFPS instead of using the
	// source fps directly (see sampleFrames).
	defaultVideoSampleFPS = 2.0
	// defaultVideoPatchSize is the ViT patch edge in pixels: smartResize rounds
	// each frame's height and width to a multiple of this before computing grids.
	defaultVideoPatchSize = 16
	// defaultVideoMergeSize is the spatial merge window: an NxN block of patches
	// (N = defaultVideoMergeSize) collapses into a single visual token.
	defaultVideoMergeSize = 2
	// defaultVideoTemporalPatch is the temporal merge window: this many
	// consecutive sampled frames collapse into a single time step (grid cell)
	// for both patch counting and timestamp averaging.
	defaultVideoTemporalPatch = 2
	// defaultVideoMaxPixels is the total pixel budget shared across all sampled
	// frames: smartResize divides it by nframes to get a per-frame budget, then
	// downscales height/width to fit before rounding to the patch grid.
	defaultVideoMaxPixels = 25_165_824

	// Fallback values used when MP4 metadata (duration, fps, resolution)
	// cannot be extracted from the payload, so totalFrames = duration*fps and
	// the resolution passed to smartResize both fall back to model-typical values.
	defaultVideoFallbackDuration = 16.0
	defaultVideoFallbackFPS      = 2.0
	defaultVideoFallbackWidth    = 640
	defaultVideoFallbackHeight   = 360

	// minVideoFrames and maxVideoFrames clamp the resampled frame count
	// (see sampleFrames) so extremely short or long videos still produce a
	// sane number of temporal grid cells.
	minVideoFrames = 4
	maxVideoFrames = 768
)

// videoEstimator implements the Qwen3-VL-accurate token estimation:
// vLLM frame cap, smart_resize, temporal patch grouping, and per-grid timestamp tokens.
// It has no tunable fields: every parameter is a Qwen3-VL/vLLM architecture
// constant, not a per-deployment setting.
// Validated against live Qwen3-VL-30B-A3B-Instruct API responses on the
// Video-MME dataset: github.com/llm-d/llm-d-router/pull/1408#issuecomment-4697915077
type videoEstimator struct{}

// newVideoEstimator returns a videoEstimator using built-in constants.
func newVideoEstimator() videoEstimator {
	return videoEstimator{}
}

// placeholderCount estimates placeholder tokens for a video URL. Base64 data URLs
// are decoded for duration, fps, and resolution; other URLs fall back to defaults.
func (e videoEstimator) placeholderCount(url string) int {
	duration, fps, width, height := defaultVideoFallbackDuration, defaultVideoFallbackFPS, defaultVideoFallbackWidth, defaultVideoFallbackHeight

	if strings.HasPrefix(url, "data:video/") && strings.Contains(url, "base64,") {
		idx := strings.Index(url, "base64,")
		decoded, err := base64.StdEncoding.DecodeString(url[idx+len("base64,"):])
		if err == nil {
			if meta, err := parseMP4Metadata(decoded); err == nil {
				duration = meta.Duration
				fps = meta.FPS
				if meta.Width > 0 {
					width = meta.Width
				}
				if meta.Height > 0 {
					height = meta.Height
				}
			}
		}
	}

	totalFrames := int(duration * fps)
	nframes := e.sampleFrames(totalFrames, fps)
	hBar, wBar := e.smartResize(nframes, height, width)
	return e.visualTokens(totalFrames, fps, nframes, hBar, wBar)
}

// sampleFrames mirrors the vLLM VideoMediaIO frame-selection logic:
// videos with more than maxFrames total frames are pre-sampled to exactly
// maxFrames; shorter videos are resampled to sampleFPS and clamped.
func (e videoEstimator) sampleFrames(totalFrames int, srcFPS float64) int {
	if totalFrames > defaultVideoMaxFrames {
		return defaultVideoMaxFrames
	}
	if srcFPS <= 0 {
		srcFPS = defaultVideoFallbackFPS
	}
	n := int(float64(totalFrames) / srcFPS * defaultVideoSampleFPS)
	if n < minVideoFrames {
		return minVideoFrames
	}
	if n > maxVideoFrames {
		return maxVideoFrames
	}
	return n
}

// smartResize scales h and w to fit within the per-frame pixel budget and rounds
// each dimension to the nearest patchSize multiple using banker's rounding
// (round half to even) to match Python's round() and the HF transformers output.
func (e videoEstimator) smartResize(nframes, h, w int) (int, int) {
	pixPerFrame := defaultVideoMaxPixels / nframes
	if pixPerFrame < 1 {
		pixPerFrame = 1
	}
	if h*w > pixPerFrame {
		scale := math.Sqrt(float64(pixPerFrame) / float64(h*w))
		h = int(math.Round(float64(h) * scale))
		w = int(math.Round(float64(w) * scale))
	}
	hBar := roundHalfEven(h, defaultVideoPatchSize)
	wBar := roundHalfEven(w, defaultVideoPatchSize)
	if hBar < defaultVideoPatchSize {
		hBar = defaultVideoPatchSize
	}
	if wBar < defaultVideoPatchSize {
		wBar = defaultVideoPatchSize
	}
	return hBar, wBar
}

// roundHalfEven rounds n to the nearest multiple of factor using banker's
// rounding (round half to nearest even quotient), matching Python's round().
func roundHalfEven(n, factor int) int {
	q := n / factor
	rem := n % factor
	half := factor / 2
	switch {
	case rem < half:
		return q * factor
	case rem > half:
		return (q + 1) * factor
	default: // rem == half: round to nearest even quotient
		if q%2 == 0 {
			return q * factor
		}
		return (q + 1) * factor
	}
}

// visualTokens computes the total visual placeholder count:
// patch tokens + 2 boundary tokens per temporal grid cell + timestamp overhead.
func (e videoEstimator) visualTokens(totalFrames int, srcFPS float64, nframes, hBar, wBar int) int {
	tp := defaultVideoTemporalPatch
	ms := defaultVideoMergeSize
	ps := defaultVideoPatchSize
	tBar := ((nframes + tp - 1) / tp) * tp
	gridT := tBar / tp
	gridH := hBar / ps
	gridW := wBar / ps
	patchTokens := gridT * gridH * gridW / (ms * ms)
	return patchTokens + gridT*2 + e.timestampTokens(totalFrames, srcFPS, nframes, gridT)
}

// timestampTokens computes the "<X.X seconds>" token overhead per temporal grid
// cell. Each timestamp costs 6 tokens plus 1 for two-digit seconds and 1 more
// for three-digit seconds, mirroring the Qwen3-VL chat-template formatting.
func (e videoEstimator) timestampTokens(totalFrames int, srcFPS float64, nframes, gridT int) int {
	timestamps := make([]float64, 0, gridT)
	tp := defaultVideoTemporalPatch
	if totalFrames > defaultVideoMaxFrames {
		// vLLM pre-sampled nframes evenly from [0, totalFrames-1]; average pairs.
		frameIdx := videoLinspace(0, totalFrames-1, nframes)
		rawTS := make([]float64, nframes)
		for i, idx := range frameIdx {
			rawTS[i] = float64(idx) / srcFPS
		}
		for j := 0; j < gridT; j++ {
			base := j * tp
			var sum float64
			for k := 0; k < tp; k++ {
				sum += rawTS[base+k]
			}
			timestamps = append(timestamps, sum/float64(tp))
		}
	} else {
		duration := float64(totalFrames) / srcFPS
		for i := 0; i < gridT; i++ {
			timestamps = append(timestamps, (float64(i)+0.5)*duration/float64(gridT))
		}
	}
	total := 0
	for _, t := range timestamps {
		total += 6
		if t >= 10 {
			total++
		}
		if t >= 100 {
			total++
		}
	}
	return total
}

// videoLinspace returns n evenly spaced integer indices from start to end inclusive,
// matching numpy.linspace(start, end, n, dtype=int).
func videoLinspace(start, end, n int) []int {
	if n <= 0 {
		return nil
	}
	result := make([]int, n)
	if n == 1 {
		result[0] = start
		return result
	}
	for i := range result {
		result[i] = start + int(math.Round(float64(i)*float64(end-start)/float64(n-1)))
	}
	return result
}
