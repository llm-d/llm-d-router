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
	"encoding/base64"
	"encoding/binary"
	"errors"
	"image"
	"io"
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

	// Audio estimation modes.
	audioModeDynamic = "dynamic"
	audioModeStatic  = "static"

	// Audio estimation defaults, applied when a property is absent from both the
	// request headers and the config.
	// defaultAudioDuration is the clip length used when neither the
	// x-llm-d-audio-duration-seconds header nor a payload supplies one.
	defaultAudioDuration = 10 // seconds
	// defaultAudioTokensPerSecond models a Whisper-style 50Hz encoder frame rate
	// merged 2:1, the shape the common audio towers share. Per-model rates belong
	// in configuration; this is only the no-config fallback.
	defaultAudioTokensPerSecond = 25
	// defaultAudioBytesPerSecond converts a payload length into seconds for clips
	// that are not PCM WAV, modeling ~128kbps compressed audio.
	defaultAudioBytesPerSecond = 16000
	// wavHeaderBytes bounds how much of a payload is decoded while looking for the
	// WAV format and data chunk headers. Real headers are far smaller; a payload
	// whose data chunk starts past this is treated as unreadable.
	wavHeaderBytes = 1024
	// WAV chunk ids read while resolving a clip's duration.
	wavFmtChunkID  = "fmt "
	wavDataChunkID = "data"
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
	// Image decoding is streamed to reduce memory overhead, since
	// only headers are necessary.
	r := base64.NewDecoder(base64.StdEncoding, strings.NewReader(rawB64))
	cfg, _, err := image.DecodeConfig(r)
	if err != nil || cfg.Width <= 0 || cfg.Height <= 0 {
		return 0, 0, false
	}
	return cfg.Width, cfg.Height, true
}

// mmMetadata carries per-request multimodal properties parsed from the
// x-llm-d-* request headers. Video and audio are populated today; image fields
// follow the same pattern when their headers are added.
type mmMetadata struct {
	video videoMetadata
	audio audioMetadata
}

// audioMetadata carries per-request audio properties parsed from the
// x-llm-d-audio- request headers (metadata.AudioDurationHeaderKey). A zero field
// means "not provided"; the estimator then falls back to the payload,
// configuration, and built-in defaults, in that order.
type audioMetadata struct {
	duration float64 // seconds
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

// audioEstimator estimates an audio clip's placeholder-token count as
// min(duration*tokensPerSecond + fixedOverhead, maxAudioTokens). Audio towers
// convert a clip to encoder frames at a fixed rate and pool them into tokens, so
// the count tracks duration rather than payload size. Duration is resolved per
// clip: a header value wins, then the payload itself, then configuration, then
// the built-in default. The zero value is valid and uses all built-in defaults.
type audioEstimator struct {
	mode            string
	tokensPerSecond float64
	fixedOverhead   int
	bytesPerSecond  int
	defDuration     float64
	maxAudioTokens  int
	staticToken     int
}

// newAudioEstimator resolves an estimateConfig into an audioEstimator, leaving
// unset fields zero so placeholderCount applies built-in defaults.
func newAudioEstimator(cfg *estimateConfig) audioEstimator {
	if cfg == nil || cfg.Audio == nil {
		return audioEstimator{}
	}
	aud := cfg.Audio
	est := audioEstimator{
		mode:           aud.Mode,
		defDuration:    aud.DefaultDuration,
		maxAudioTokens: aud.MaxAudioTokens,
	}
	if aud.Static != nil {
		est.staticToken = aud.Static.StaticToken
	}
	if aud.Dynamic != nil {
		est.tokensPerSecond = aud.Dynamic.TokensPerSecond
		est.fixedOverhead = aud.Dynamic.FixedOverheadTokens
		est.bytesPerSecond = aud.Dynamic.BytesPerSecond
	}
	return est
}

// placeholderCount estimates placeholder tokens for an audio clip. Always >= 1 so
// every clip carries weight.
func (e audioEstimator) placeholderCount(block fwkrh.AudioBlock, meta audioMetadata) int {
	if e.mode == audioModeStatic {
		if e.staticToken > 0 {
			return e.staticToken
		}
		return 1
	}
	rate := e.tokensPerSecond
	if rate <= 0 {
		rate = defaultAudioTokensPerSecond
	}
	tokens := int(e.durationSeconds(block, meta)*rate) + e.fixedOverhead
	if e.maxAudioTokens > 0 && tokens > e.maxAudioTokens {
		tokens = e.maxAudioTokens
	}
	if tokens < 1 {
		tokens = 1
	}
	return tokens
}

// durationSeconds resolves a clip's length in seconds. A header value wins, then
// the payload, which is exact for PCM WAV and payloadBytes/bytesPerSecond for
// everything else, then configuration, then the built-in default. A clip carried
// by reference has no payload and so falls through to configuration.
func (e audioEstimator) durationSeconds(block fwkrh.AudioBlock, meta audioMetadata) float64 {
	if meta.duration > 0 {
		return meta.duration
	}
	if block.Data != "" {
		if seconds, ok := wavDurationFromBase64(block.Data); ok {
			return seconds
		}
		rate := e.bytesPerSecond
		if rate <= 0 {
			rate = defaultAudioBytesPerSecond
		}
		if n := base64DecodedLen(audioBase64Payload(block.Data)); n > 0 {
			return float64(n) / float64(rate)
		}
	}
	if e.defDuration > 0 {
		return e.defDuration
	}
	return defaultAudioDuration
}

// wavDurationFromBase64 returns the length of a base64 PCM WAV payload, its data
// chunk divided by the byte rate its own header declares. ok is false when the
// payload is not a readable WAV, leaving the caller on the byte-rate estimate.
func wavDurationFromBase64(data string) (seconds float64, ok bool) {
	payload := audioBase64Payload(data)
	// Only the headers are needed, so decoding is streamed and bounded.
	head := make([]byte, wavHeaderBytes)
	n, err := io.ReadFull(base64.NewDecoder(base64.StdEncoding, strings.NewReader(payload)), head)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		return 0, false
	}
	head = head[:n]
	if len(head) < 12 || string(head[0:4]) != "RIFF" || string(head[8:12]) != "WAVE" {
		return 0, false
	}
	// Walk the chunk list: a 4-byte id, a 4-byte little-endian size, then a
	// payload padded to an even length. fmt carries the byte rate and always
	// precedes data.
	var byteRate uint32
	for pos := 12; pos+8 <= len(head); {
		id := string(head[pos : pos+4])
		size := int64(binary.LittleEndian.Uint32(head[pos+4 : pos+8]))
		body := int64(pos) + 8
		switch id {
		case wavFmtChunkID:
			if body+12 <= int64(len(head)) {
				byteRate = binary.LittleEndian.Uint32(head[body+8 : body+12])
			}
		case wavDataChunkID:
			if byteRate == 0 {
				return 0, false
			}
			// A streamed WAV can declare a placeholder size, so what the payload
			// actually carries bounds the data chunk.
			available := int64(base64DecodedLen(payload)) - body
			if size <= 0 || size > available {
				size = available
			}
			if size <= 0 {
				return 0, false
			}
			return float64(size) / float64(byteRate), true
		}
		next := body + size
		if size%2 == 1 {
			next++
		}
		if next <= int64(pos) || next > int64(len(head)) {
			return 0, false
		}
		pos = int(next)
	}
	return 0, false
}

// audioBase64Payload strips a "data:...;base64," prefix when present, so a bare
// input_audio payload and a data URL resolve to the same bytes.
func audioBase64Payload(data string) string {
	if !strings.HasPrefix(data, "data:") {
		return data
	}
	if idx := strings.Index(data, "base64,"); idx > 0 {
		return data[idx+len("base64,"):]
	}
	return data
}

// base64DecodedLen returns the decoded byte length of a standard base64 payload
// without decoding it.
func base64DecodedLen(rawB64 string) int {
	n := len(rawB64)
	for n > 0 && rawB64[n-1] == '=' {
		n--
	}
	return n * 3 / 4
}
