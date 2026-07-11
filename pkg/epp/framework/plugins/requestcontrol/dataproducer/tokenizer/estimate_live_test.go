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
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// videoMP4DataURLPrefix is the data-URL scheme prefix for a base64 MP4 payload.
const videoMP4DataURLPrefix = "data:video/mp4;base64,"

// resolutionDims maps a lorem.video resolution label to its 16:9 pixel
// dimensions, standing in for what the x-llm-d-video-resolution header carries.
var resolutionDims = map[string][2]int{
	"360p":  {640, 360},
	"720p":  {1280, 720},
	"1080p": {1920, 1080},
}

// videoEstimateEndpointEnv names the env var holding the vLLM endpoint
// (host:port) for the live comparison. Tests are skipped unless it is set.
const videoEstimateEndpointEnv = "VIDEO_ESTIMATE_ENDPOINT"

// videoModelCase describes a model-specific live video estimation setup: the
// served model name and the estimator configuration to compare against the
// server at videoEstimateEndpointEnv.
//
// qwen3-vl scales tokens with resolution (dynamic tokens-per-frame) and samples
// ~2fps merging frame pairs. gemma-4 uses a fixed per-frame cost regardless of
// resolution (static tokens-per-frame) because its SigLIP vision encoder
// condenses every frame to a fixed soft-token count, and strides the source
// frames.
type videoModelCase struct {
	name  string
	model string
	cfg   *estimateConfig
}

// qwen3vlVideoCase configures estimation for a Qwen3-VL server: sampled frames
// at 2fps merged in pairs, dynamic per-frame tokens (width*height / (32*32)).
var qwen3vlVideoCase = videoModelCase{
	name:  "qwen3vl",
	model: "Qwen/Qwen3-VL-30B-A3B-Instruct",
	cfg: &estimateConfig{Video: &videoEstimateConfig{
		Frames:         &framesConfig{Mode: videoFramesModeSampled, SampleFPS: 2, MinFrames: 4, TemporalPatchSize: 2},
		TokensPerFrame: &tokensPerFrameConfig{Mode: videoTPFModeDynamic, Factor: 32 * 32},
		MaxVideoTokens: 12288,
	}},
}

// gemma4VideoCase configures estimation for a Gemma-4 server. The SigLIP vision
// encoder condenses every frame to a fixed 256 soft tokens regardless of
// resolution; adding the ~40 tokens of per-frame formatting (timestamp /
// begin-of-image / end-of-image markers) makes a frame block ~296 tokens. The
// server samples every 4th source frame (frameStride) and caps at 8 frames.
// Source FPS falls back to the built-in default.
var gemma4VideoCase = videoModelCase{
	name:  "gemma4",
	model: "google/gemma-4-31B-it",
	cfg: &estimateConfig{Video: &videoEstimateConfig{
		Frames:         &framesConfig{Mode: videoFramesModeStrided, FrameStride: 4, MaxFrames: 8},
		TokensPerFrame: &tokensPerFrameConfig{Mode: videoTPFModeStatic, StaticToken: 296},
	}},
}

// TestEstimateBackend_QWEN3VL_ProduceTokenCount_Live compares the whole-prompt
// token count from estimateBackend.produce against the server-reported
// prompt_tokens for a "describe the video" + video chat request on a live
// Qwen3-VL server, across a matrix of resolutions and durations.
//
// It is skipped unless VIDEO_ESTIMATE_ENDPOINT (host:port) is set. Videos are
// sourced from lorem.video, then base64-encoded into a data URL before both
// estimation and the API call, so the server and the estimator see identical
// bytes.
//
//	VIDEO_ESTIMATE_ENDPOINT=10.0.0.1:8000 \
//	  go test ./pkg/.../tokenizer/ -run TestEstimateBackend_QWEN3VL_ProduceTokenCount_Live -v
func TestEstimateBackend_QWEN3VL_ProduceTokenCount_Live(t *testing.T) {
	runEstimateBackendProduceLive(t, qwen3vlVideoCase)
}

// TestEstimateBackend_GEMMA4_ProduceTokenCount_Live is the Gemma-4 counterpart of
// TestEstimateBackend_QWEN3VL_ProduceTokenCount_Live, skipped unless
// VIDEO_ESTIMATE_ENDPOINT is set.
//
//	VIDEO_ESTIMATE_ENDPOINT=10.0.0.1:8000 \
//	  go test ./pkg/.../tokenizer/ -run TestEstimateBackend_GEMMA4_ProduceTokenCount_Live -v
func TestEstimateBackend_GEMMA4_ProduceTokenCount_Live(t *testing.T) {
	runEstimateBackendProduceLive(t, gemma4VideoCase)
}

// runEstimateBackendProduceLive runs the resolution/duration matrix for one
// model: for each clip it estimates the whole-prompt token count via
// estimateBackend.produce and logs it against the server-reported prompt_tokens.
func runEstimateBackendProduceLive(t *testing.T, c videoModelCase) {
	endpoint := liveEndpoint(t)

	b := estimateBackend{vid: newVideoEstimator(c.cfg)}
	resolutions := []string{"360p", "720p", "1080p"}
	durations := []int{1, 10, 20, 30, 60, 90}
	client := &http.Client{Timeout: 120 * time.Second}

	t.Logf("%-10s %-6s %10s %10s %8s", "resolution", "dur", "estimate", "actual", "err%")
	for _, res := range resolutions {
		for _, dur := range durations {
			t.Run(fmt.Sprintf("%s/%ds", res, dur), func(t *testing.T) {
				videoURL := fmt.Sprintf("https://lorem.video/%s_h264_%ds", res, dur)
				raw, err := download(context.Background(), client, videoURL)
				if err != nil {
					t.Fatalf("download %s: %v", videoURL, err)
				}
				dataURL := videoMP4DataURLPrefix + base64.StdEncoding.EncodeToString(raw)

				// Match the content the server sees; the video metadata rides the
				// context as the x-llm-d-video-* request headers would supply it.
				dims := resolutionDims[res]
				ctx := withMMMetadata(context.Background(), mmMetadata{video: videoMetadata{width: dims[0], height: dims[1], duration: float64(dur)}})
				body := &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
					Messages: []fwkrh.Message{{
						Role: "user",
						Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
							{Type: "text", Text: "describe the video"},
							{Type: "video_url", VideoURL: fwkrh.VideoBlock{URL: dataURL}},
						}},
					}},
				}}
				tp, err := b.produce(ctx, body)
				if err != nil {
					t.Fatalf("produce: %v", err)
				}
				estimate := tp.TokenCount()

				actual, err := livePromptTokens(context.Background(), client, endpoint, c.model, dataURL)
				if err != nil {
					t.Fatalf("query server: %v", err)
				}

				var errPct float64
				if actual != 0 {
					errPct = float64(estimate-actual) / float64(actual) * 100
				}
				t.Logf("%-10s %-5ds %10d %10d %7.1f%%", res, dur, estimate, actual, errPct)
			})
		}
	}
}

// download fetches url and returns its body.
func download(ctx context.Context, client *http.Client, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// liveEndpoint resolves the vLLM endpoint (host:port) from
// videoEstimateEndpointEnv. The test is skipped unless it is set.
func liveEndpoint(t *testing.T) string {
	t.Helper()
	endpoint := os.Getenv(videoEstimateEndpointEnv)
	if endpoint == "" {
		t.Skipf("set %s (host:port of a vLLM server) to run the live comparison", videoEstimateEndpointEnv)
	}
	return endpoint
}

// livePromptTokens posts a "describe the video" + single-video chat completion
// and returns the server-reported usage.prompt_tokens.
func livePromptTokens(ctx context.Context, client *http.Client, endpoint, model, videoDataURL string) (int, error) {
	// Build the body from a JSON template so the model name and (untrusted) data
	// URL are properly escaped while the request shape stays a single literal.
	modelJSON, err := json.Marshal(model)
	if err != nil {
		return 0, err
	}
	urlJSON, err := json.Marshal(videoDataURL)
	if err != nil {
		return 0, err
	}
	body := fmt.Appendf(nil, `{"model":%s,"messages":[{"role":"user","content":[{"type":"text","text":"describe the video"},{"type":"video_url","video_url":{"url":%s}}]}],"max_tokens":1,"temperature":0}`, modelJSON, urlJSON)

	url := fmt.Sprintf("http://%s/v1/chat/completions", endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("status %d: %s", resp.StatusCode, respBody)
	}

	var parsed struct {
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return 0, fmt.Errorf("decode response: %w", err)
	}
	return parsed.Usage.PromptTokens, nil
}
