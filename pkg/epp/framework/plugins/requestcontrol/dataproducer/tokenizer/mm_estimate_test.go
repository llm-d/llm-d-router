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
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestRoundHalfEven verifies banker's rounding (round half to even) matches
// Python's round(), which the HF transformers smart_resize uses.
func TestRoundHalfEven(t *testing.T) {
	for _, tc := range []struct {
		n, factor, want int
	}{
		{360, 16, 352},  // 360/16 = 22.5, 22 is even -> round down
		{368, 16, 368},  // exact multiple
		{640, 16, 640},  // exact multiple
		{634, 16, 640},  // 634/16 = 39.625 -> round up to 40*16
		{0, 16, 0},      // zero
		{8, 16, 0},      // 8/16 = 0.5, 0 is even -> round down
		{24, 16, 32},    // 24/16 = 1.5, 1 is odd -> round up to 2*16
	} {
		assert.Equal(t, tc.want, roundHalfEven(tc.n, tc.factor),
			"roundHalfEven(%d, %d)", tc.n, tc.factor)
	}
}

// TestVideoEstimator_SampleFrames verifies the frame-count logic:
// videos with more than maxFrames are capped; shorter ones are resampled.
func TestVideoEstimator_SampleFrames(t *testing.T) {
	e := newVideoEstimator(nil)
	for _, tc := range []struct {
		totalFrames int
		srcFPS      float64
		want        int
	}{
		{2220, 30, 32},  // long video (>32): cap at maxFrames
		{2825, 25, 32},  // long video (>32): cap at maxFrames
		{60, 30, 32},    // 60 > 32: cap at maxFrames
		{32, 30, 4},     // exactly at cap (not >): resample 32/30*2=2 -> clamp to min 4
		{2, 2, 4},       // 2/2*2=2 -> clamped to minVideoFrames
		{1536, 2, 32},   // cap at maxFrames=32
	} {
		assert.Equal(t, tc.want, e.sampleFrames(tc.totalFrames, tc.srcFPS),
			"sampleFrames(%d, %.1f)", tc.totalFrames, tc.srcFPS)
	}
}

// TestVideoEstimator_SmartResize verifies pixel-budget scaling and patch rounding.
func TestVideoEstimator_SmartResize(t *testing.T) {
	e := newVideoEstimator(nil)
	for _, tc := range []struct {
		nframes, h, w, wantH, wantW int
	}{
		// No scaling needed (within budget); only rounding applies.
		{32, 360, 640, 352, 640},  // 360/16=22.5 rounds to 352 (22 even)
		{32, 360, 634, 352, 640},  // 634/16=39.625 rounds up to 640
		{32, 480, 640, 480, 640},  // 480/16=30 exact
		// Scaling required: high-resolution 4K video (3840x2160) with 32 frames.
		// pixPerFrame = 25165824/32 = 786432; 3840*2160 = 8294400 > 786432 -> scale
		// scale = sqrt(786432/8294400) ~= 0.3078; h=2160*0.3078~=665, w=3840*0.3078~=1182
		// roundHalfEven(665,16)=672 (665/16=41.5625->42*16=672), roundHalfEven(1182,16)=1184 (1182/16=73.875->74*16=1184)
		{32, 2160, 3840, 672, 1184},
	} {
		h, w := e.smartResize(tc.nframes, tc.h, tc.w)
		assert.Equal(t, tc.wantH, h, "smartResize(%d,%d,%d) height", tc.nframes, tc.h, tc.w)
		assert.Equal(t, tc.wantW, w, "smartResize(%d,%d,%d) width", tc.nframes, tc.h, tc.w)
	}
}

// TestVideoEstimator_Qwen3VLFormula verifies the full visual-token count for
// the Qwen3-VL formula (frame cap, smart_resize, patch grid, timestamp overhead).
// These values exclude the chat-template prompt overhead, which is request-specific.
func TestVideoEstimator_Qwen3VLFormula(t *testing.T) {
	e := newVideoEstimator(nil)
	for _, tc := range []struct {
		name        string
		totalFrames int
		srcFPS      float64
		h, w        int
		want        int
	}{
		// Long videos capped at 32 frames.
		{"640x360/74s/30fps", 2220, 30, 360, 640, 3662},
		{"640x360/113s/25fps", 2825, 25, 360, 640, 3664},
		// Short video: resampled to sampleFPS, clamped to minVideoFrames.
		{"short/1s/2fps", 2, 2, 360, 640, 456},
	} {
		t.Run(tc.name, func(t *testing.T) {
			nframes := e.sampleFrames(tc.totalFrames, tc.srcFPS)
			hBar, wBar := e.smartResize(nframes, tc.h, tc.w)
			got := e.visualTokens(tc.totalFrames, tc.srcFPS, nframes, hBar, wBar)
			assert.Equal(t, tc.want, got)
		})
	}
}
