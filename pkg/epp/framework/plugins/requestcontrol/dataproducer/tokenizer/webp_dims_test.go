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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// webpHeader builds a minimal RIFF/WEBP header of the given format carrying
// the given pixel size. Only the size fields are meaningful; no encoder
// produced this.
func webpHeader(format string, width, height int) []byte {
	var payload []byte
	switch format {
	case "VP8X":
		payload = make([]byte, 10)
		putUint24(payload[4:7], width-1)
		putUint24(payload[7:10], height-1)
	case "VP8 ":
		payload = make([]byte, 10)
		copy(payload[3:6], []byte{0x9d, 0x01, 0x2a})
		binary.LittleEndian.PutUint16(payload[6:8], uint16(width))
		binary.LittleEndian.PutUint16(payload[8:10], uint16(height))
	case "VP8L":
		payload = make([]byte, 5)
		payload[0] = 0x2f
		binary.LittleEndian.PutUint32(payload[1:5],
			uint32(width-1)|uint32(height-1)<<14)
	}
	hdr := make([]byte, 0, 20+len(payload))
	hdr = append(hdr, 'R', 'I', 'F', 'F')
	hdr = binary.LittleEndian.AppendUint32(hdr, uint32(4+len(payload)))
	hdr = append(hdr, 'W', 'E', 'B', 'P')
	hdr = append(hdr, format...)
	hdr = binary.LittleEndian.AppendUint32(hdr, uint32(len(payload)))
	return append(hdr, payload...)
}

// putUint24 writes v as a 3-byte little-endian value, the RIFF container's
// encoding for 24-bit fields such as the VP8X canvas size.
func putUint24(b []byte, v int) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
}

func TestWebpDimensions(t *testing.T) {
	tests := []struct {
		name          string
		format        string
		width, height int
	}{
		{"extended", "VP8X", 4033, 2047},
		{"lossy", "VP8 ", 1920, 1080},
		{"lossless", "VP8L", 750, 500},
		{"lossless-one-pixel", "VP8L", 1, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			w, h, ok := webpDimensions(webpHeader(tc.format, tc.width, tc.height))
			require.True(t, ok)
			assert.Equal(t, tc.width, w)
			assert.Equal(t, tc.height, h)
		})
	}
}

func TestWebpDimensionsRejects(t *testing.T) {
	badVP8 := webpHeader("VP8 ", 10, 10)
	badVP8[23] = 0x00 // corrupt the start code
	badVP8L := webpHeader("VP8L", 10, 10)
	badVP8L[20] = 0x00 // corrupt the signature byte
	unknownChunk := webpHeader("VP8X", 10, 10)
	copy(unknownChunk[12:16], "ANIM") // dims only live in ANMF chunks, not here

	for name, hdr := range map[string][]byte{
		"too-short":         webpHeader("VP8X", 10, 10)[:11],
		"truncated-payload": webpHeader("VP8X", 10, 10)[:24],
		"not-riff":          append([]byte("GIF8"), webpHeader("VP8X", 10, 10)[4:]...),
		"not-webp-container": func() []byte {
			h := webpHeader("VP8X", 10, 10)
			copy(h[8:12], "WAVE")
			return h
		}(),
		"unknown-chunk":      unknownChunk,
		"bad-vp8-start-code": badVP8,
		"bad-vp8l-signature": badVP8L,
	} {
		t.Run(name, func(t *testing.T) {
			w, h, ok := webpDimensions(hdr)
			assert.False(t, ok)
			assert.Zero(t, w)
			assert.Zero(t, h)
		})
	}
}

// TestWebpDimensionsFromBase64Payload asserts a data:image/webp payload is
// measured at its real size instead of falling back to the default resolution.
func TestWebpDimensionsFromBase64Payload(t *testing.T) {
	raw := base64.StdEncoding.EncodeToString(webpHeader("VP8X", 2048, 2048))
	w, h, ok := imageDimensionsFromBase64Payload(raw)
	require.True(t, ok)
	assert.Equal(t, 2048, w)
	assert.Equal(t, 2048, h)

	est := imageEstimator{} // dynamic mode, all defaults
	assert.Equal(t, 2048*2048/imageTokenFactor, est.placeholderCount("data:image/webp;base64,"+raw),
		"a webp payload must estimate at its real size, not the 640x360 default")

	// Registered formats keep their decoder path untouched.
	pw, ph, pok := imageDimensionsFromBase64Payload(pngBase64Raw)
	require.True(t, pok)
	assert.Equal(t, 64, pw)
	assert.Equal(t, 32, ph)
}
