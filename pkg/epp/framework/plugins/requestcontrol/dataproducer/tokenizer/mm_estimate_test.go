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

	"github.com/stretchr/testify/require"
)

func TestImageDimensionsFromBase64PayloadWebP(t *testing.T) {
	tests := []struct {
		name       string
		payload    []byte
		wantWidth  int
		wantHeight int
	}{
		{
			name:       "VP8X canvas",
			payload:    webpVP8X(1280, 720),
			wantWidth:  1280,
			wantHeight: 720,
		},
		{
			name:       "VP8 lossy frame",
			payload:    webpVP8(640, 480),
			wantWidth:  640,
			wantHeight: 480,
		},
		{
			name:       "VP8L lossless frame",
			payload:    webpVP8L(1024, 768),
			wantWidth:  1024,
			wantHeight: 768,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := base64.StdEncoding.EncodeToString(tt.payload)
			width, height, ok := imageDimensionsFromBase64Payload(raw)
			require.True(t, ok)
			require.Equal(t, tt.wantWidth, width)
			require.Equal(t, tt.wantHeight, height)
		})
	}
}

func TestImageEstimatorWebPPlaceholderCount(t *testing.T) {
	raw := base64.StdEncoding.EncodeToString(webpVP8X(1280, 720))
	count := (imageEstimator{}).placeholderCount("data:image/webp;base64," + raw)
	require.Equal(t, 900, count)
}

func TestImageDimensionsFromBase64PayloadStillDecodesPNG(t *testing.T) {
	width, height, ok := imageDimensionsFromBase64Payload(pngBase64Raw)
	require.True(t, ok)
	require.Equal(t, 64, width)
	require.Equal(t, 32, height)
}

func webpChunk(chunk string, payload []byte) []byte {
	b := make([]byte, 20+len(payload))
	copy(b[0:4], "RIFF")
	binary.LittleEndian.PutUint32(b[4:8], uint32(len(b)-8))
	copy(b[8:12], "WEBP")
	copy(b[12:16], chunk)
	binary.LittleEndian.PutUint32(b[16:20], uint32(len(payload)))
	copy(b[20:], payload)
	return b
}

func webpVP8X(width, height int) []byte {
	payload := make([]byte, 10)
	putUint24LE(payload[4:7], width-1)
	putUint24LE(payload[7:10], height-1)
	return webpChunk("VP8X", payload)
}

func webpVP8(width, height int) []byte {
	payload := make([]byte, 10)
	payload[3], payload[4], payload[5] = 0x9d, 0x01, 0x2a
	binary.LittleEndian.PutUint16(payload[6:8], uint16(width))
	binary.LittleEndian.PutUint16(payload[8:10], uint16(height))
	return webpChunk("VP8 ", payload)
}

func webpVP8L(width, height int) []byte {
	payload := make([]byte, 5)
	payload[0] = 0x2f
	binary.LittleEndian.PutUint32(payload[1:5], uint32(width-1)|uint32(height-1)<<14)
	return webpChunk("VP8L", payload)
}

func putUint24LE(dst []byte, v int) {
	dst[0] = byte(v)
	dst[1] = byte(v >> 8)
	dst[2] = byte(v >> 16)
}
