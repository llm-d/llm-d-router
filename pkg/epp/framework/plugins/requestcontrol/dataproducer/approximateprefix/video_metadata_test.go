/*
Copyright 2026 The Kubernetes Authors.

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

package approximateprefix

import (
	"encoding/base64"
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// box builds a box of the given type with the given payload, prefixed with a 32-bit size header.
func box(boxType string, payload []byte) []byte {
	b := make([]byte, 8+len(payload))
	binary.BigEndian.PutUint32(b[0:4], uint32(8+len(payload)))
	copy(b[4:8], boxType)
	copy(b[8:], payload)
	return b
}

// fixedTkhd builds a version-0 tkhd payload long enough to hold the matrix and trailing
// width/height fields, with width and height set as 16.16 fixed-point values.
func fixedTkhd(width, height uint32) []byte {
	payload := make([]byte, 84)
	binary.BigEndian.PutUint32(payload[76:80], width<<16)
	binary.BigEndian.PutUint32(payload[80:84], height<<16)
	return payload
}

// fixedMdhd builds a version-0 mdhd payload with the given timescale and duration.
func fixedMdhd(timescale, duration uint32) []byte {
	payload := make([]byte, 20)
	binary.BigEndian.PutUint32(payload[12:16], timescale)
	binary.BigEndian.PutUint32(payload[16:20], duration)
	return payload
}

// fixedStsz builds an stsz payload with the given sample count and a uniform sample size.
func fixedStsz(sampleCount uint32) []byte {
	payload := make([]byte, 12)
	binary.BigEndian.PutUint32(payload[8:12], sampleCount)
	return payload
}

// concat joins byte slices into a single new slice.
func concat(parts ...[]byte) []byte {
	var out []byte
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// buildMP4 assembles a minimal moov box with a single video track: 5s duration at 1000
// timescale, 150 samples (frames), and a 1280x720 resolution.
func buildMP4(timescale, duration, sampleCount, width, height uint32) []byte {
	hdlr := box("hdlr", concat(make([]byte, 8), []byte("vide")))
	mdhd := box("mdhd", fixedMdhd(timescale, duration))
	stsz := box("stsz", fixedStsz(sampleCount))
	stbl := box("stbl", stsz)
	minf := box("minf", stbl)
	mdia := box("mdia", concat(hdlr, mdhd, minf))
	tkhd := box("tkhd", fixedTkhd(width, height))
	trak := box("trak", concat(tkhd, mdia))
	moov := box("moov", trak)
	return moov
}

func TestParseMP4Metadata(t *testing.T) {
	data := buildMP4(1000, 5000, 150, 1280, 720)

	meta, err := parseMP4Metadata(data)
	require.NoError(t, err)
	assert.Equal(t, 5.0, meta.Duration)
	assert.Equal(t, 30.0, meta.FPS)
	assert.Equal(t, 1280, meta.Width)
	assert.Equal(t, 720, meta.Height)
}

func TestParseMP4Metadata_NoVideoTrack(t *testing.T) {
	hdlr := box("hdlr", concat(make([]byte, 8), []byte("soun")))
	mdia := box("mdia", hdlr)
	trak := box("trak", mdia)
	moov := box("moov", trak)

	_, err := parseMP4Metadata(moov)
	assert.Error(t, err)
}

func TestGetVideoPlaceholders_UsesRealMetadataOverConfig(t *testing.T) {
	data := buildMP4(1000, 5000, 150, 1280, 720)
	encoded := "data:video/mp4;base64," + base64.StdEncoding.EncodeToString(data)

	cfg := &multiModalTokenEstimatorConfig{
		Video: &videoTokenEstimatorConfig{
			// Config defaults are deliberately very different from the real metadata
			// above, to prove the real values are what get used.
			Duration: 100.0,
			FPS:      0.5,
			TokensPerFrame: &imageTokenEstimatorConfig{
				Mode: ModeDynamic,
				DefaultResolution: resolution{
					Width:  16,
					Height: 16,
				},
				DynamicCfg: &dynamicTokenEstimatorConfig{Factor: 32},
			},
		},
	}

	// Real metadata: 5s * 30fps = 150 frames (within the [4, 768] clamp), resolution 1280x720.
	// tokensPerFrame = 1280*720/(32*32*2) = 450. Total = 150 * 450 = 67500.
	got := getVideoPlaceholders(t.Context(), encoded, cfg)
	assert.Equal(t, 67500, got)
}
