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
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseMP4Metadata(t *testing.T) {
	data := buildTestMP4(t, 10.0, 25.0, 640, 360)

	meta, err := parseMP4Metadata(data)

	assert.NoError(t, err)
	assert.Equal(t, 10.0, meta.Duration)
	assert.Equal(t, 25.0, meta.FPS)
	assert.Equal(t, 640, meta.Width)
	assert.Equal(t, 360, meta.Height)
}

func TestParseMP4Metadata_NotAnMP4(t *testing.T) {
	_, err := parseMP4Metadata([]byte("not an mp4 file"))
	assert.Error(t, err)
}

// buildTestMP4 builds the minimal box structure parseMP4Metadata reads:
// moov > trak > (tkhd, mdia > (hdlr, mdhd, minf > stbl > stsz)).
func buildTestMP4(t *testing.T, durationSec, fps float64, width, height int) []byte {
	t.Helper()
	const timescale = 1000

	mdhd := make([]byte, 20) // version/flags + times(8) + timescale(4) + duration(4)
	binary.BigEndian.PutUint32(mdhd[12:16], timescale)
	binary.BigEndian.PutUint32(mdhd[16:20], uint32(durationSec*timescale))

	hdlr := make([]byte, 12) // version/flags(4) + pre_defined(4) + handler_type(4)
	copy(hdlr[8:12], "vide")

	stsz := make([]byte, 12) // version/flags(4) + sample_size(4) + sample_count(4)
	binary.BigEndian.PutUint32(stsz[8:12], uint32(durationSec*fps))

	tkhd := make([]byte, 8) // width, height as 16.16 fixed-point
	binary.BigEndian.PutUint32(tkhd[0:4], uint32(width)<<16)
	binary.BigEndian.PutUint32(tkhd[4:8], uint32(height)<<16)

	mdia := concatBoxes(mp4Box("hdlr", hdlr), mp4Box("mdhd", mdhd), mp4Box("minf", mp4Box("stbl", mp4Box("stsz", stsz))))
	trak := concatBoxes(mp4Box("tkhd", tkhd), mp4Box("mdia", mdia))
	return mp4Box("moov", mp4Box("trak", trak))
}

func mp4Box(boxType string, payload []byte) []byte {
	buf := make([]byte, 8+len(payload))
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(buf)))
	copy(buf[4:8], boxType)
	copy(buf[8:], payload)
	return buf
}

func concatBoxes(boxes ...[]byte) []byte {
	total := 0
	for _, b := range boxes {
		total += len(b)
	}
	out := make([]byte, 0, total)
	for _, b := range boxes {
		out = append(out, b...)
	}
	return out
}
