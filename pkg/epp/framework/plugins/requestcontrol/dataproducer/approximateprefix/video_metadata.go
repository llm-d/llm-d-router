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
	"encoding/binary"
	"errors"
)

// videoMetadata holds the metadata read directly from a video container, as opposed to
// configured fallback values.
type videoMetadata struct {
	Duration float64 // seconds
	FPS      float64
	Width    int
	Height   int
}

// parseMP4Metadata reads duration, fps, and resolution from the first video track of an
// ISO base media file (MP4) by walking its box structure. It reads only the box headers and
// the small fixed-size boxes under moov/trak/mdia, not the sample data itself.
func parseMP4Metadata(data []byte) (*videoMetadata, error) {
	moov, ok := findBox(data, "moov")
	if !ok {
		return nil, errors.New("moov box not found")
	}

	var meta *videoMetadata
	var trakErr error
	iterateBoxes(moov, func(boxType string, payload []byte) bool {
		if boxType != "trak" {
			return true
		}
		m, err := parseVideoTrak(payload)
		if err != nil {
			trakErr = err
			return true
		}
		meta = m
		return false
	})
	if meta == nil {
		if trakErr == nil {
			trakErr = errors.New("no video track found")
		}
		return nil, trakErr
	}
	return meta, nil
}

// parseVideoTrak parses a single trak box, returning an error if it is not a video track or
// its timing/sample metadata is missing or malformed.
func parseVideoTrak(trak []byte) (*videoMetadata, error) {
	mdia, ok := findBox(trak, "mdia")
	if !ok {
		return nil, errors.New("mdia box not found")
	}
	hdlr, ok := findBox(mdia, "hdlr")
	if !ok || len(hdlr) < 12 || string(hdlr[8:12]) != "vide" {
		return nil, errors.New("not a video track")
	}
	mdhd, ok := findBox(mdia, "mdhd")
	if !ok {
		return nil, errors.New("mdhd box not found")
	}
	timescale, duration, err := parseMdhd(mdhd)
	if err != nil {
		return nil, err
	}
	minf, ok := findBox(mdia, "minf")
	if !ok {
		return nil, errors.New("minf box not found")
	}
	stbl, ok := findBox(minf, "stbl")
	if !ok {
		return nil, errors.New("stbl box not found")
	}
	stsz, ok := findBox(stbl, "stsz")
	if !ok {
		return nil, errors.New("stsz box not found")
	}
	sampleCount, err := parseStszSampleCount(stsz)
	if err != nil {
		return nil, err
	}
	if timescale == 0 || duration == 0 || sampleCount == 0 {
		return nil, errors.New("invalid mp4 timing metadata")
	}

	durationSec := float64(duration) / float64(timescale)
	width, height := 0, 0
	if tkhd, ok := findBox(trak, "tkhd"); ok {
		width, height, _ = parseTkhdDimensions(tkhd)
	}

	return &videoMetadata{
		Duration: durationSec,
		FPS:      float64(sampleCount) / durationSec,
		Width:    width,
		Height:   height,
	}, nil
}

// parseMdhd extracts timescale and duration from an mdhd box payload (version 0 or 1).
func parseMdhd(mdhd []byte) (timescale uint32, duration uint64, err error) {
	if len(mdhd) < 1 {
		return 0, 0, errors.New("mdhd box too short")
	}
	if mdhd[0] == 1 {
		if len(mdhd) < 32 {
			return 0, 0, errors.New("mdhd v1 box too short")
		}
		timescale = binary.BigEndian.Uint32(mdhd[20:24])
		duration = binary.BigEndian.Uint64(mdhd[24:32])
		return timescale, duration, nil
	}
	if len(mdhd) < 20 {
		return 0, 0, errors.New("mdhd v0 box too short")
	}
	timescale = binary.BigEndian.Uint32(mdhd[12:16])
	duration = uint64(binary.BigEndian.Uint32(mdhd[16:20]))
	return timescale, duration, nil
}

// parseStszSampleCount extracts the sample count from an stsz box payload. For a video track,
// the sample count equals the number of frames.
func parseStszSampleCount(stsz []byte) (uint32, error) {
	if len(stsz) < 12 {
		return 0, errors.New("stsz box too short")
	}
	return binary.BigEndian.Uint32(stsz[8:12]), nil
}

// parseTkhdDimensions extracts width and height from a tkhd box payload. Width and height are
// always the last two 16.16 fixed-point fields of the box, regardless of version.
func parseTkhdDimensions(tkhd []byte) (int, int, error) {
	if len(tkhd) < 8 {
		return 0, 0, errors.New("tkhd box too short")
	}
	width := binary.BigEndian.Uint32(tkhd[len(tkhd)-8:len(tkhd)-4]) >> 16
	height := binary.BigEndian.Uint32(tkhd[len(tkhd)-4:]) >> 16
	return int(width), int(height), nil
}

// findBox returns the payload (content after the box header) of the first direct child box of
// the given type within data.
func findBox(data []byte, boxType string) ([]byte, bool) {
	var payload []byte
	found := false
	iterateBoxes(data, func(t string, p []byte) bool {
		if t == boxType {
			payload, found = p, true
			return false
		}
		return true
	})
	return payload, found
}

// iterateBoxes walks the direct child boxes within data, calling fn with each box's type and
// payload. Iteration stops early if fn returns false.
func iterateBoxes(data []byte, fn func(boxType string, payload []byte) bool) {
	i := 0
	for i+8 <= len(data) {
		size := binary.BigEndian.Uint32(data[i : i+4])
		boxType := string(data[i+4 : i+8])

		headerSize := 8
		var boxSize int
		switch size {
		case 0:
			boxSize = len(data) - i
		case 1:
			if i+16 > len(data) {
				return
			}
			boxSize = int(binary.BigEndian.Uint64(data[i+8 : i+16]))
			headerSize = 16
		default:
			boxSize = int(size)
		}
		if boxSize < headerSize || i+boxSize > len(data) {
			return
		}

		if !fn(boxType, data[i+headerSize:i+boxSize]) {
			return
		}
		i += boxSize
	}
}
