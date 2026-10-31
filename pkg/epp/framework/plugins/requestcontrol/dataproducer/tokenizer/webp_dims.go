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

import "encoding/binary"

// webpHeaderBytes covers the RIFF container plus the complete size fields of
// every webp format: the VP8X canvas size, the VP8 keyframe size, and the
// VP8L lossless size all end within the first 30 bytes of the file.
const webpHeaderBytes = 30

// webpDimensions reads the pixel dimensions from the start of a webp header.
// ok is false unless the bytes carry the minimal complete chunk header of the
// format they declare.
func webpDimensions(hdr []byte) (width, height int, ok bool) {
	if len(hdr) < 12 || string(hdr[:4]) != "RIFF" || string(hdr[8:12]) != "WEBP" {
		return 0, 0, false
	}
	payload := hdr[20:]
	switch string(hdr[12:16]) {
	case "VP8X":
		// Extended format: 1-byte flags and 3 reserved bytes, then the
		// canvas size minus one in each direction.
		if len(payload) < 10 {
			return 0, 0, false
		}
		w := int(payload[4]) | int(payload[5])<<8 | int(payload[6])<<16
		h := int(payload[7]) | int(payload[8])<<8 | int(payload[9])<<16
		return w + 1, h + 1, true
	case "VP8 ":
		// Lossy: 3-byte frame tag, 3-byte start code, then 14-bit sizes.
		if len(payload) < 10 ||
			payload[3] != 0x9d || payload[4] != 0x01 || payload[5] != 0x2a {
			return 0, 0, false
		}
		w := int(binary.LittleEndian.Uint16(payload[6:8]) & 0x3fff)
		h := int(binary.LittleEndian.Uint16(payload[8:10]) & 0x3fff)
		if w == 0 || h == 0 {
			return 0, 0, false
		}
		return w, h, true
	case "VP8L":
		// Lossless: 0x2f signature byte, then 14-bit width-1 and height-1.
		if len(payload) < 5 || payload[0] != 0x2f {
			return 0, 0, false
		}
		bits := binary.LittleEndian.Uint32(payload[1:5])
		w := int(bits&0x3fff) + 1
		h := int((bits>>14)&0x3fff) + 1
		return w, h, true
	default:
		return 0, 0, false
	}
}
