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
	"strings"

	// Registers decoders so image.DecodeConfig can read dimensions.
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
)

const (
	// defaultImageWidth and defaultImageHeight model a 360p image, used when an
	// image URL is not a decodable base64 payload.
	defaultImageWidth  = 640
	defaultImageHeight = 360
	// imageTokenFactor maps image pixels to placeholder tokens (width*height/factor).
	imageTokenFactor = 1024
)

// imagePlaceholderCount estimates the number of placeholder tokens an image
// occupies. Decodable base64 payloads use their pixel dimensions; everything
// else falls back to the default resolution. The result is always >= 1 so an
// image contributes weight to the pseudo-token stream.
func imagePlaceholderCount(url string) int {
	w, h := defaultImageWidth, defaultImageHeight
	if rw, rh, ok := imageDimensionsFromBase64(url); ok {
		w, h = rw, rh
	}
	if n := (w * h) / imageTokenFactor; n > 0 {
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
	decoded, err := base64.StdEncoding.DecodeString(url[idx+len("base64,"):])
	if err != nil {
		return 0, 0, false
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(decoded))
	if err != nil || cfg.Width <= 0 || cfg.Height <= 0 {
		return 0, 0, false
	}
	return cfg.Width, cfg.Height, true
}
