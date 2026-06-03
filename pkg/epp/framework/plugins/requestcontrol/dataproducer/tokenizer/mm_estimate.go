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
	// Image estimation modes.
	imageModeDynamic = "dynamic"
	imageModeStatic  = "static"

	// defaultImageWidth and defaultImageHeight model a 360p image, used when an
	// image URL is not a decodable base64 payload.
	defaultImageWidth  = 640
	defaultImageHeight = 360
	// imageTokenFactor maps image pixels to placeholder tokens (width*height/factor).
	imageTokenFactor = 1024
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

// placeholderCount estimates the number of placeholder tokens an image occupies.
// Static mode returns a constant count; dynamic mode uses decoded pixel dimensions
// (or the default resolution) divided by the factor. The result is always >= 1 so
// an image contributes weight to the pseudo-token stream.
func (e imageEstimator) placeholderCount(url string) int {
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
	if rw, rh, ok := imageDimensionsFromBase64(url); ok {
		w, h = rw, rh
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
