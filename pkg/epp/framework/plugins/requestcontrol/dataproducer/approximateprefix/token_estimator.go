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
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"strings"

	// needed for image dimension parse
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// TokenEstimator estimates the number of tokens for different content types.
type TokenEstimator interface {
	Estimate(block fwkrh.ContentBlock) int
}

type approximatePrefixCacheTokenEstimator struct {
	ctx              context.Context
	multimodalConfig *multiModalTokenEstimatorConfig
}

// NewApproximatePrefixCacheTokenEstimator returns a new TokenEstimator.
func NewApproximatePrefixCacheTokenEstimator(ctx context.Context, multimodalConfig *multiModalTokenEstimatorConfig) TokenEstimator {
	return &approximatePrefixCacheTokenEstimator{
		ctx:              ctx,
		multimodalConfig: multimodalConfig,
	}
}

func (e *approximatePrefixCacheTokenEstimator) Estimate(block fwkrh.ContentBlock) int {
	switch block.Type {
	case "text":
		return len(block.Text) / averageCharactersPerToken
	case "image_url":
		return getImagePlaceholders(e.ctx, block.ImageURL.URL, e.multimodalConfig)
	case "video_url":
		return getVideoPlaceholders(e.multimodalConfig)
	case "input_audio", "audio_url":
		return getAudioPlaceholders(e.multimodalConfig)
	default:
		return 0
	}
}

func getImagePlaceholders(ctx context.Context, url string, multimodalConfig *multiModalTokenEstimatorConfig) int {
	if multimodalConfig == nil || multimodalConfig.Image == nil {
		multimodalConfig = &defaultMultimodalConfig
	}
	logger := log.FromContext(ctx).V(logutil.DEBUG)
	var numPlaceHolders int
	switch multimodalConfig.Image.Mode {
	case ModeFixed:
		numPlaceHolders = multimodalConfig.Image.FixedCfg.FixedToken
		logger.Info("using fixed token placeholders")
	case ModeDynamic:
		if strings.HasPrefix(url, "data:image/") && strings.Contains(url, "base64,") {
			resolution, err := getImageDimensionsFromBase64(url)
			if err != nil {
				logger.Error(err, "failed to get image dimensions from base64 content, using default image resolution")
				numPlaceHolders = multimodalConfig.Image.DefaultResolution.Width * multimodalConfig.Image.DefaultResolution.Height / multimodalConfig.Image.DynamicCfg.Factor
			} else {
				logger.Info(fmt.Sprintf("Using image resolution height %d width %d", resolution.Height, resolution.Width))
				numPlaceHolders = (resolution.Width * resolution.Height) / (multimodalConfig.Image.DynamicCfg.Factor)
			}
		} else {
			logger.Info("Failed to get image dimensions with unsupported type, now we only support base64 encoded image content, using default image resolution")
			numPlaceHolders = multimodalConfig.Image.DefaultResolution.Width * multimodalConfig.Image.DefaultResolution.Height / multimodalConfig.Image.DynamicCfg.Factor
		}
	}
	logger.Info(fmt.Sprintf("Using numPlaceHolders %d", numPlaceHolders))
	return numPlaceHolders
}

func getAudioPlaceholders(multimodalConfig *multiModalTokenEstimatorConfig) int {
	if multimodalConfig == nil || multimodalConfig.Audio == nil {
		return defaultMultimodalConfig.Audio.FixedToken
	}
	return multimodalConfig.Audio.FixedToken
}

func getVideoPlaceholders(multimodalConfig *multiModalTokenEstimatorConfig) int {
	if multimodalConfig == nil || multimodalConfig.Video == nil {
		return defaultMultimodalConfig.Video.FixedToken
	}
	return multimodalConfig.Video.FixedToken
}

func getImageDimensionsFromBase64(url string) (*resolution, error) {
	idx := strings.Index(url, "base64,")
	base64Data := url[idx+7:]
	decoded, err := base64.StdEncoding.DecodeString(base64Data)
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64: %w", err)
	}
	config, _, err := image.DecodeConfig(bytes.NewReader(decoded))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image config: %w", err)
	}
	if config.Width <= 0 || config.Height <= 0 {
		return nil, errors.New("image config width and height must be positive")
	}
	return &resolution{
		Width:  config.Width,
		Height: config.Height,
	}, nil
}
