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
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

func TestApproximatePrefixCacheTokenEstimator(t *testing.T) {
	tests := []struct {
		name          string
		multimodalCfg *multiModalTokenEstimatorConfig
		block         fwkrh.PromptBlock
		expected      int
	}{
		{
			name:          "EmptyText",
			multimodalCfg: nil,
			block:         fwkrh.PromptBlock{Type: fwkrh.BlockTypeText, Text: ""},
			expected:      0,
		},
		{
			name:          "Text_4Chars",
			multimodalCfg: nil,
			block:         fwkrh.PromptBlock{Type: fwkrh.BlockTypeText, Text: "aaaa"},
			expected:      1,
		},
		{
			name:          "Text_5Chars",
			multimodalCfg: nil,
			block:         fwkrh.PromptBlock{Type: fwkrh.BlockTypeText, Text: "aaaaa"},
			expected:      1,
		},
		{
			name: "Image_Fixed",
			multimodalCfg: &multiModalTokenEstimatorConfig{
				Image: &imageTokenEstimatorConfig{
					Mode: ModeFixed,
					FixedCfg: &fixedTokenEstimatorConfig{
						FixedToken: 10,
					},
				},
			},
			block: fwkrh.PromptBlock{
				Type:     fwkrh.BlockTypeImage,
				AssetURI: "https://example.com/image.jpg",
			},
			expected: 10,
		},
		{
			name: "Image_Dynamic",
			multimodalCfg: &multiModalTokenEstimatorConfig{
				Image: &imageTokenEstimatorConfig{
					Mode: ModeDynamic,
					DefaultResolution: resolution{
						Width:  1920,
						Height: 1080,
					},
					DynamicCfg: &dynamicTokenEstimatorConfig{
						Factor: 1024,
					},
				},
			},
			block: fwkrh.PromptBlock{
				Type:     fwkrh.BlockTypeImage,
				AssetURI: base64Image180p1,
			},
			expected: 56,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			estimator := NewApproximatePrefixCacheTokenEstimator(context.Background(), tt.multimodalCfg)
			assert.Equal(t, tt.expected, estimator.Estimate(tt.block))
		})
	}
}
