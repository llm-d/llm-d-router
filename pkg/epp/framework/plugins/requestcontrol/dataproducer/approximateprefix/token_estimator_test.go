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
		block         fwkrh.ContentBlock
		expected      int
	}{
		{
			name:          "EmptyText",
			multimodalCfg: nil,
			block:         fwkrh.ContentBlock{Type: "text", Text: ""},
			expected:      0,
		},
		{
			name:          "Text_4Chars",
			multimodalCfg: nil,
			block:         fwkrh.ContentBlock{Type: "text", Text: "aaaa"},
			expected:      1,
		},
		{
			name:          "Text_5Chars",
			multimodalCfg: nil,
			block:         fwkrh.ContentBlock{Type: "text", Text: "aaaaa"},
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
			block: fwkrh.ContentBlock{
				Type:     "image_url",
				ImageURL: fwkrh.ImageBlock{URL: "https://example.com/image.jpg"},
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
			block: fwkrh.ContentBlock{
				Type:     "image_url",
				ImageURL: fwkrh.ImageBlock{URL: base64Image180p1},
			},
			expected: 56,
		},
		{
			name: "Audio_Fixed",
			multimodalCfg: &multiModalTokenEstimatorConfig{
				Audio: &fixedTokenEstimatorConfig{FixedToken: 512},
			},
			block: fwkrh.ContentBlock{
				Type:       "input_audio",
				InputAudio: fwkrh.AudioBlock{Data: "abc", Format: "wav"},
			},
			expected: 512,
		},
		{
			name: "Video_Fixed",
			multimodalCfg: &multiModalTokenEstimatorConfig{
				Video: &fixedTokenEstimatorConfig{FixedToken: 2000},
			},
			block: fwkrh.ContentBlock{
				Type:     "video_url",
				VideoURL: fwkrh.VideoBlock{URL: "https://example.com/video.mp4"},
			},
			expected: 2000,
		},
		{
			name:          "Audio_Default",
			multimodalCfg: nil,
			block: fwkrh.ContentBlock{
				Type:       "input_audio",
				InputAudio: fwkrh.AudioBlock{Data: "abc", Format: "mp3"},
			},
			expected: 256,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			estimator := NewApproximatePrefixCacheTokenEstimator(context.Background(), tt.multimodalCfg)
			assert.Equal(t, tt.expected, estimator.Estimate(tt.block))
		})
	}
}
