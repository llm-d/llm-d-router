package approximateprefix

import (
	"context"
	"encoding/binary"
	"testing"

	"github.com/cespare/xxhash/v2"
	"github.com/stretchr/testify/assert"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	base64Image180p1 = "data:image/jpeg;base64,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"
	base64Image180p2 = "data:image/jpeg;base64,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"
)

func TestGetContentBlocks(t *testing.T) {
	tests := []struct {
		name                  string
		request               *fwksched.InferenceRequest
		blockSizeTokens       int
		multimodalCfg         *multiModalTokenEstimatorConfig
		expectedContentBlocks []HashBlock
		expectErr             bool
	}{
		{
			name: "Completions",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "aaaabbbb"},
					},
				},
			},
			blockSizeTokens: 16,
			multimodalCfg:   nil,
			expectedContentBlocks: []HashBlock{
				{PseudoTokens: []byte("aaaabbbb")},
			},
			expectErr: false,
		},
		{
			name: "ChatCompletions_Text",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{Role: "user", Content: fwkrh.Content{Raw: "Hello"}},
						},
					},
				},
			},
			blockSizeTokens: 16,
			multimodalCfg:   nil,
			expectedContentBlocks: []HashBlock{
				{PseudoTokens: []byte("userHello")},
			},
			expectErr: false,
		},
		{
			name: "ChatCompletions_Roles",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{Role: "user", Content: fwkrh.Content{Raw: "Hello"}},
							{Role: "assistant", Content: fwkrh.Content{Raw: "cici"}},
						},
					},
				},
			},
			blockSizeTokens: 16,
			multimodalCfg:   nil,
			expectedContentBlocks: []HashBlock{
				{PseudoTokens: []byte("userHelloassistantcici")},
			},
			expectErr: false,
		},
		{
			name: "ChatCompletions_ImageURL_Fixed",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{
								Role: "user",
								Content: fwkrh.Content{
									Structured: []fwkrh.ContentBlock{
										{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/image.jpg"}},
									},
								},
							},
						},
					},
				},
			},
			blockSizeTokens: 16,
			multimodalCfg: &multiModalTokenEstimatorConfig{
				Image: &imageTokenEstimatorConfig{
					Mode: ModeFixed,
					FixedCfg: &fixedTokenEstimatorConfig{
						FixedToken: 280,
					},
				},
			},
			expectedContentBlocks: []HashBlock{
				{PseudoTokens: append([]byte("user"), repeatBytes(imageHashBytes("https://example.com/image.jpg"), 15)...)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes("https://example.com/image.jpg"), 9)},
			},
			expectErr: false,
		},
		{
			name: "ChatCompletions_ImageURL_Dynamic_Fallback",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{
								Role: "system",
								Content: fwkrh.Content{
									Structured: []fwkrh.ContentBlock{
										{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "data:image/jpeg;base64,bm90IGFuIGltYWdl"}},
									},
								},
							},
						},
					},
				},
			},
			blockSizeTokens: 16,
			multimodalCfg: &multiModalTokenEstimatorConfig{
				Image: &imageTokenEstimatorConfig{
					Mode: ModeDynamic,
					DefaultResolution: resolution{
						Width:  10,
						Height: 10,
					},
					DynamicCfg: &dynamicTokenEstimatorConfig{
						Factor: 10,
					},
				},
			},
			expectedContentBlocks: []HashBlock{
				{PseudoTokens: append([]byte("system\x00\x00"), repeatBytes(imageHashBytes("data:image/jpeg;base64,bm90IGFuIGltYWdl"), 10)...)},
			},
			expectErr: false,
		},
		{
			name: "ChatCompletions_ImageContent_Fixed",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{
								Role: "user",
								Content: fwkrh.Content{
									Structured: []fwkrh.ContentBlock{
										{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: base64Image180p1}},
									},
								},
							},
						},
					},
				},
			},
			blockSizeTokens: 16,
			multimodalCfg: &multiModalTokenEstimatorConfig{
				Image: &imageTokenEstimatorConfig{
					Mode: ModeFixed,
					FixedCfg: &fixedTokenEstimatorConfig{
						FixedToken: 10,
					},
				},
			},
			expectedContentBlocks: []HashBlock{
				{PseudoTokens: append([]byte("user"), repeatBytes(imageHashBytes(base64Image180p1), 10)...)},
			},
			expectErr: false,
		},
		{
			name: "ChatCompletions_ImageContent_Dynamic_Success",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{
								Role: "user",
								Content: fwkrh.Content{
									Structured: []fwkrh.ContentBlock{
										{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: base64Image180p1}},
									},
								},
							},
						},
					},
				},
			},
			blockSizeTokens: 16,
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
			expectedContentBlocks: []HashBlock{
				{PseudoTokens: append([]byte("user"), repeatBytes(imageHashBytes(base64Image180p1), 15)...)},
				{PseudoTokens: repeatBytes(imageHashBytes(base64Image180p1), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes(base64Image180p1), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes(base64Image180p1), 9)},
			},
			expectErr: false,
		},
		{
			name: "ChatCompletions_TEXT_imageContent_Dynamic_Success",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{
								Role: "user",
								Content: fwkrh.Content{
									Structured: []fwkrh.ContentBlock{
										{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: base64Image180p1}},
										{Type: "text", Text: "aaaaaa"},
										{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: base64Image180p2}},
									},
								},
							},
						},
					},
				},
			},
			blockSizeTokens: 16,
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
			expectedContentBlocks: []HashBlock{
				{PseudoTokens: append([]byte("user"), repeatBytes(imageHashBytes(base64Image180p1), 15)...)},
				{PseudoTokens: repeatBytes(imageHashBytes(base64Image180p1), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes(base64Image180p1), 16)},
				{PseudoTokens: append(append(repeatBytes(imageHashBytes(base64Image180p1), 9), []byte("aaaaaa\x00\x00")...), repeatBytes(imageHashBytes(base64Image180p2), 5)...)},
				{PseudoTokens: repeatBytes(imageHashBytes(base64Image180p2), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes(base64Image180p2), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes(base64Image180p2), 16)},
				{PseudoTokens: repeatBytes(imageHashBytes(base64Image180p2), 3)},
			},
			expectErr: false,
		},
		{
			name: "Generate_TokenIDs",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Generate: &fwkrh.GenerateRequest{
						TokenIDs: []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
					},
				},
			},
			blockSizeTokens: 4,
			multimodalCfg:   nil,
			expectedContentBlocks: []HashBlock{
				{Tokens: []uint32{1, 2, 3, 4}},
				{Tokens: []uint32{5, 6, 7, 8}},
				{Tokens: []uint32{9, 10}},
			},
			expectErr: false,
		},
		{
			name: "Invalid_Body",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{},
			},
			blockSizeTokens:       16,
			multimodalCfg:         nil,
			expectedContentBlocks: nil,
			expectErr:             true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			seq, err := getKVCacheBlocksFromRawPrompt(context.Background(), tt.request, tt.blockSizeTokens, NewApproximatePrefixCacheTokenEstimator(context.Background(), tt.multimodalCfg))
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				blocks := make([]HashBlock, 0, len(tt.expectedContentBlocks))
				for block := range seq {
					blocks = append(blocks, block)
				}
				assert.Equal(t, tt.expectedContentBlocks, blocks)
			}
		})
	}
}

func TestKVCacheBlock_Hash(t *testing.T) {
	tests := []struct {
		name     string
		blockA   HashBlock
		blockB   HashBlock
		shouldEq bool
	}{
		{
			name: "Identical Blocks",
			blockA: HashBlock{
				PseudoTokens: []byte("Hello"),
				Tokens:       []uint32{1, 2},
			},
			blockB: HashBlock{
				PseudoTokens: []byte("Hello"),
				Tokens:       []uint32{1, 2},
			},
			shouldEq: true,
		},
		{
			name: "Different PseudoBytes Content",
			blockA: HashBlock{
				PseudoTokens: []byte("Hello"),
			},
			blockB: HashBlock{
				PseudoTokens: []byte("Hellp"),
			},
			shouldEq: false,
		},
		{
			name: "Different Token IDs",
			blockA: HashBlock{
				Tokens: []uint32{1, 2},
			},
			blockB: HashBlock{
				Tokens: []uint32{1, 3},
			},
			shouldEq: false,
		},
		{
			name:     "Empty fields match",
			blockA:   HashBlock{},
			blockB:   HashBlock{},
			shouldEq: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hashA := tt.blockA.Hash()
			hashB := tt.blockB.Hash()
			if tt.shouldEq {
				assert.Equal(t, hashA, hashB)
			} else {
				assert.NotEqual(t, hashA, hashB)
			}
		})
	}
}

func repeatBytes(b []byte, count int) []byte {
	res := make([]byte, 0, len(b)*count)
	for i := 0; i < count; i++ {
		res = append(res, b...)
	}
	return res
}

func imageHashBytes(url string) []byte {
	h := xxhash.Sum64([]byte(url))
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, uint32(h))
	return buf
}
