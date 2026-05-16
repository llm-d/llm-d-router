package preciseprefixcache

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

func TestScorer_GetEngineKeysForRequest(t *testing.T) {
	d, err := os.Getwd()
	require.NoError(t, err)
	_ = filepath.Join(d, "testdata") // modelDir retained for future tokenizer-based tests

	ctx := utils.NewTestContext(t)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		HashAlgorithm:   kvblock.HashAlgorithmSHA256CBOR,
		HashSeed:        "10",
		BlockSizeTokens: 64,
	})
	require.NoError(t, err)

	scorer := &Scorer{
		tokenProcessor:       tokenProcessor,
		tokenProcessorConfig: &kvblock.TokenProcessorConfig{BlockSizeTokens: 64},
	}

	tests := []struct {
		name          string
		request       *scheduling.InferenceRequest
		wantErr       bool
		wantKeysCount int
		checkNonZero  bool
	}{
		{
			name:          "nil request",
			request:       nil,
			wantErr:       false,
			wantKeysCount: 0,
		},
		{
			name: "nil request body",
			request: &scheduling.InferenceRequest{
				RequestID:   "test-1",
				TargetModel: "test-model",
				Body:        nil,
			},
			wantErr:       false,
			wantKeysCount: 0,
		},
		{
			name: "nil tokenized prompt",
			request: &scheduling.InferenceRequest{
				RequestID:   "test-2",
				TargetModel: "test-model",
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: nil,
				},
			},
			wantErr:       false,
			wantKeysCount: 0,
		},
		{
			name: "empty token IDs",
			request: &scheduling.InferenceRequest{
				RequestID:   "test-3",
				TargetModel: "test-model",
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &types.TokenizedPrompt{
						TokenIDs: []int{},
					},
				},
			},
			wantErr:       false,
			wantKeysCount: 0,
		},
		{
			name: "valid request with tokens",
			request: &scheduling.InferenceRequest{
				RequestID:   "test-4",
				TargetModel: "test-model",
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &types.TokenizedPrompt{
						TokenIDs: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
					},
				},
			},
			wantErr:       false,
			wantKeysCount: 1,
			checkNonZero:  true,
		},
		{
			name: "request with multiple blocks",
			request: &scheduling.InferenceRequest{
				RequestID:   "test-5",
				TargetModel: "test-model",
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &types.TokenizedPrompt{
						TokenIDs: make([]int, 150),
					},
				},
			},
			wantErr:       false,
			wantKeysCount: 3,
			checkNonZero:  true,
		},
		{
			name: "request with multimodal features",
			request: &scheduling.InferenceRequest{
				RequestID:   "test-6",
				TargetModel: "test-model",
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &types.TokenizedPrompt{
						TokenIDs: []int{1, 2, 3, 4, 5},
						MultiModalFeatures: []types.MultiModalFeature{
							{
								Type:        "image",
								Hash:        "test-hash-1",
								Placeholder: "<image>",
							},
						},
					},
				},
			},
			wantErr:       false,
			wantKeysCount: 1,
			checkNonZero:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.request != nil && tt.request.Body != nil && tt.request.Body.TokenizedPrompt != nil {
				for i := range tt.request.Body.TokenizedPrompt.TokenIDs {
					if tt.request.Body.TokenizedPrompt.TokenIDs[i] == 0 {
						tt.request.Body.TokenizedPrompt.TokenIDs[i] = i + 1
					}
				}
			}

			engineKeys, err := scorer.GetEngineKeysForRequest(ctx, tt.request)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, tt.wantKeysCount, len(engineKeys), "unexpected number of engine keys")

			if tt.checkNonZero {
				for i, key := range engineKeys {
					assert.NotZero(t, key, "engine key at index %d should not be zero", i)
				}
			}
		})
	}
}

func TestScorer_GetEngineKeysForRequest_WithoutTokenProcessor(t *testing.T) {
	scorer := &Scorer{
		tokenProcessor: nil,
		tokenProcessorConfig: &kvblock.TokenProcessorConfig{
			HashAlgorithm:   kvblock.HashAlgorithmSHA256CBOR,
			HashSeed:        "10",
			BlockSizeTokens: 64,
		},
	}

	ctx := utils.NewTestContext(t)
	request := &scheduling.InferenceRequest{
		RequestID:   "test-no-processor",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &types.TokenizedPrompt{
				TokenIDs: []int{1, 2, 3, 4, 5},
			},
		},
	}

	engineKeys, err := scorer.GetEngineKeysForRequest(ctx, request)
	assert.NoError(t, err)
	assert.NotEmpty(t, engineKeys, "should return engine keys even without pre-existing token processor")
	assert.Equal(t, 1, len(engineKeys), "should return 1 block for 5 tokens")
}

func TestScorer_GetEngineKeysForRequest_Consistency(t *testing.T) {
	ctx := utils.NewTestContext(t)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		HashAlgorithm:   kvblock.HashAlgorithmSHA256CBOR,
		HashSeed:        "10",
		BlockSizeTokens: 64,
	})
	require.NoError(t, err)

	scorer := &Scorer{
		tokenProcessor:       tokenProcessor,
		tokenProcessorConfig: &kvblock.TokenProcessorConfig{BlockSizeTokens: 64},
	}

	request := &scheduling.InferenceRequest{
		RequestID:   "test-consistency",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &types.TokenizedPrompt{
				TokenIDs: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			},
		},
	}

	keys1, err := scorer.GetEngineKeysForRequest(ctx, request)
	require.NoError(t, err)

	keys2, err := scorer.GetEngineKeysForRequest(ctx, request)
	require.NoError(t, err)

	assert.Equal(t, keys1, keys2, "engine keys should be consistent for the same request")
}

func TestScorer_GetEngineKeysForRequest_DifferentRequests(t *testing.T) {
	ctx := utils.NewTestContext(t)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		HashAlgorithm:   kvblock.HashAlgorithmSHA256CBOR,
		HashSeed:        "10",
		BlockSizeTokens: 64,
	})
	require.NoError(t, err)

	scorer := &Scorer{
		tokenProcessor:       tokenProcessor,
		tokenProcessorConfig: &kvblock.TokenProcessorConfig{BlockSizeTokens: 64},
	}

	request1 := &scheduling.InferenceRequest{
		RequestID:   "test-diff-1",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &types.TokenizedPrompt{
				TokenIDs: []int{1, 2, 3, 4, 5},
			},
		},
	}

	request2 := &scheduling.InferenceRequest{
		RequestID:   "test-diff-2",
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &types.TokenizedPrompt{
				TokenIDs: []int{6, 7, 8, 9, 10},
			},
		},
	}

	keys1, err := scorer.GetEngineKeysForRequest(ctx, request1)
	require.NoError(t, err)

	keys2, err := scorer.GetEngineKeysForRequest(ctx, request2)
	require.NoError(t, err)

	assert.NotEqual(t, keys1, keys2, "different requests should produce different engine keys")
}
