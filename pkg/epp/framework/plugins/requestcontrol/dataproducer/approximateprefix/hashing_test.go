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

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
)

func newCompletionRequest(prompt fwkrh.Prompt) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: prompt},
		},
	}
}

func newEmbeddingsRequest(input fwkrh.EmbeddingsInput) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Embeddings: &fwkrh.EmbeddingsRequest{Input: input},
		},
	}
}

func TestGetUserInputBytes_CompletionsShapes(t *testing.T) {
	rawBytes, err := getUserInputBytes(newCompletionRequest(fwkrh.Prompt{Raw: "hello world"}))
	require.NoError(t, err)
	assert.Equal(t, []byte("hello world"), rawBytes)

	stringsBytes, err := getUserInputBytes(newCompletionRequest(fwkrh.Prompt{Strings: []string{"hello", "world"}}))
	require.NoError(t, err)
	assert.Equal(t, []byte("hello world"), stringsBytes, "Strings should join with space, matching Raw equivalent")

	tokenIDs := []uint32{1, 2, 3}
	tokenBytes, err := getUserInputBytes(newCompletionRequest(fwkrh.Prompt{TokenIDs: tokenIDs}))
	require.NoError(t, err)
	assert.Len(t, tokenBytes, 4*len(tokenIDs))
	assert.NotEmpty(t, tokenBytes, "pre-tokenized prompts must contribute non-empty hash input")
}

func TestGetUserInputBytes_EmbeddingsShapes(t *testing.T) {
	rawBytes, err := getUserInputBytes(newEmbeddingsRequest(fwkrh.EmbeddingsInput{Raw: "embed me"}))
	require.NoError(t, err)
	assert.Equal(t, []byte("embed me"), rawBytes)

	stringsBytes, err := getUserInputBytes(newEmbeddingsRequest(fwkrh.EmbeddingsInput{Strings: []string{"embed", "me"}}))
	require.NoError(t, err)
	assert.Equal(t, []byte("embed me"), stringsBytes)

	tokenIDs := []uint32{42, 43}
	tokenBytes, err := getUserInputBytes(newEmbeddingsRequest(fwkrh.EmbeddingsInput{TokenIDs: tokenIDs}))
	require.NoError(t, err)
	assert.Len(t, tokenBytes, 4*len(tokenIDs))
	assert.NotEmpty(t, tokenBytes)
}

func TestGetUserInputBytes_DistinctTokenIDsDistinctBytes(t *testing.T) {
	a, err := getUserInputBytes(newCompletionRequest(fwkrh.Prompt{TokenIDs: []uint32{1, 2, 3}}))
	require.NoError(t, err)
	b, err := getUserInputBytes(newCompletionRequest(fwkrh.Prompt{TokenIDs: []uint32{1, 2, 4}}))
	require.NoError(t, err)
	assert.NotEqual(t, a, b)
}

func TestHashPrompt_PreTokenizedProducesBlocks(t *testing.T) {
	// 16 token IDs * 4 bytes/token = 64 bytes; blockSize 4 tokens * avgChars
	// per token = 16 chars per block, so we expect at least one block.
	ids := make([]uint32, 16)
	for i := range ids {
		ids[i] = uint32(i + 1)
	}
	req := newCompletionRequest(fwkrh.Prompt{TokenIDs: ids})

	hashes := hashPrompt(context.Background(), req, 4, defaultMaxPrefixBlocks)
	assert.NotEmpty(t, hashes, "pre-tokenized prompt must produce prefix-cache blocks")
}
