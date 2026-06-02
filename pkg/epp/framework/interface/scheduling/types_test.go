/*
Copyright 2025 The Kubernetes Authors.

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

package scheduling

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

type cloneableString string

func (s cloneableString) Clone() fwkdl.Cloneable { return s }

func newTestMetadata(name string) *fwkdl.EndpointMetadata {
	return &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Namespace: "ns", Name: name},
		PodName:        name,
		Address:        "10.0.0.1",
		Port:           "8000",
		MetricsHost:    "10.0.0.1:9100",
		Labels:         map[string]string{"app": "inference"},
	}
}

func newTestMetrics() *fwkdl.Metrics {
	m := fwkdl.NewMetrics()
	m.RunningRequestsSize = 3
	m.WaitingQueueSize = 1
	m.KVCacheUsagePercent = 0.42
	return m
}

func TestInferenceRequest_String_Nil(t *testing.T) {
	var r *InferenceRequest
	assert.Equal(t, nilString, r.String())
}

func TestInferenceRequest_String_HasFields(t *testing.T) {
	r := &InferenceRequest{
		RequestID:   "req-1",
		TargetModel: "llama-7b",
		Body:        &fwkrh.InferenceRequestBody{},
		Headers:     map[string]string{"x-trace": "abc"},
	}
	s := r.String()
	assert.Contains(t, s, "req-1")
	assert.Contains(t, s, "llama-7b")
	assert.Contains(t, s, "x-trace")
}

func TestNewEndpoint_CopiesInputs(t *testing.T) {
	meta := newTestMetadata("pod-a")
	metrics := newTestMetrics()
	attr := fwkdl.NewAttributes()
	attr.Put("key", cloneableString("value"))

	ep := NewEndpoint(meta, metrics, attr)
	assert.NotNil(t, ep)

	// mutating original metadata must not affect endpoint
	meta.Labels["app"] = "mutated"
	assert.Equal(t, "inference", ep.GetMetadata().Labels["app"])

	// mutating original metrics must not affect endpoint
	metrics.RunningRequestsSize = 999
	assert.Equal(t, 3, ep.GetMetrics().RunningRequestsSize)

	// values from attribute map should be retrievable
	v, ok := ep.Get("key")
	assert.True(t, ok)
	assert.Equal(t, cloneableString("value"), v)
}

func TestNewEndpoint_NilAttributeUsesDefault(t *testing.T) {
	ep := NewEndpoint(newTestMetadata("pod-b"), newTestMetrics(), nil)
	assert.NotNil(t, ep)
	assert.Empty(t, ep.Keys())

	// Should still be safe to write into the default-allocated attribute map
	ep.Put("k", cloneableString("v"))
	v, ok := ep.Get("k")
	assert.True(t, ok)
	assert.Equal(t, cloneableString("v"), v)
}

func TestEndpoint_StringNil(t *testing.T) {
	var ep *endpoint
	assert.Equal(t, nilString, ep.String())
}

func TestEndpoint_StringContainsFields(t *testing.T) {
	ep := NewEndpoint(newTestMetadata("pod-c"), newTestMetrics(), nil)
	s := ep.String()
	assert.Contains(t, s, "pod-c")
}

func TestEndpoint_Clone(t *testing.T) {
	attr := fwkdl.NewAttributes()
	attr.Put("k", cloneableString("v"))
	ep := NewEndpoint(newTestMetadata("pod-d"), newTestMetrics(), attr)

	cloned := ep.Clone()
	v, ok := cloned.Get("k")
	assert.True(t, ok)
	assert.Equal(t, cloneableString("v"), v)
}

func TestEndpointComparer_Equal(t *testing.T) {
	attrA := fwkdl.NewAttributes()
	attrA.Put("k", cloneableString("v"))
	attrB := fwkdl.NewAttributes()
	attrB.Put("k", cloneableString("v"))

	a := NewEndpoint(newTestMetadata("pod"), newTestMetrics(), attrA)
	b := NewEndpoint(newTestMetadata("pod"), newTestMetrics(), attrB)

	assert.True(t, EndpointComparer(a, b))
}

func TestEndpointComparer_DifferByMetadata(t *testing.T) {
	a := NewEndpoint(newTestMetadata("pod-a"), newTestMetrics(), nil)
	b := NewEndpoint(newTestMetadata("pod-b"), newTestMetrics(), nil)
	assert.False(t, EndpointComparer(a, b))
}

func TestEndpointComparer_DifferByMetrics(t *testing.T) {
	mA := newTestMetrics()
	mB := newTestMetrics()
	mB.WaitingQueueSize = 99
	a := NewEndpoint(newTestMetadata("pod"), mA, nil)
	b := NewEndpoint(newTestMetadata("pod"), mB, nil)
	assert.False(t, EndpointComparer(a, b))
}

func TestEndpointComparer_DifferByAttributeKeys(t *testing.T) {
	attrA := fwkdl.NewAttributes()
	attrA.Put("k1", cloneableString("v"))
	attrB := fwkdl.NewAttributes()
	attrB.Put("k1", cloneableString("v"))
	attrB.Put("extra", cloneableString("x"))

	a := NewEndpoint(newTestMetadata("pod"), newTestMetrics(), attrA)
	b := NewEndpoint(newTestMetadata("pod"), newTestMetrics(), attrB)

	assert.False(t, EndpointComparer(a, b))
}

func TestEndpointComparer_DifferByAttributeValues(t *testing.T) {
	attrA := fwkdl.NewAttributes()
	attrA.Put("k", cloneableString("v1"))
	attrB := fwkdl.NewAttributes()
	attrB.Put("k", cloneableString("v2"))

	a := NewEndpoint(newTestMetadata("pod"), newTestMetrics(), attrA)
	b := NewEndpoint(newTestMetadata("pod"), newTestMetrics(), attrB)

	assert.False(t, EndpointComparer(a, b))
}

func TestScoredEndpointComparer(t *testing.T) {
	a := NewEndpoint(newTestMetadata("pod"), newTestMetrics(), nil)
	b := NewEndpoint(newTestMetadata("pod"), newTestMetrics(), nil)

	assert.True(t, ScoredEndpointComparer(ScoredEndpoint{Endpoint: a, Score: 0.5}, ScoredEndpoint{Endpoint: b, Score: 0.5}))
	assert.False(t, ScoredEndpointComparer(ScoredEndpoint{Endpoint: a, Score: 0.5}, ScoredEndpoint{Endpoint: b, Score: 0.6}))

	other := NewEndpoint(newTestMetadata("pod-other"), newTestMetrics(), nil)
	assert.False(t, ScoredEndpointComparer(ScoredEndpoint{Endpoint: a, Score: 1}, ScoredEndpoint{Endpoint: other, Score: 1}))
}

func TestModalityAliases(t *testing.T) {
	// These aliases exist for ergonomic re-export. Confirm the values line up.
	assert.Equal(t, fwkrh.ModalityImage, ModalityImage)
}

func TestInferenceRequest_EstimatedTokenLength(t *testing.T) {
	tests := []struct {
		name              string
		request           *InferenceRequest
		expected          int64
		expectedTokenized bool
	}{
		{
			name: "TokenizedPrompt available",
			request: &InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &fwkrh.TokenizedPrompt{
						TokenIDs: []uint32{1, 2, 3, 4},
					},
				},
				RequestSizeBytes: 100, // should be ignored
			},
			expected:          4,
			expectedTokenized: true,
		},
		{
			name: "hint available (token inputs)",
			request: &InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					TokenInputs: []fwkrh.TokenizedInput{
						{TokenIDs: []uint32{1, 2, 3}},
					},
				},
				RequestSizeBytes: 100, // should be ignored
			},
			expected:          3,
			expectedTokenized: true,
		},
		{
			name: "hint available (generate request)",
			request: &InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					Generate: &fwkrh.GenerateRequest{
						TokenIDs: []uint32{1, 2, 3, 4, 5},
					},
				},
				RequestSizeBytes: 100, // should be ignored
			},
			expected:          5,
			expectedTokenized: true,
		},
		{
			name: "hint not available, large request size",
			request: &InferenceRequest{
				Body:             &fwkrh.InferenceRequestBody{},
				RequestSizeBytes: 100,
			},
			expected:          25,
			expectedTokenized: false,
		},
		{
			name: "hint not available, small request size (rounds to 1)",
			request: &InferenceRequest{
				Body:             &fwkrh.InferenceRequestBody{},
				RequestSizeBytes: 2,
			},
			expected:          1,
			expectedTokenized: false,
		},
		{
			name: "hint not available, zero request size",
			request: &InferenceRequest{
				Body:             &fwkrh.InferenceRequestBody{},
				RequestSizeBytes: 0,
			},
			expected:          1,
			expectedTokenized: false,
		},
		{
			name: "nil body, uses request size",
			request: &InferenceRequest{
				Body:             nil,
				RequestSizeBytes: 80,
			},
			expected:          20,
			expectedTokenized: false,
		},
		{
			name: "nil body, zero request size",
			request: &InferenceRequest{
				Body:             nil,
				RequestSizeBytes: 0,
			},
			expected:          0,
			expectedTokenized: false,
		},
		{
			name:              "nil request",
			request:           nil,
			expected:          0,
			expectedTokenized: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, tokenized := tt.request.EstimatedTokenLength()
			assert.Equal(t, tt.expected, result)
			assert.Equal(t, tt.expectedTokenized, tokenized)
		})
	}
}

func TestCompareTokenEstimationApproaches(t *testing.T) {
	// Structs for raw JSON marshaling comparison
	type ChatMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type ChatRequest struct {
		Model    string        `json:"model"`
		Messages []ChatMessage `json:"messages"`
	}
	type CompletionRequest struct {
		Model  string `json:"model"`
		Prompt string `json:"prompt"`
	}

	cases := []struct {
		name             string
		promptText       string
		rawRequestStruct any
	}{
		{
			name: "Short Completion Request",
			promptText: "Hello, how are you?",
			rawRequestStruct: CompletionRequest{
				Model:  "meta-llama/Llama-2-7b-chat-hf",
				Prompt: "Hello, how are you?",
			},
		},
		{
			name: "Long Completion Request",
			promptText: `The process of tokenization involves splitting a stream of text into smaller pieces, such as words or subwords. ` +
				`In the context of Large Language Models (LLMs), tokenization is a crucial preprocessing step because models do not process ` +
				`raw text directly. Instead, they operate on sequences of token IDs. There are many different algorithms for tokenization, ` +
				`including Byte-Pair Encoding (BPE), WordPiece, and SentencePiece. Each algorithm has its own trade-offs between vocabulary size ` +
				`and sequence length. BPE is commonly used in models like GPT and Llama. It starts with a vocabulary of individual bytes and ` +
				`iteratively merges the most frequent pairs of tokens. This allows the vocabulary to represent common words as single tokens, ` +
				`while falling back to byte-level representations for rare or unseen words. In EPP, we need a fast and efficient way to ` +
				`estimate token counts for incoming requests before actually invoking the tokenizer, which can be expensive. This benchmark ` +
				`compares different estimation approaches.`,
			rawRequestStruct: CompletionRequest{
				Model: "meta-llama/Llama-2-7b-chat-hf",
				Prompt: `The process of tokenization involves splitting a stream of text into smaller pieces, such as words or subwords. ` +
					`In the context of Large Language Models (LLMs), tokenization is a crucial preprocessing step because models do not process ` +
					`raw text directly. Instead, they operate on sequences of token IDs. There are many different algorithms for tokenization, ` +
					`including Byte-Pair Encoding (BPE), WordPiece, and SentencePiece. Each algorithm has its own trade-offs between vocabulary size ` +
					`and sequence length. BPE is commonly used in models like GPT and Llama. It starts with a vocabulary of individual bytes and ` +
					`iteratively merges the most frequent pairs of tokens. This allows the vocabulary to represent common words as single tokens, ` +
					`while falling back to byte-level representations for rare or unseen words. In EPP, we need a fast and efficient way to ` +
					`estimate token counts for incoming requests before actually invoking the tokenizer, which can be expensive. This benchmark ` +
					`compares different estimation approaches.`,
			},
		},
		{
			name: "Short Chat Request (1 message)",
			promptText: "What is the capital of France?",
			rawRequestStruct: ChatRequest{
				Model: "meta-llama/Llama-2-7b-chat-hf",
				Messages: []ChatMessage{
					{Role: "user", Content: "What is the capital of France?"},
				},
			},
		},
		{
			name: "Medium Chat Request (2 turns)",
			promptText: "You are a helpful assistant. What is the capital of France?",
			rawRequestStruct: ChatRequest{
				Model: "meta-llama/Llama-2-7b-chat-hf",
				Messages: []ChatMessage{
					{Role: "system", Content: "You are a helpful assistant."},
					{Role: "user", Content: "What is the capital of France?"},
				},
			},
		},
		{
			name: "Long Multi-turn Chat Request",
			promptText: `You are a helpful assistant. ` +
				`Hi, I need help with my research on disaggregated inference for LLMs. ` +
				`Sure! Disaggregated inference separates the prefill stage from the decode stage. ` +
				`Why is that beneficial? ` +
				`Because prefill is compute-bound (parallelizable) while decode is memory-bound (sequential). Separating them improves efficiency and throughput.`,
			rawRequestStruct: ChatRequest{
				Model: "meta-llama/Llama-2-7b-chat-hf",
				Messages: []ChatMessage{
					{Role: "system", Content: "You are a helpful assistant."},
					{Role: "user", Content: "Hi, I need help with my research on disaggregated inference for LLMs."},
					{Role: "assistant", Content: "Sure! Disaggregated inference separates the prefill stage from the decode stage."},
					{Role: "user", Content: "Why is that beneficial?"},
					{Role: "assistant", Content: "Because prefill is compute-bound (parallelizable) while decode is memory-bound (sequential). Separating them improves efficiency and throughput."},
				},
			},
		},
	}

	fmt.Println("\n=============================================================================")
	fmt.Printf("%-30s | %-10s | %-10s | %-10s | %-8s\n", "Test Case", "Req Size", "Size Est", "Text Est", "Err %")
	fmt.Println("-----------------------------------------------------------------------------")

	for _, tc := range cases {
		// Marshal to JSON to simulate real network payload size
		jsonBytes, err := json.Marshal(tc.rawRequestStruct)
		assert.NoError(t, err)

		reqSize := len(jsonBytes)

		// Approach A: Estimate from RequestSizeBytes / 4 (assuming 4 bytes per token fallback)
		sizeEstimate := max(int64(reqSize)/4, 1)

		// Approach B: Estimate from raw prompt characters / 4 (assuming roughly 4 chars per token)
		// This is what prompt text length-based logic would give us.
		textEstimate := max(int64(len(tc.promptText))/4, 1)

		// Calculate error percentage of size-based estimation compared to text-based estimation
		diff := sizeEstimate - textEstimate
		errPercent := (float64(diff) / float64(textEstimate)) * 100.0

		fmt.Printf("%-30s | %-10d | %-10d | %-10d | %-8.2f%%\n",
			tc.name, reqSize, sizeEstimate, textEstimate, errPercent)
	}
	fmt.Println("=============================================================================")
}
