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

package tokenization_test

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"

	tokenizerpb "github.com/llm-d/llm-d-kv-cache/api/tokenizerpb"
	. "github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
)

// mockTokenizationServer implements the TokenizationServiceServer interface for testing.
type mockTokenizationServer struct {
	tokenizerpb.UnimplementedTokenizationServiceServer
	initializeError bool
	tokenizeError   bool
	chatError       bool
	initialized     map[string]bool
	mmFeatures      *tokenizerpb.MultiModalFeatures
	lastChatRequest *tokenizerpb.RenderChatCompletionRequest
}

func newMockTokenizationServer() *mockTokenizationServer {
	return &mockTokenizationServer{
		initialized: make(map[string]bool),
	}
}

func (m *mockTokenizationServer) InitializeTokenizer(
	ctx context.Context,
	req *tokenizerpb.InitializeTokenizerRequest,
) (*tokenizerpb.InitializeTokenizerResponse, error) {
	if m.initializeError {
		return &tokenizerpb.InitializeTokenizerResponse{
			Success:      false,
			ErrorMessage: "mock initialization error",
		}, nil
	}

	m.initialized[req.ModelName] = true
	return &tokenizerpb.InitializeTokenizerResponse{
		Success: true,
	}, nil
}

func (m *mockTokenizationServer) Tokenize(
	ctx context.Context,
	req *tokenizerpb.TokenizeRequest,
) (*tokenizerpb.TokenizeResponse, error) {
	if m.tokenizeError {
		return &tokenizerpb.TokenizeResponse{
			Success:      false,
			ErrorMessage: "mock tokenization error",
		}, nil
	}

	// Check if model was initialized (matches real service behavior)
	if !m.initialized[req.ModelName] {
		return &tokenizerpb.TokenizeResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("model %s not initialized", req.ModelName),
		}, nil
	}

	// Simple deterministic mock tokenization: convert each rune to a token ID
	// This makes tests more realistic - different inputs produce different tokens
	input := req.Input
	tokens := make([]uint32, 0, len(input))
	offsets := make([]uint32, 0, len(input)*2)

	for i, r := range input {
		tokens = append(tokens, uint32(r)) // #nosec G115 -- rune from valid string is always non-negative
		// #nosec G115 -- i is bounded by string length, safe conversion
		offsets = append(offsets, uint32(i), uint32(i+1))
	}

	return &tokenizerpb.TokenizeResponse{
		InputIds:     tokens,
		Success:      true,
		OffsetPairs:  offsets,
		ErrorMessage: "",
	}, nil
}

func (m *mockTokenizationServer) RenderChatCompletion(
	_ context.Context,
	req *tokenizerpb.RenderChatCompletionRequest,
) (*tokenizerpb.RenderChatCompletionResponse, error) {
	if m.chatError {
		return &tokenizerpb.RenderChatCompletionResponse{
			Success:      false,
			ErrorMessage: "mock render chat completion error",
		}, nil
	}
	m.lastChatRequest = req

	// Produce fake token IDs from native proto message content.
	var tokens []uint32
	for _, msg := range req.Messages {
		for _, r := range msg.GetContent() {
			tokens = append(tokens, uint32(r)) // #nosec G115 -- rune from valid string is always non-negative
		}
	}

	resp := &tokenizerpb.RenderChatCompletionResponse{
		RequestId: "mock-request-id",
		TokenIds:  tokens,
		Success:   true,
	}

	if m.mmFeatures != nil {
		resp.Features = m.mmFeatures
	}

	return resp, nil
}

func (m *mockTokenizationServer) RenderCompletion(
	_ context.Context,
	req *tokenizerpb.RenderCompletionRequest,
) (*tokenizerpb.RenderCompletionResponse, error) {
	if m.tokenizeError {
		return &tokenizerpb.RenderCompletionResponse{
			Success:      false,
			ErrorMessage: "mock render completion error",
		}, nil
	}

	tokens := make([]uint32, 0, len(req.Prompt))
	for _, r := range req.Prompt {
		tokens = append(tokens, uint32(r)) // #nosec G115 -- rune from valid string is always non-negative
	}

	return &tokenizerpb.RenderCompletionResponse{
		RequestId: "mock-request-id",
		TokenIds:  tokens,
		Success:   true,
	}, nil
}

// UdsTokenizerTestSuite holds the test suite state.
type UdsTokenizerTestSuite struct {
	suite.Suite
	mockServer *mockTokenizationServer
	socketPath string
	grpcServer *grpc.Server
	listener   net.Listener
	tokenizer  *UdsTokenizer // Shared tokenizer for most tests
}

// SetupSuite runs once before all tests in the suite.
func (s *UdsTokenizerTestSuite) SetupSuite() {
	s.mockServer = newMockTokenizationServer()

	tmpDir, err := os.MkdirTemp("", "tok-test-")
	require.NoError(s.T(), err)
	s.socketPath = filepath.Join(tmpDir, "tokenizer-uds.sock")

	// Create a Unix listener
	lc := net.ListenConfig{}
	listener, err := lc.Listen(context.Background(), "unix", s.socketPath)
	require.NoError(s.T(), err)
	s.listener = listener

	// Create and start the gRPC server
	s.grpcServer = grpc.NewServer()
	tokenizerpb.RegisterTokenizationServiceServer(s.grpcServer, s.mockServer)

	go func() {
		if err := s.grpcServer.Serve(s.listener); err != nil {
			s.T().Logf("Server error: %v", err)
		}
	}()

	config := &UdsTokenizerConfig{SocketFile: s.socketPath}
	s.tokenizer, err = NewUdsTokenizer(s.T().Context(), config, "test-model")
	require.NoError(s.T(), err)
	require.NotNil(s.T(), s.tokenizer)
}

// TearDownSuite runs once after all tests in the suite.
func (s *UdsTokenizerTestSuite) TearDownSuite() {
	// Close client connection first for graceful shutdown
	if s.tokenizer != nil {
		s.tokenizer.Close()
	}

	// Then stop the server
	if s.grpcServer != nil {
		s.grpcServer.Stop()
	}
	if s.listener != nil {
		s.listener.Close()
	}

	// Clean up the temp directory
	if s.socketPath != "" {
		os.RemoveAll(filepath.Dir(s.socketPath))
	}
}

// SetupTest runs before each test to reset mock state.
func (s *UdsTokenizerTestSuite) SetupTest() {
	// Reset error flags for each test
	s.mockServer.initializeError = false
	s.mockServer.tokenizeError = false
	s.mockServer.chatError = false
	s.mockServer.lastChatRequest = nil
	// Clear initialized models to ensure test isolation
	s.mockServer.initialized = make(map[string]bool)
	// Re-initialize the shared tokenizer's model
	if s.tokenizer != nil {
		s.mockServer.initialized["test-model"] = true
	}
}

func TestUdsTokenizerSuite(t *testing.T) {
	suite.Run(t, new(UdsTokenizerTestSuite))
}

func (s *UdsTokenizerTestSuite) TestNewUdsTokenizer_InitializationFailure() {
	s.mockServer.initializeError = true

	config := &UdsTokenizerConfig{
		SocketFile: s.socketPath,
	}

	tokenizer, err := NewUdsTokenizer(s.T().Context(), config, "test-model")
	s.Assert().Error(err)
	s.Assert().Nil(tokenizer)
}

func (s *UdsTokenizerTestSuite) TestNewUdsTokenizer_ConnectionFailure() {
	config := &UdsTokenizerConfig{
		SocketFile: "/non/existent/socket.sock",
	}

	// This should fail quickly because the socket doesn't exist
	tokenizer, err := NewUdsTokenizer(s.T().Context(), config, "test-model")
	s.Assert().Error(err)
	s.Assert().Nil(tokenizer)
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_WithModelTokenizerMap() {
	config := &UdsTokenizerConfig{
		SocketFile: s.socketPath,
		ModelTokenizerMap: map[string]string{
			"other-model": "/mnt/models/model-other",
			"test-model":  "/mnt/models/model-a/tokenizer.json",
		},
	}

	tokenizer, err := NewUdsTokenizer(s.T().Context(), config, "test-model")
	s.Require().NoError(err)
	s.Require().NotNil(tokenizer)
	s.Assert().True(s.mockServer.initialized["/mnt/models/model-a"])
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_ModelNotInMap() {
	config := &UdsTokenizerConfig{
		SocketFile: s.socketPath,
		ModelTokenizerMap: map[string]string{
			"test-model":  "/mnt/models/model-a",
			"other-model": "/mnt/models/model-other",
		},
	}

	tokenizer, err := NewUdsTokenizer(s.T().Context(), config, "unknown-model")
	s.Assert().Error(err)
	s.Assert().Nil(tokenizer)
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_Render() {
	input := "hello world"
	tokens, offsets, err := s.tokenizer.Render(input)
	s.Require().NoError(err)
	s.Assert().Equal(len([]rune(input)), len(tokens))
	s.Assert().Nil(offsets, "RenderCompletion does not return character offsets")

	// Verify specific characters (mock converts runes to token IDs)
	s.Assert().Equal(uint32('h'), tokens[0])
	s.Assert().Equal(uint32(' '), tokens[5])
	s.Assert().Equal(uint32('d'), tokens[10])
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_RenderChat() {
	renderReq := &types.RenderChatRequest{
		Conversation: []types.Conversation{
			{Role: "user", Content: types.Content{Raw: "Hello"}},
			{Role: "assistant", Content: types.Content{Raw: "Hi there"}},
		},
		AddGenerationPrompt: true,
		ChatTemplateKWArgs: map[string]interface{}{
			"key1": "value1",
			"key2": float64(42),
		},
	}

	tokens, _, err := s.tokenizer.RenderChat(renderReq)
	s.Require().NoError(err)
	s.Assert().Greater(len(tokens), 0, "should return tokens from rendered chat")
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_RenderChatPreservesToolCalls() {
	renderReq := &types.RenderChatRequest{
		Conversation: []types.Conversation{
			{Role: "user", Content: types.Content{Raw: "List files"}},
			{
				Role:    "assistant",
				Content: types.Content{Raw: "Reflection."},
				ToolCalls: []interface{}{
					map[string]interface{}{
						"id":   "chatcmpl-tool-1",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "bash",
							"arguments": `{"command":"ls -la"}`,
						},
					},
				},
			},
		},
	}

	_, _, err := s.tokenizer.RenderChat(renderReq)
	s.Require().NoError(err)
	s.Require().NotNil(s.mockServer.lastChatRequest)
	s.Require().Len(s.mockServer.lastChatRequest.Messages, 2)

	toolCallsJSON := s.mockServer.lastChatRequest.Messages[1].GetToolCallsJson()
	s.Require().NotEmpty(toolCallsJSON)
	s.Require().JSONEq(
		`[{"id":"chatcmpl-tool-1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls -la\"}"}}]`,
		toolCallsJSON,
	)
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_Type() {
	s.Assert().Equal("external-uds", s.tokenizer.Type())
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_TokenizeError() {
	s.mockServer.tokenizeError = true

	_, _, err := s.tokenizer.Render("test")
	s.Assert().Error(err)
	s.Assert().Contains(err.Error(), "render completion failed")
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_RenderChatTemplateError() {
	s.mockServer.chatError = true

	renderReq := &types.RenderChatRequest{
		Conversation: []types.Conversation{
			{Role: "user", Content: types.Content{Raw: "Hello"}},
		},
	}

	_, _, err := s.tokenizer.RenderChat(renderReq)
	s.Assert().Error(err)
	s.Assert().Contains(err.Error(), "render chat completion failed")
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_RenderChatWithMultiModalFeatures() {
	// Configure mock to return MM features.
	s.mockServer.mmFeatures = &tokenizerpb.MultiModalFeatures{
		MmHashes: map[string]*tokenizerpb.StringList{
			"image": {Values: []string{"hash_img1", "hash_img2"}},
		},
		MmPlaceholders: map[string]*tokenizerpb.PlaceholderRangeList{
			"image": {Ranges: []*tokenizerpb.PlaceholderRange{
				{Offset: 10, Length: 100},
				{Offset: 120, Length: 80},
			}},
		},
	}

	renderReq := &types.RenderChatRequest{
		Conversation: []types.Conversation{
			{Role: "user", Content: types.Content{Raw: "Describe these images"}},
		},
		AddGenerationPrompt: true,
	}

	tokens, features, err := s.tokenizer.RenderChat(renderReq)
	s.Require().NoError(err)
	s.Assert().Greater(len(tokens), 0)

	// Verify MM features are propagated.
	s.Require().NotNil(features, "multimodal features should be returned")
	s.Require().Contains(features.MMHashes, "image")
	s.Assert().Equal([]string{"hash_img1", "hash_img2"}, features.MMHashes["image"])

	s.Require().Contains(features.MMPlaceholders, "image")
	placeholders := features.MMPlaceholders["image"]
	s.Require().Len(placeholders, 2)
	s.Assert().Equal(10, placeholders[0].Offset)
	s.Assert().Equal(100, placeholders[0].Length)
	s.Assert().Equal(120, placeholders[1].Offset)
	s.Assert().Equal(80, placeholders[1].Length)
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_RenderChatTextOnlyNoFeatures() {
	// Default mock has no mmFeatures set.
	renderReq := &types.RenderChatRequest{
		Conversation: []types.Conversation{
			{Role: "user", Content: types.Content{Raw: "Hello"}},
		},
		AddGenerationPrompt: true,
	}

	_, features, err := s.tokenizer.RenderChat(renderReq)
	s.Require().NoError(err)
	s.Assert().Nil(features, "text-only request should have nil features")
}
