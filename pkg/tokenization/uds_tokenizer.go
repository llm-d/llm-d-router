/*
Copyright 2025 The llm-d Authors.

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

package tokenization

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"sigs.k8s.io/controller-runtime/pkg/log"

	tokenizerpb "github.com/llm-d/llm-d-kv-cache/api/tokenizerpb"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

// UdsTokenizerConfig represents the configuration for the UDS-based tokenizer,
// including the socket file path or TCP address (for testing only).
type UdsTokenizerConfig struct {
	SocketFile string `json:"socketFile"` // UDS socket path (production) or host:port for TCP (testing only)
	UseTCP     bool   `json:"useTCP"`     // If true, use TCP instead of UDS (for testing only, default: false)

	// ModelTokenizerMap maps a model name to the location of its tokenizer data.
	//
	// Each value may be either:
	//   1) A directory path that contains tokenizer files for the model (preferred), or
	//   2) A full path to a tokenizer.json file.
	//
	// Examples:
	//   {
	//     "model-a": "/mnt/models/model-a",                  // directory containing tokenizer.json, vocab, merges, etc.
	//     "model-b": "/opt/tokenizers/model-b/tokenizer.json" // explicit tokenizer.json path
	//   }
	ModelTokenizerMap map[string]string `json:"modelTokenizerMap,omitempty"`
}

func (cfg *UdsTokenizerConfig) IsEnabled() bool {
	return cfg != nil && cfg.SocketFile != ""
}

// UdsTokenizer communicates with a Unix Domain Socket server for tokenization.
// It implements the Tokenizer interface and manages a gRPC connection to the tokenizer service.
// The connection must be closed when the tokenizer is no longer needed by calling Close().
type UdsTokenizer struct {
	model  string
	conn   *grpc.ClientConn
	client tokenizerpb.TokenizationServiceClient
}

const (
	defaultSocketFile = "/tmp/tokenizer/tokenizer-uds.socket"

	// Default timeout for requests.
	defaultTimeout = 5 * time.Second
	// Timeout for multimodal requests (image download + processing).
	mmTimeout = 30 * time.Second
)

// NewUdsTokenizer creates a new UDS-based tokenizer client with connection pooling.
func NewUdsTokenizer(ctx context.Context, config *UdsTokenizerConfig, modelName string) (*UdsTokenizer, error) {
	socketFile := config.SocketFile
	if socketFile == "" {
		socketFile = defaultSocketFile
	}

	resolvedModel := modelName
	if config.ModelTokenizerMap != nil { //nolint:nestif // simple model path resolution logic
		if path, ok := config.ModelTokenizerMap[modelName]; ok {
			if filepath.Base(path) == "tokenizer.json" { // explicit tokenizer.json path — use its parent directory
				resolvedModel = filepath.Dir(path)
			} else {
				resolvedModel = path
			}
		} else {
			return nil, fmt.Errorf("tokenizer for model %q not found", modelName)
		}
	}

	// Determine address based on UseTCP flag
	var address string
	if config.UseTCP {
		// TCP address (for testing only)
		address = socketFile
	} else {
		// UDS socket path (production default)
		address = fmt.Sprintf("unix://%s", socketFile)
	}

	// Create gRPC connection
	conn, err := grpc.NewClient(
		address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                10 * time.Minute,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(100<<20), // 100MB
			grpc.MaxCallRecvMsgSize(100<<20), // 100MB
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC connection: %w", err)
	}

	client := tokenizerpb.NewTokenizationServiceClient(conn)

	udsTokenizer := &UdsTokenizer{
		conn:   conn,
		client: client,
		model:  resolvedModel,
	}

	// Start a goroutine to monitor the context and close the connection when the context ends
	go func() {
		<-ctx.Done()
		udsTokenizer.Close()
	}()

	// Initialize the tokenizer for the specified model
	if err := udsTokenizer.initializeTokenizerForModel(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize tokenizer for model %s: %w", modelName, err)
	}

	// Warm up the renderer with a minimal request to force any lazy
	// downloads (e.g. image processor configs for multimodal models).
	udsTokenizer.warmup(ctx)

	return udsTokenizer, nil
}

// initializeTokenizerForModel initializes the tokenizer service for a specific model.
func (u *UdsTokenizer) initializeTokenizerForModel(ctx context.Context) error {
	// Use default configuration values for now
	req := &tokenizerpb.InitializeTokenizerRequest{
		ModelName:           u.model,
		EnableThinking:      false, // Can be made configurable later
		AddGenerationPrompt: true,  // Can be made configurable later
	}

	// Retry logic with exponential backoff
	const maxRetries = 5
	const baseDelay = time.Second

	var lastErr error
	for i := 0; i < maxRetries; i++ {
		if i > 0 {
			delay := time.Duration(i) * baseDelay
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return ctx.Err()
			}
		}

		resp, err := u.client.InitializeTokenizer(ctx, req)
		if err != nil {
			lastErr = fmt.Errorf("gRPC InitializeTokenizer request failed: %w", err)
			continue
		}

		if !resp.Success {
			lastErr = fmt.Errorf("tokenizer initialization failed: %s", resp.ErrorMessage)
			continue
		}

		// Success
		return nil
	}

	return fmt.Errorf("tokenizer initialization failed after %d attempts: %w", maxRetries, lastErr)
}

// warmup sends a minimal text request to force any lazy downloads in the
// renderer (e.g. image processor configs for multimodal models). Failures
// are logged but not fatal — the first real request will retry.
func (u *UdsTokenizer) warmup(ctx context.Context) {
	warmupCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	addGen := true
	_, err := u.client.RenderChatCompletion(warmupCtx, &tokenizerpb.RenderChatCompletionRequest{
		ModelName: u.model,
		Messages: []*tokenizerpb.ChatMessage{{
			Role:    "user",
			Content: strPtr("warmup"),
		}},
		AddGenerationPrompt: &addGen,
	})
	if err != nil {
		log.FromContext(ctx).V(logging.DEBUG).Info("Renderer warmup failed (non-critical)", "model", u.model, "error", err)
	}
}

func strPtr(s string) *string { return &s }

// Render tokenizes a plain-text prompt via the UDS renderer service.
func (u *UdsTokenizer) Render(prompt string) ([]uint32, []types.Offset, error) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	resp, err := u.client.RenderCompletion(ctx, &tokenizerpb.RenderCompletionRequest{
		ModelName: u.model,
		Prompt:    prompt,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC RenderCompletion request failed: %w", err)
	}

	if !resp.Success {
		return nil, nil, fmt.Errorf("render completion failed: %s", resp.ErrorMessage)
	}

	return resp.TokenIds, nil, nil
}

// Encode tokenizes the input string and returns the token IDs and offsets.
func (u *UdsTokenizer) Encode(prompt string, addSpecialTokens bool) ([]uint32, []types.Offset, error) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	pbReq := &tokenizerpb.TokenizeRequest{
		Input:            prompt,
		ModelName:        u.model,
		AddSpecialTokens: addSpecialTokens,
	}

	resp, err := u.client.Tokenize(ctx, pbReq)
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC tokenize request failed: %w", err)
	}

	if !resp.Success {
		return nil, nil, fmt.Errorf("tokenization failed: %s", resp.ErrorMessage)
	}

	// Use offset_pairs field in format [start, end, start, end, ...]
	var tokenizersOffsets []types.Offset

	if len(resp.OffsetPairs) > 0 && len(resp.OffsetPairs)%2 == 0 {
		// Use offset_pairs field in format [start, end, start, end, ...]
		pairCount := len(resp.OffsetPairs) / 2
		tokenizersOffsets = make([]types.Offset, pairCount)
		for i := 0; i < pairCount; i++ {
			start := resp.OffsetPairs[2*i]
			end := resp.OffsetPairs[2*i+1]
			tokenizersOffsets[i] = types.Offset{uint(start), uint(end)}
		}
	} else {
		return nil, nil, fmt.Errorf("invalid offset_pairs field in response")
	}

	return resp.InputIds, tokenizersOffsets, nil
}

// RenderChat renders a chat completion request using the UDS renderer service.
// It calls the RenderChatCompletion RPC which runs vLLM's OpenAIServingRender
// on the CPU, returning token IDs and optional multimodal features.
func (u *UdsTokenizer) RenderChat(
	renderReq *types.RenderChatRequest,
) ([]uint32, *MultiModalFeatures, error) {
	timeout := defaultTimeout
	for _, msg := range renderReq.Conversation {
		if len(msg.Content.Structured) > 0 {
			timeout = mmTimeout
			break
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Convert conversation messages to proto format
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("UdsTokenizer.RenderChat")
	messages := make([]*tokenizerpb.ChatMessage, 0, len(renderReq.Conversation))
	for _, msg := range renderReq.Conversation {
		pbMsg := &tokenizerpb.ChatMessage{Role: msg.Role}
		if len(msg.ToolCalls) > 0 {
			b, err := json.Marshal(msg.ToolCalls)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to marshal message tool_calls: %w", err)
			}
			s := string(b)
			pbMsg.ToolCallsJson = &s
		}
		if len(msg.Content.Structured) > 0 {
			parts := make([]*tokenizerpb.ContentPart, 0, len(msg.Content.Structured))
			for _, block := range msg.Content.Structured {
				part := &tokenizerpb.ContentPart{Type: block.Type}
				switch block.Type {
				case "text":
					text := block.Text
					part.Text = &text
				case "image_url":
					part.ImageUrl = &tokenizerpb.ImageUrl{Url: block.ImageURL.URL}
				default:
					traceLogger.Info("dropping unsupported chat message content block type, it will not be rendered", "type", block.Type)
					continue
				}
				parts = append(parts, part)
			}
			pbMsg.ContentParts = parts
		} else {
			content := msg.Content.Raw
			pbMsg.Content = &content
		}
		messages = append(messages, pbMsg)
	}

	// Convert tools to JSON string
	var toolsJSON *string
	if len(renderReq.Tools) > 0 {
		b, err := json.Marshal(renderReq.Tools)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal tools: %w", err)
		}
		s := string(b)
		toolsJSON = &s
	}

	// Convert ChatTemplateKWArgs to JSON string
	var chatTemplateKwargsJSON *string
	if len(renderReq.ChatTemplateKWArgs) > 0 {
		b, err := json.Marshal(renderReq.ChatTemplateKWArgs)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal chat_template_kwargs: %w", err)
		}
		s := string(b)
		chatTemplateKwargsJSON = &s
	}

	resp, err := u.client.RenderChatCompletion(ctx, &tokenizerpb.RenderChatCompletionRequest{
		ModelName:            u.model,
		Messages:             messages,
		ToolsJson:            toolsJSON,
		ChatTemplate:         renderReq.ChatTemplate,
		AddGenerationPrompt:  &renderReq.AddGenerationPrompt,
		ContinueFinalMessage: renderReq.ContinueFinalMessage,
		ChatTemplateKwargs:   chatTemplateKwargsJSON,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC RenderChatCompletion request failed: %w", err)
	}

	if !resp.Success {
		return nil, nil, fmt.Errorf("render chat completion failed: %s", resp.ErrorMessage)
	}

	var features *MultiModalFeatures
	if resp.Features != nil {
		features = convertProtoFeatures(resp.Features)
	}

	return resp.TokenIds, features, nil
}

// convertProtoFeatures converts proto MultiModalFeatures to domain type.
func convertProtoFeatures(pf *tokenizerpb.MultiModalFeatures) *MultiModalFeatures {
	if pf == nil {
		return nil
	}

	features := &MultiModalFeatures{
		MMHashes:       make(map[string][]string, len(pf.MmHashes)),
		MMPlaceholders: make(map[string][]kvblock.PlaceholderRange, len(pf.MmPlaceholders)),
	}

	for modality, sl := range pf.MmHashes {
		features.MMHashes[modality] = sl.Values
	}

	for modality, pl := range pf.MmPlaceholders {
		ranges := make([]kvblock.PlaceholderRange, len(pl.Ranges))
		for i, r := range pl.Ranges {
			ranges[i] = kvblock.PlaceholderRange{
				Offset: int(r.Offset),
				Length: int(r.Length),
			}
		}
		features.MMPlaceholders[modality] = ranges
	}

	return features
}

func (u *UdsTokenizer) Type() string {
	return "external-uds"
}

// Close closes the underlying gRPC connection to the tokenizer service.
func (u *UdsTokenizer) Close() error {
	if u.conn != nil {
		return u.conn.Close()
	}
	return nil
}
