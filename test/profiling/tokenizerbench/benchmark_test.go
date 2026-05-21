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

// Package external_tokenizer_scorer benchmarks the end-to-end request flow
// through the EPP when using the external tokenizer DataProducer plugin
// combined with the precise-prefix-cache-scorer.
//
// Prerequisites:
//   - A kind cluster with the EPP deployed using the external tokenizer config.
//   - Gateway reachable on localhost (default: port 30080).
//
// Run:
//
//	make bench-tokenizer
//
// Or manually:
//
//	EXTERNAL_TOKENIZER_ENABLED=true KV_CACHE_ENABLED=true make env-dev-kind
//	MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0" go test -bench=. -benchmem -count=5 -timeout=5m ./test/profiling/tokenizerbench/
package tokenizerbench

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

var (
	baseURL string
	model   string
	client  openai.Client
)

func TestMain(m *testing.M) {
	port := envOrDefault("E2E_PORT", "30080")
	baseURL = fmt.Sprintf("http://localhost:%s/v1", port)
	model = envOrDefault("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

	client = openai.NewClient(option.WithBaseURL(baseURL))

	// Wait for the EPP to be reachable before running any benchmarks.
	if err := waitForReady(30 * time.Second); err != nil {
		fmt.Fprintf(os.Stderr, "EPP not reachable at %s: %v\n", baseURL, err)
		os.Exit(1)
	}

	os.Exit(m.Run())
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func waitForReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	httpClient := &http.Client{Timeout: 2 * time.Second}

	for time.Now().Before(deadline) {
		resp, err := httpClient.Get(baseURL + "/models")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("timeout after %s", timeout)
}

// --- Prompts ---

const shortPrompt = "What is the capital of France?"

const longPrompt = `You are an expert food critic. You have been asked to review the following restaurant.
Please provide a detailed analysis of the food quality, service, ambiance, and overall experience.
Consider the menu variety, ingredient freshness, presentation, and value for money.
Your review should be comprehensive and cover all aspects of the dining experience.
The restaurant is located in downtown San Francisco and specializes in modern California cuisine
with influences from Japanese and Mediterranean cooking traditions. The chef has over 20 years
of experience working in Michelin-starred restaurants across Europe and Asia. The menu changes
seasonally to reflect the freshest local ingredients available from nearby farms and fisheries.`

var sharedPrefixSuffixes = []string{
	" What appetizer would you recommend?",
	" How is the wine selection?",
	" What about the dessert menu?",
	" Is there outdoor seating?",
	" What are the prices like?",
}

var multiTurnMessages = []openai.ChatCompletionMessageParamUnion{
	openai.SystemMessage(strings.Repeat("You are a helpful assistant that provides detailed food reviews. ", 20)),
	openai.UserMessage("I visited a new Italian restaurant last night. The pasta was excellent."),
	openai.AssistantMessage("That sounds wonderful! Italian cuisine is one of my favorites. Could you tell me more about the specific pasta dishes you tried?"),
	openai.UserMessage("I had the truffle carbonara and the lobster linguine. Both were outstanding."),
}

// --- Benchmarks ---

func BenchmarkCompletion_ShortPrompt(b *testing.B) {
	benchCompletion(b, shortPrompt)
}

func BenchmarkCompletion_LongPrompt(b *testing.B) {
	benchCompletion(b, longPrompt)
}

func BenchmarkChatCompletion_ShortPrompt(b *testing.B) {
	benchChatCompletion(b, shortPrompt)
}

func BenchmarkChatCompletion_LongPrompt(b *testing.B) {
	benchChatCompletion(b, longPrompt)
}

// BenchmarkCompletion_SharedPrefix sends repeated requests that share a long
// prefix but vary their suffix, exercising prefix cache affinity.
func BenchmarkCompletion_SharedPrefix(b *testing.B) {
	// Warmup: seed the prefix cache and report token count.
	prompt0 := longPrompt + sharedPrefixSuffixes[0]
	resp, err := client.Completions.New(context.Background(), completionParams(prompt0))
	if err != nil {
		b.Fatalf("warmup: %v", err)
	}
	promptTokens := resp.Usage.PromptTokens
	b.Logf("prompt_tokens=%d prefix_chars=%d suffixes=%d", promptTokens, len(longPrompt), len(sharedPrefixSuffixes))

	b.ResetTimer()
	for i := range b.N {
		prompt := longPrompt + sharedPrefixSuffixes[i%len(sharedPrefixSuffixes)]
		if _, err := client.Completions.New(context.Background(), completionParams(prompt)); err != nil {
			b.Fatalf("iter %d: %v", i, err)
		}
	}
	b.ReportMetric(float64(promptTokens), "prompt_tokens")
}

// BenchmarkChatCompletion_MultiMessage exercises the RenderChatTemplate + Tokenize
// gRPC path with a multi-turn conversation.
func BenchmarkChatCompletion_MultiMessage(b *testing.B) {
	params := chatCompletionParams(multiTurnMessages)

	// Warmup and report token count.
	resp, err := client.Chat.Completions.New(context.Background(), params)
	if err != nil {
		b.Fatalf("warmup: %v", err)
	}
	promptTokens := resp.Usage.PromptTokens
	b.Logf("prompt_tokens=%d messages=%d", promptTokens, len(multiTurnMessages))

	b.ResetTimer()
	for i := range b.N {
		if _, err := client.Chat.Completions.New(context.Background(), params); err != nil {
			b.Fatalf("iter %d: %v", i, err)
		}
	}
	b.ReportMetric(float64(promptTokens), "prompt_tokens")
}

// --- Helpers ---

func benchCompletion(b *testing.B, prompt string) {
	b.Helper()
	params := completionParams(prompt)

	// Warmup to get token count.
	resp, err := client.Completions.New(context.Background(), params)
	if err != nil {
		b.Fatalf("warmup: %v", err)
	}
	promptTokens := resp.Usage.PromptTokens
	b.Logf("prompt_tokens=%d prompt_chars=%d", promptTokens, len(prompt))

	b.ResetTimer()
	for i := range b.N {
		if _, err := client.Completions.New(context.Background(), params); err != nil {
			b.Fatalf("iter %d: %v", i, err)
		}
	}
	b.ReportMetric(float64(promptTokens), "prompt_tokens")
}

func benchChatCompletion(b *testing.B, prompt string) {
	b.Helper()

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(prompt),
	}
	params := chatCompletionParams(messages)

	// Warmup to get token count.
	resp, err := client.Chat.Completions.New(context.Background(), params)
	if err != nil {
		b.Fatalf("warmup: %v", err)
	}
	promptTokens := resp.Usage.PromptTokens
	b.Logf("prompt_tokens=%d prompt_chars=%d", promptTokens, len(prompt))

	b.ResetTimer()
	for i := range b.N {
		if _, err := client.Chat.Completions.New(context.Background(), params); err != nil {
			b.Fatalf("iter %d: %v", i, err)
		}
	}
	b.ReportMetric(float64(promptTokens), "prompt_tokens")
}

func completionParams(prompt string) openai.CompletionNewParams {
	return openai.CompletionNewParams{
		Prompt: openai.CompletionNewParamsPromptUnion{
			OfString: openai.String(prompt),
		},
		Model: openai.CompletionNewParamsModel(model),
	}
}

func chatCompletionParams(messages []openai.ChatCompletionMessageParamUnion) openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    model,
	}
}
