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

package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"math/rand/v2"
	"strings"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

// benchBytesPerToken approximates English text for sizing input sequence length (ISL).
const benchBytesPerToken = 4

var (
	benchmarkChatParseResult *fwkrh.ParseResult
	benchmarkChatBody        *fwkrh.InferenceRequestBody
	benchmarkChatEnvelope    map[string]any
)

// benchText returns n bytes of ASCII prose containing quotes, newlines, and tabs
// so the decoder exercises string unescaping.
func benchText(n int) string {
	const sentence = "The router said \"send it to the least loaded pod\" and the pod replied.\n\tQueue depth: 12, KV cache: 0.43. "
	var sb strings.Builder
	sb.Grow(n + len(sentence))
	for sb.Len() < n {
		sb.WriteString(sentence)
	}
	return sb.String()[:n]
}

func benchBase64(rawBytes int) string {
	raw := make([]byte, rawBytes)
	r := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // deterministic benchmark data
	for i := range raw {
		raw[i] = byte(r.Uint32())
	}
	return base64.StdEncoding.EncodeToString(raw)
}

// benchConversation spreads islTokens of text across turns messages. Multi-turn
// conversations start with a system prompt, alternate user and assistant, and end on user.
func benchConversation(islTokens, turns int) []any {
	perTurn := islTokens * benchBytesPerToken / turns
	messages := make([]any, 0, turns)
	for i := range turns {
		role := "user"
		switch {
		case turns > 1 && i == 0:
			role = "system"
		case i%2 == 0:
			role = "assistant"
		}
		messages = append(messages, map[string]any{"role": role, "content": benchText(perTurn)})
	}
	return messages
}

// benchAgenticConversation is a system prompt and user turn followed by toolRounds
// assistant tool calls, each answered by a tool result.
func benchAgenticConversation(islTokens, toolRounds int) []any {
	messages := benchConversation(islTokens/2, 2)
	perResult := islTokens * benchBytesPerToken / 2 / toolRounds
	for i := range toolRounds {
		id := fmt.Sprintf("call_%d", i)
		messages = append(messages,
			map[string]any{
				"role":    "assistant",
				"content": nil,
				"tool_calls": []any{map[string]any{
					"id":   id,
					"type": "function",
					"function": map[string]any{
						"name":      fmt.Sprintf("tool_%d", i),
						"arguments": `{"query":"pods with the lowest queue depth","limit":10}`,
					},
				}},
			},
			map[string]any{"role": "tool", "tool_call_id": id, "content": benchText(perResult)},
		)
	}
	return messages
}

func benchMultimodalMessage(text string, media ...any) map[string]any {
	blocks := append([]any{map[string]any{"type": "text", "text": text}}, media...)
	return map[string]any{"role": "user", "content": blocks}
}

func benchImageData(rawBytes int) any {
	return map[string]any{
		"type":      "image_url",
		"image_url": map[string]any{"url": "data:image/png;base64," + benchBase64(rawBytes)},
	}
}

func benchImageURL(i int) any {
	return map[string]any{
		"type":      "image_url",
		"image_url": map[string]any{"url": fmt.Sprintf("https://images.example.com/datasets/eval/%06d.png", i)},
	}
}

func benchAudioData(rawBytes int) any {
	return map[string]any{
		"type":        "input_audio",
		"input_audio": map[string]any{"data": benchBase64(rawBytes), "format": "wav"},
	}
}

func benchTools(n int) []any {
	tools := make([]any, n)
	for i := range n {
		tools[i] = map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        fmt.Sprintf("tool_%d", i),
				"description": benchText(200),
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query":   map[string]any{"type": "string", "description": benchText(80)},
						"limit":   map[string]any{"type": "integer", "minimum": 1, "maximum": 100},
						"filters": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
					},
					"required": []any{"query"},
				},
			},
		}
	}
	return tools
}

func benchChatBody(b *testing.B, messages []any, extra map[string]any) []byte {
	b.Helper()
	body := map[string]any{
		"model":                 "meta-llama/Llama-3.1-8B-Instruct",
		"messages":              messages,
		"stream":                true,
		"max_completion_tokens": 1024,
		"temperature":           0.7,
	}
	maps.Copy(body, extra)
	data, err := json.Marshal(body)
	if err != nil {
		b.Fatal(err)
	}
	return data
}

// BenchmarkOpenAIParser_ChatCompletions measures /v1/chat/completions parsing across
// text-only, multi-turn, agentic, and multimodal workloads. Each workload reports the
// full ParseRequest and its two decode stages: the typed body and the routing envelope.
//
// Run:
//
//	GOTOOLCHAIN=go1.26.8 go test -run='^$' -bench=BenchmarkOpenAIParser_ChatCompletions \
//	    -benchmem -count=10 ./pkg/epp/framework/plugins/requesthandling/parsers/openai/ | tee bench.out
//	benchstat bench.out
func BenchmarkOpenAIParser_ChatCompletions(b *testing.B) {
	const shortPrompt = 1024 * benchBytesPerToken
	workloads := []struct {
		name string
		body []byte
	}{
		{name: "Text/ISL1K", body: benchChatBody(b, benchConversation(1024, 1), nil)},
		{name: "Text/ISL8K", body: benchChatBody(b, benchConversation(8*1024, 1), nil)},
		{name: "Text/ISL32K", body: benchChatBody(b, benchConversation(32*1024, 1), nil)},
		{name: "Text/ISL128K", body: benchChatBody(b, benchConversation(128*1024, 1), nil)},
		{name: "Text/ISL32K-32turns", body: benchChatBody(b, benchConversation(32*1024, 32), nil)},
		{name: "Text/ISL128K-64turns", body: benchChatBody(b, benchConversation(128*1024, 64), nil)},
		{name: "Agentic/ISL32K-20tools-8calls", body: benchChatBody(b, benchAgenticConversation(32*1024, 8),
			map[string]any{"tools": benchTools(20)})},
		{name: "MM/ImageURLx4-ISL1K", body: benchChatBody(b, []any{benchMultimodalMessage(benchText(shortPrompt),
			benchImageURL(0), benchImageURL(1), benchImageURL(2), benchImageURL(3))}, nil)},
		{name: "MM/Image256KiBx1-ISL1K", body: benchChatBody(b, []any{benchMultimodalMessage(benchText(shortPrompt),
			benchImageData(256*1024))}, nil)},
		{name: "MM/Image1MiBx4-ISL8K", body: benchChatBody(b, []any{benchMultimodalMessage(benchText(8*1024*benchBytesPerToken),
			benchImageData(1024*1024), benchImageData(1024*1024), benchImageData(1024*1024), benchImageData(1024*1024))}, nil)},
		{name: "MM/Audio2MiBx1-ISL1K", body: benchChatBody(b, []any{benchMultimodalMessage(benchText(shortPrompt),
			benchAudioData(2*1024*1024))}, nil)},
		{name: "Mixed/ISL32K-16turns-Image512KiBx2-20tools", body: benchChatBody(b,
			append(benchConversation(32*1024, 16), benchMultimodalMessage(benchText(shortPrompt),
				benchImageData(512*1024), benchImageData(512*1024))),
			map[string]any{"tools": benchTools(20)})},
	}

	parser := NewOpenAIParser()
	headers := map[string]string{":path": "/v1/chat/completions"}
	ctx := context.Background()
	stages := []struct {
		name string
		run  func(body []byte) error
	}{
		{name: "ParseRequest", run: func(body []byte) error {
			result, err := parser.ParseRequest(ctx, body, headers)
			benchmarkChatParseResult = result
			return err
		}},
		{name: "TypedBody", run: func(body []byte) error {
			result, err := extractRequestBody(chatCompletionsAPI, body)
			benchmarkChatBody = result
			return err
		}},
		{name: "Envelope", run: func(body []byte) error {
			result, err := parserutil.UnmarshalEnvelope(body, promptField)
			benchmarkChatEnvelope = result
			return err
		}},
	}

	for _, wl := range workloads {
		b.Run(wl.name, func(b *testing.B) {
			for _, stage := range stages {
				b.Run(stage.name, func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(int64(len(wl.body)))
					for b.Loop() {
						if err := stage.run(wl.body); err != nil {
							b.Fatal(err)
						}
					}
					b.ReportMetric(float64(len(wl.body))/1024, "body-KiB")
				})
			}
		})
	}
}
