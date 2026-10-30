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

package requesthandling

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func chatText(role, text string) Message {
	return Message{Role: role, Content: Content{Raw: text}}
}

func chatBody(messages ...Message) *InferenceRequestBody {
	return &InferenceRequestBody{ChatCompletions: &ChatCompletionsRequest{Messages: messages}}
}

func anthropicText(role, text string) AnthropicMessage {
	return AnthropicMessage{Role: role, Content: AnthropicContent{
		Structured: []AnthropicContentBlock{{Type: "text", Text: text}},
	}}
}

func anthropicBody(system string, messages ...AnthropicMessage) *InferenceRequestBody {
	return &InferenceRequestBody{Messages: &MessagesRequest{System: AnthropicContent{Raw: system}, Messages: messages}}
}

// responsesBody decodes a Responses request body the way the OpenAI parser does.
func responsesBody(t *testing.T, raw string) *InferenceRequestBody {
	t.Helper()
	var responses ResponsesRequest
	require.NoError(t, json.Unmarshal([]byte(raw), &responses))
	return &InferenceRequestBody{Responses: &responses}
}

func conversationPrefix(t *testing.T, body *InferenceRequestBody) string {
	t.Helper()
	var buf bytes.Buffer
	require.True(t, body.WriteConversationPrefix(&buf), "expected a conversation prefix")
	return buf.String()
}

// TestWriteConversationPrefixBytes pins the written bytes: consumers hash them,
// and replicas running different versions must derive the same key.
func TestWriteConversationPrefixBytes(t *testing.T) {
	tests := []struct {
		name string
		body *InferenceRequestBody
		want string
	}{
		{
			name: "chat completions",
			body: &InferenceRequestBody{ChatCompletions: &ChatCompletionsRequest{
				CacheSalt: "salt",
				Messages:  []Message{chatText("system", "sys"), chatText("user", "hi"), chatText("assistant", "ignored")},
			}},
			want: "salt\x00system\x00sys\x00user\x00hi\x00",
		},
		{
			name: "anthropic messages",
			body: &InferenceRequestBody{Messages: &MessagesRequest{
				CacheSalt: "salt",
				System:    AnthropicContent{Raw: "sys"},
				Messages:  []AnthropicMessage{anthropicText("user", "hi"), anthropicText("assistant", "ignored")},
			}},
			want: "salt\x00sys\x00user\x00hi\x00",
		},
		{
			name: "responses items",
			body: responsesBody(t, `{"cache_salt":"salt","instructions":"sys","input":[
				{"role":"developer","content":"dev"},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"h"},{"type":"input_text","text":"i"}]},
				{"role":"assistant","content":"ignored"}]}`),
			want: "salt\x00sys\x00developer\x00dev\x00user\x00hi\x00",
		},
		{
			name: "responses string input",
			body: responsesBody(t, `{"input":"hi"}`),
			want: "\x00\x00user\x00hi\x00",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.want, conversationPrefix(t, test.body))
		})
	}
}

func TestWriteConversationPrefixStableAcrossTurns(t *testing.T) {
	tests := []struct {
		name      string
		firstTurn *InferenceRequestBody
		laterTurn *InferenceRequestBody
		other     *InferenceRequestBody
	}{
		{
			name:      "chat completions",
			firstTurn: chatBody(chatText("system", "sys"), chatText("user", "question")),
			laterTurn: chatBody(chatText("system", "sys"), chatText("user", "question"), chatText("assistant", "answer"), chatText("user", "more")),
			other:     chatBody(chatText("system", "sys"), chatText("user", "different question")),
		},
		{
			name:      "chat completions structured text",
			firstTurn: chatBody(chatText("user", "question")),
			laterTurn: chatBody(Message{Role: "user", Content: Content{Structured: []ContentBlock{{Type: "text", Text: "question"}}}}, chatText("assistant", "answer")),
			other:     chatBody(chatText("user", "different question")),
		},
		{
			name:      "anthropic messages",
			firstTurn: anthropicBody("sys", anthropicText("user", "question")),
			laterTurn: anthropicBody("sys", anthropicText("user", "question"), anthropicText("assistant", "answer"), anthropicText("user", "more")),
			other:     anthropicBody("sys", anthropicText("user", "different question")),
		},
		{
			name:      "responses",
			firstTurn: responsesBody(t, `{"instructions":"sys","input":"question"}`),
			laterTurn: responsesBody(t, `{"instructions":"sys","input":[
				{"role":"user","content":[{"type":"input_text","text":"question"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
				{"role":"user","content":"more"}]}`),
			other: responsesBody(t, `{"instructions":"sys","input":"different question"}`),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			first := conversationPrefix(t, test.firstTurn)
			assert.Equal(t, first, conversationPrefix(t, test.laterTurn), "later turn must repeat the first turn's prefix")
			assert.NotEqual(t, first, conversationPrefix(t, test.other), "sessions sharing only a system prompt must differ")
		})
	}
}

func TestWriteConversationPrefixCacheSalt(t *testing.T) {
	unsalted := conversationPrefix(t, chatBody(chatText("user", "hi")))
	salted := conversationPrefix(t, &InferenceRequestBody{ChatCompletions: &ChatCompletionsRequest{
		CacheSalt: "tenant-a",
		Messages:  []Message{chatText("user", "hi")},
	}})
	assert.NotEqual(t, unsalted, salted)
}

func TestWriteConversationPrefixUnkeyed(t *testing.T) {
	image := json.RawMessage(`{"type":"url","url":"https://example.com/a.jpg"}`)
	tests := []struct {
		name string
		body *InferenceRequestBody
	}{
		{"nil body", nil},
		{"completions", &InferenceRequestBody{Completions: &CompletionsRequest{Prompt: Prompt{Raw: "hi"}}}},
		{"generate", &InferenceRequestBody{Generate: &GenerateRequest{Text: "hi"}}},
		{"embeddings", &InferenceRequestBody{Embeddings: &EmbeddingsRequest{Input: EmbeddingsInput{Raw: "hi"}}}},
		{"conversations", &InferenceRequestBody{Conversations: &ConversationsRequest{
			Items: []ConversationItem{{Type: "message", Role: "user", Content: "hi"}},
		}}},
		{"chat no messages", chatBody()},
		{"chat system without user", chatBody(chatText("system", "preamble"))},
		{"chat empty user text", chatBody(chatText("user", ""))},
		{"chat empty structured blocks", chatBody(Message{Role: "user", Content: Content{Structured: []ContentBlock{}}})},
		{"chat text and image", chatBody(Message{Role: "user", Content: Content{Structured: []ContentBlock{
			{Type: "text", Text: "What is in this image?"},
			{Type: "image_url", ImageURL: ImageBlock{URL: "https://example.com/a.jpg"}},
		}}})},
		{"chat audio block", chatBody(Message{Role: "user", Content: Content{Structured: []ContentBlock{
			{Type: "audio_url", AudioURL: AudioURLBlock{URL: "https://example.com/a.wav"}},
		}}})},
		{"chat tool calls before user", chatBody(
			Message{Role: "assistant", ToolCalls: []any{map[string]any{"id": "call_1"}}},
			chatText("user", "next"),
		)},
		{"anthropic no messages", anthropicBody("sys")},
		{"anthropic text and image", &InferenceRequestBody{Messages: &MessagesRequest{
			Messages: []AnthropicMessage{{Role: "user", Content: AnthropicContent{Structured: []AnthropicContentBlock{
				{Type: "text", Text: "What is in this image?"},
				{Type: "image", Source: image},
			}}}},
		}}},
		{"anthropic non-text system block", &InferenceRequestBody{Messages: &MessagesRequest{
			System:   AnthropicContent{Structured: []AnthropicContentBlock{{Type: "image", Source: image}}},
			Messages: []AnthropicMessage{{Role: "user", Content: AnthropicContent{Raw: "hi"}}},
		}}},
		{"responses previous_response_id", responsesBody(t, `{"previous_response_id":"resp_1","input":"hi"}`)},
		{"responses conversation", responsesBody(t, `{"conversation":"conv_1","input":"hi"}`)},
		{"responses empty string input", responsesBody(t, `{"input":""}`)},
		{"responses empty items", responsesBody(t, `{"input":[]}`)},
		{"responses non-string instructions", responsesBody(t, `{"instructions":[{"role":"system","content":"sys"}],"input":"hi"}`)},
		{"responses system without user", responsesBody(t, `{"input":[{"role":"system","content":"sys"}]}`)},
		{"responses item without role", responsesBody(t, `{"input":[{"content":"hi"}]}`)},
		{"responses function call before user", responsesBody(t, `{"input":[
			{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},
			{"role":"user","content":"hi"}]}`)},
		{"responses text and image", responsesBody(t, `{"input":[{"role":"user","content":[
			{"type":"input_text","text":"What is in this image?"},
			{"type":"input_image","image_url":"https://example.com/a.jpg"}]}]}`)},
		{"responses text part without text", responsesBody(t, `{"input":[{"role":"user","content":[{"type":"input_text"}]}]}`)},
		{"responses empty user content parts", responsesBody(t, `{"input":[{"role":"user","content":[]}]}`)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var buf bytes.Buffer
			assert.False(t, test.body.WriteConversationPrefix(&buf))
		})
	}
}

// TestEveryRequestTypeDecidesConversationPrefix fails when InferenceRequestBody
// gains a request type that neither writes a conversation prefix nor is listed
// as lacking one.
func TestEveryRequestTypeDecidesConversationPrefix(t *testing.T) {
	keyed := map[string]*InferenceRequestBody{
		"ChatCompletions": chatBody(chatText("user", "hi")),
		"Messages":        anthropicBody("", anthropicText("user", "hi")),
		"Responses":       responsesBody(t, `{"input":"hi"}`),
	}
	unkeyed := map[string]string{
		"Completions":   "no turn boundary",
		"Generate":      "no turn boundary",
		"Embeddings":    "single-shot",
		"TextToSpeech":  "single-shot",
		"Images":        "single-shot",
		"Conversations": "creates a conversation object; no generation",
	}
	prefixer := reflect.TypeFor[interface {
		WriteConversationPrefix(PrefixWriter) bool
	}]()

	bodyType := reflect.TypeFor[InferenceRequestBody]()
	for i := range bodyType.NumField() {
		field := bodyType.Field(i)
		if field.Type.Kind() != reflect.Pointer || field.Type.Elem().Kind() != reflect.Struct ||
			field.Type == reflect.TypeFor[*TokenizedRequest]() {
			continue
		}
		t.Run(field.Name, func(t *testing.T) {
			body, isKeyed := keyed[field.Name]
			_, isUnkeyed := unkeyed[field.Name]
			require.True(t, isKeyed != isUnkeyed, "list %s in exactly one of keyed or unkeyed", field.Name)
			if isKeyed {
				require.True(t, field.Type.Implements(prefixer), "%s must implement WriteConversationPrefix", field.Type)
				require.NotNil(t, reflect.ValueOf(body).Elem().Field(i).Interface(), "sample body must populate %s", field.Name)
				var buf bytes.Buffer
				assert.True(t, body.WriteConversationPrefix(&buf), "InferenceRequestBody must dispatch to %s", field.Name)
				return
			}
			body = &InferenceRequestBody{}
			reflect.ValueOf(body).Elem().Field(i).Set(reflect.New(field.Type.Elem()))
			var buf bytes.Buffer
			assert.False(t, body.WriteConversationPrefix(&buf))
		})
	}
}
