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

package request

import (
	"encoding/base64"
	"encoding/json"
	"maps"
	"reflect"
	"slices"
	"testing"
)

// Base64DecodedLen must be exact, not an estimate: the coordinator bounds an
// inline media payload with it, so a size read even one byte high would reject
// a payload of exactly the configured cap. Padding is what makes it exact, so
// every residue of 3 is covered rather than a sample.
func TestBase64DecodedLen(t *testing.T) {
	for n := 0; n < 300; n++ {
		b64 := base64.StdEncoding.EncodeToString(make([]byte, n))
		if got := Base64DecodedLen(b64); got != n {
			t.Fatalf("Base64DecodedLen of %d encoded bytes = %d, want %d", n, got, n)
		}
	}
	// The sizes a cap actually takes: 1 MiB and 10 MiB are both 1 mod 3, the
	// residue that left the old encoded-length bound 2 bytes slack.
	for _, n := range []int{1023, 1024, 1 << 20, 10 << 20} {
		b64 := base64.StdEncoding.EncodeToString(make([]byte, n))
		if got := Base64DecodedLen(b64); got != n {
			t.Fatalf("Base64DecodedLen of %d encoded bytes = %d, want %d", n, got, n)
		}
	}
	// A payload with no data is zero rather than negative, and an unpadded one
	// is measured as the characters it carries.
	if got := Base64DecodedLen("===="); got != 0 {
		t.Errorf("all-padding payload = %d, want 0", got)
	}
}

func TestMediaPartURL(t *testing.T) {
	tests := []struct {
		name    string
		part    map[string]any
		wantURL string
	}{
		{
			name:    "image_url with nested url",
			part:    map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/image.jpg"}},
			wantURL: "https://example.com/image.jpg",
		},
		{
			name:    "audio_url with nested url",
			part:    map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "https://example.com/audio.mp3"}},
			wantURL: "https://example.com/audio.mp3",
		},
		{
			name:    "video_url with nested url",
			part:    map[string]any{"type": "video_url", "video_url": map[string]any{"url": "https://example.com/video.mp4"}},
			wantURL: "https://example.com/video.mp4",
		},
		{
			name:    "input_audio is inline and carries no url",
			part:    map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "base64data", "format": "wav"}},
			wantURL: "",
		},
		{
			name:    "text part carries no url",
			part:    map[string]any{"type": "text", "text": "hello"},
			wantURL: "",
		},
		{
			name:    "image_url missing the nested url field",
			part:    map[string]any{"type": "image_url", "image_url": map[string]any{}},
			wantURL: "",
		},
		{
			name:    "image_url whose value is not an object",
			part:    map[string]any{"type": "image_url", "image_url": "https://example.com/image.jpg"},
			wantURL: "",
		},
		{
			name:    "input_image with a bare string url",
			part:    map[string]any{"type": "input_image", "image_url": "https://example.com/image.jpg"},
			wantURL: "https://example.com/image.jpg",
		},
		{
			// file_id and image_url are siblings, so a part naming a file
			// carries no image_url at all.
			name:    "input_image naming a file_id",
			part:    map[string]any{"type": "input_image", "file_id": "file-123"},
			wantURL: "",
		},
		{
			name:    "input_image whose image_url is not a string",
			part:    map[string]any{"type": "input_image", "image_url": map[string]any{}},
			wantURL: "",
		},
		{
			name:    "part with no type",
			part:    map[string]any{"image_url": "https://example.com/image.jpg"},
			wantURL: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if url := MediaPartURL(tt.part); url != tt.wantURL {
				t.Errorf("MediaPartURL() = %q, want %q", url, tt.wantURL)
			}
		})
	}
}

func TestNewEncoderPrimingBody(t *testing.T) {
	t.Run("a responses body carries only the primed part", func(t *testing.T) {
		part := map[string]any{"type": "input_image", "image_url": "https://example.com/img1.jpg", "detail": "high"}
		clientBody := map[string]any{
			"model":               "m",
			"mm_processor_kwargs": map[string]any{"num_crops": 4},
			"media_io_kwargs":     map[string]any{"video": map[string]any{"num_frames": 8}},
			// Salts the KV cache rather than the multimodal hash, and nothing
			// reads the blocks a priming call leaves.
			"cache_salt": "salt",
			// State the router does not keep, an uncapped limit, and a second
			// image: none belong on a per-part priming request.
			"previous_response_id": "resp-123",
			"conversation":         "conv-123",
			"store":                true,
			"background":           true,
			"max_output_tokens":    500,
			"instructions":         "be nice",
			"tools":                []any{map[string]any{"type": "function", "name": "f"}},
			"input": []any{map[string]any{"role": "user", "content": []any{
				part,
				map[string]any{"type": "input_image", "image_url": "https://example.com/img2.jpg"},
			}}},
		}

		body := NewEncoderPrimingBody(clientBody, part, APITypeResponses)

		// Asserting the whole map rather than the key set: forwarding the
		// client's own input would leave every key identical and still ship
		// the second image.
		want := map[string]any{
			"model":               "m",
			"mm_processor_kwargs": map[string]any{"num_crops": 4},
			"media_io_kwargs":     map[string]any{"video": map[string]any{"num_frames": 8}},
			"store":               false,
			"stream":              false,
			"max_output_tokens":   1,
			"input":               []map[string]any{{"role": "user", "content": []map[string]any{part}}},
		}
		if !reflect.DeepEqual(body, want) {
			t.Errorf("body = %#v, want %#v", body, want)
		}
	})

	t.Run("a chat completions body carries only the primed part", func(t *testing.T) {
		part := map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/img.jpg"}}
		clientBody := map[string]any{
			"model":           "m",
			"temperature":     0.7,
			"n":               4,
			"response_format": map[string]any{"type": "json_object"},
			"logit_bias":      map[string]any{"123": -100},
			"stream_options":  map[string]any{"include_usage": true},
			"tools":           []any{map[string]any{"type": "function"}},
			"max_tokens":      50,
			// A reasoning-model client's cap must not survive uncapped
			// alongside max_tokens=1.
			"max_completion_tokens": 100,
			"min_tokens":            5,
			"messages":              []any{map[string]any{"role": "user", "content": []any{part}}},
		}

		body := NewEncoderPrimingBody(clientBody, part, APITypeChatCompletions)

		wantKeys := []string{"max_completion_tokens", "max_tokens", "messages", "model", "stream"}
		if gotKeys := slices.Sorted(maps.Keys(body)); !slices.Equal(gotKeys, wantKeys) {
			t.Errorf("keys = %v, want %v", gotKeys, wantKeys)
		}
		if body["max_tokens"] != 1 || body["max_completion_tokens"] != 1 {
			t.Errorf("expected both output caps rewritten to 1, got %#v", body)
		}
		if body["stream"] != false {
			t.Errorf("expected stream disabled, got %#v", body["stream"])
		}
		if !reflect.DeepEqual(body["messages"], []map[string]any{{"role": "user", "content": []map[string]any{part}}}) {
			t.Errorf("messages = %#v", body["messages"])
		}
	})

	t.Run("every api type but responses is treated as chat completions", func(t *testing.T) {
		part := map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/img.jpg"}}
		clientBody := map[string]any{"model": "m"}

		want := NewEncoderPrimingBody(clientBody, part, APITypeChatCompletions)
		// Asserting the whole body, not just the carrier: the caps an API
		// writes depend on tokenLimitFields and on CapSingleToken's
		// sampling_params case, so a type sharing the carrier can still
		// differ on where its output cap lands.
		for _, apiType := range []APIType{
			APITypeCompletions,
			APITypeVLLMGenerate,
			APITypeSGLangGenerate,
			APITypeMessages,
			APIType(7),
		} {
			body := NewEncoderPrimingBody(clientBody, part, apiType)
			if !reflect.DeepEqual(body, want) {
				t.Errorf("%s: body = %#v, want %#v", apiType, body, want)
			}
		}
	})

	t.Run("an absent model stays absent", func(t *testing.T) {
		// An explicit JSON null is rejected differently by vLLM's request
		// validation than a missing field.
		part := map[string]any{"type": "input_image", "image_url": "https://example.com/img.jpg"}

		body := NewEncoderPrimingBody(map[string]any{}, part, APITypeResponses)

		if _, ok := body["model"]; ok {
			t.Errorf("expected no model key, got %#v", body["model"])
		}
	})

	t.Run("raw json values survive to the wire", func(t *testing.T) {
		// A caller that decodes only the fields it reads passes the rest
		// through as raw bytes, which have to marshal back unchanged.
		part := map[string]any{"type": "input_image", "image_url": "https://example.com/img.jpg"}
		clientBody := map[string]any{
			"model":               json.RawMessage(`"m"`),
			"mm_processor_kwargs": json.RawMessage(`{"max_pixels":313600}`),
		}

		body := NewEncoderPrimingBody(clientBody, part, APITypeResponses)

		encoded, err := json.Marshal(body)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		var roundTripped map[string]any
		if err := json.Unmarshal(encoded, &roundTripped); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		if roundTripped["model"] != "m" {
			t.Errorf("model = %#v, want \"m\"", roundTripped["model"])
		}
		if want := map[string]any{"max_pixels": float64(313600)}; !reflect.DeepEqual(roundTripped["mm_processor_kwargs"], want) {
			t.Errorf("mm_processor_kwargs = %#v, want %#v", roundTripped["mm_processor_kwargs"], want)
		}
	})
}

func TestItemPartArrays(t *testing.T) {
	tests := []struct {
		name    string
		item    map[string]any
		apiType APIType
		want    []PartArray
	}{
		{
			name:    "chat message content",
			item:    map[string]any{"role": "user", "content": []any{"a"}},
			apiType: APITypeChatCompletions,
			want:    []PartArray{{Field: FieldContent, Parts: []any{"a"}}},
		},
		{
			name:    "responses input item content",
			item:    map[string]any{"role": "user", "content": []any{"a"}},
			apiType: APITypeResponses,
			want:    []PartArray{{Field: FieldContent, Parts: []any{"a"}}},
		},
		{
			// vLLM forwards a function_call_output's output as a tool message's
			// content, so media in it reaches the model and has to be walked.
			name:    "responses function_call_output output",
			item:    map[string]any{"type": "function_call_output", "output": []any{"a"}},
			apiType: APITypeResponses,
			want:    []PartArray{{Field: FieldOutput, Parts: []any{"a"}}},
		},
		{
			name:    "content precedes output",
			item:    map[string]any{"content": []any{"a"}, "output": []any{"b"}},
			apiType: APITypeResponses,
			want: []PartArray{
				{Field: FieldContent, Parts: []any{"a"}},
				{Field: FieldOutput, Parts: []any{"b"}},
			},
		},
		{
			// Chat completions defines no output, so walking one would collect a
			// part the client never sent.
			name:    "chat output is not walked",
			item:    map[string]any{"content": []any{"a"}, "output": []any{"b"}},
			apiType: APITypeChatCompletions,
			want:    []PartArray{{Field: FieldContent, Parts: []any{"a"}}},
		},
		{
			// A computer_call_output's output is an object, not an array.
			name:    "non-array output is skipped",
			item:    map[string]any{"type": "computer_call_output", "output": map[string]any{"type": "computer_screenshot"}},
			apiType: APITypeResponses,
			want:    nil,
		},
		{
			name:    "non-array content is skipped",
			item:    map[string]any{"role": "user", "content": "plain text"},
			apiType: APITypeChatCompletions,
			want:    nil,
		},
		{
			name:    "item with neither field",
			item:    map[string]any{"role": "user"},
			apiType: APITypeResponses,
			want:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ItemPartArrays(tt.item, tt.apiType)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("ItemPartArrays() = %#v, want %#v", got, tt.want)
			}
		})
	}
}
