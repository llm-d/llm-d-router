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

package proxy

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/tidwall/gjson"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func TestExtractMMItems(t *testing.T) {
	tests := []struct {
		name     string
		request  map[string]any
		expected int
	}{
		{
			name: "no multimodal items",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "Hello, world!",
					},
				},
			},
			expected: 0,
		},
		{
			name: "single image item",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type": "text",
								"text": "What's in this image?",
							},
							map[string]any{
								"type": "image_url",
								"image_url": map[string]any{
									"url": "https://example.com/image.jpg",
								},
							},
						},
					},
				},
			},
			expected: 1,
		},
		{
			name: "multiple multimodal items",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type": "image_url",
								"image_url": map[string]any{
									"url": "https://example.com/image1.jpg",
								},
							},
							map[string]any{
								"type": "audio_url",
								"audio_url": map[string]any{
									"url": "https://example.com/audio.mp3",
								},
							},
							map[string]any{
								"type": "text",
								"text": "Describe these",
							},
						},
					},
				},
			},
			expected: 2,
		},
		{
			name: "input_audio type",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type": "input_audio",
								"input_audio": map[string]any{
									"data":   "base64data",
									"format": "wav",
								},
							},
						},
					},
				},
			},
			expected: 1,
		},
		{
			name: "single video item",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type": "video_url",
								"video_url": map[string]any{
									"url": "https://example.com/video.mp4",
								},
							},
						},
					},
				},
			},
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			items := extractMMItems(mustJSON(t, tt.request))
			assert.Equal(t, tt.expected, len(items), "unexpected number of MM items")
		})
	}
}

func TestBuildEncoderRequest(t *testing.T) {
	originalRequest := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "text",
						"text": "What's in this image?",
					},
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url": "https://example.com/image.jpg",
						},
					},
				},
			},
		},
		"max_tokens": 100,
		"stream":     true,
	}

	mmItem := map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": "https://example.com/image.jpg",
		},
	}

	encoderRequest, err := buildEncoderRequest(mustJSON(t, originalRequest), gjson.ParseBytes(mustJSON(t, mmItem)))
	assert.NoError(t, err)

	// Verify encoder request modifications
	assert.Equal(t, int64(1), gjson.GetBytes(encoderRequest, "max_tokens").Int())
	assert.False(t, gjson.GetBytes(encoderRequest, "stream").Bool())
	assert.False(t, gjson.GetBytes(encoderRequest, "stream_options").Exists())
	assert.Equal(t, "test-model", gjson.GetBytes(encoderRequest, "model").Str)

	// Verify messages contain only the MM item
	messages := gjson.GetBytes(encoderRequest, "messages").Array()
	assert.Equal(t, 1, len(messages))

	content := messages[0].Get("content").Array()
	assert.Equal(t, 1, len(content))
	assert.Equal(t, "image_url", content[0].Get("type").Str)
}

// TestBuildEncoderRequest_MaxCompletionTokens is a regression test: a shallow
// copy previously left the client's max_completion_tokens value untouched
// alongside the newly-capped max_tokens=1, so a reasoning-model client's
// large max_completion_tokens would survive uncapped into the encoder request.
func TestBuildEncoderRequest_MaxCompletionTokens(t *testing.T) {
	originalRequest := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url": "https://example.com/image.jpg",
						},
					},
				},
			},
		},
		"max_tokens":            50,
		"max_completion_tokens": 100,
	}

	mmItem := map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": "https://example.com/image.jpg",
		},
	}

	encoderRequest, err := buildEncoderRequest(mustJSON(t, originalRequest), gjson.ParseBytes(mustJSON(t, mmItem)))
	assert.NoError(t, err)

	assert.Equal(t, int64(1), gjson.GetBytes(encoderRequest, "max_tokens").Int())
	assert.Equal(t, int64(1), gjson.GetBytes(encoderRequest, "max_completion_tokens").Int())
}

func TestMMItemURL(t *testing.T) {
	tests := []struct {
		name     string
		item     map[string]any
		expected string
	}{
		{
			name: "image_url with url",
			item: map[string]any{
				"type": "image_url",
				"image_url": map[string]any{
					"url": "https://example.com/image.jpg",
				},
			},
			expected: "https://example.com/image.jpg",
		},
		{
			name: "audio_url with url",
			item: map[string]any{
				"type": "audio_url",
				"audio_url": map[string]any{
					"url": "https://example.com/audio.mp3",
				},
			},
			expected: "https://example.com/audio.mp3",
		},
		{
			name: "video_url with url",
			item: map[string]any{
				"type": "video_url",
				"video_url": map[string]any{
					"url": "https://example.com/video.mp4",
				},
			},
			expected: "https://example.com/video.mp4",
		},
		{
			name: "input_audio has no url",
			item: map[string]any{
				"type": "input_audio",
				"input_audio": map[string]any{
					"data":   "base64data",
					"format": "wav",
				},
			},
			expected: "",
		},
		{
			name:     "text type has no url",
			item:     map[string]any{"type": "text", "text": "hello"},
			expected: "",
		},
		{
			name: "image_url missing nested url field",
			item: map[string]any{
				"type":      "image_url",
				"image_url": map[string]any{},
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, mmItemURL(gjson.ParseBytes(mustJSON(t, tt.item))))
		})
	}
}

// mustJSON marshals v, failing the test on error.
func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// imageURLItem builds an image_url content item.
func imageURLItem(url string) map[string]any {
	return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}
}

// videoURLItem builds a video_url content item.
func videoURLItem(url string) map[string]any {
	return map[string]any{"type": "video_url", "video_url": map[string]any{"url": url}}
}

// inlineAudioItem builds an input_audio content item.
func inlineAudioItem(data, format string) map[string]any {
	return map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": data, "format": format}}
}

// userMessageRequest wraps content items in a minimal chat-completions request body.
func userMessageRequest(items ...map[string]any) []byte {
	content := make([]any, len(items))
	for i, item := range items {
		content[i] = item
	}
	b, err := json.Marshal(map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": content},
		},
	})
	if err != nil {
		panic(err)
	}
	return b
}

func TestFanoutEncoderPrimerDeduplication(t *testing.T) {
	var requestCount atomic.Int32
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}]}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	encoderHostPort := encoderURL.Host

	tests := []struct {
		name          string
		request       []byte
		expectedCalls int32
	}{
		{
			name:          "no duplicates — all items sent",
			request:       userMessageRequest(imageURLItem("https://example.com/img1.jpg"), imageURLItem("https://example.com/img2.jpg")),
			expectedCalls: 2,
		},
		{
			name:          "duplicate image URLs — second is skipped",
			request:       userMessageRequest(imageURLItem("https://example.com/same.jpg"), imageURLItem("https://example.com/same.jpg")),
			expectedCalls: 1,
		},
		{
			name:          "duplicate video URLs — second is skipped",
			request:       userMessageRequest(videoURLItem("https://example.com/same.mp4"), videoURLItem("https://example.com/same.mp4")),
			expectedCalls: 1,
		},
		{
			name:          "inline audio items are never deduplicated",
			request:       userMessageRequest(inlineAudioItem("aaa", "wav"), inlineAudioItem("aaa", "wav")),
			expectedCalls: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requestCount.Store(0)
			err := srv.fanoutEncoderPrimer(context.Background(), tt.request, []string{encoderHostPort}, "test-req-id")
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedCalls, requestCount.Load())
		})
	}
}
