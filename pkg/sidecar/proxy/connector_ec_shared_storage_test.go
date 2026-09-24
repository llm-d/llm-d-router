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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/go-logr/logr/funcr"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func TestExtractMMItems(t *testing.T) {
	tests := []struct {
		name    string
		request map[string]any
		apiType reqcommon.APIType
		// expected is the item count. wantURLs, where the request mixes part
		// types, is the URL of each item in order: a count alone passes when
		// the selection is inverted and the wrong part of the pair survives.
		expected int
		wantURLs []string
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
		{
			name: "responses single input_image item",
			request: map[string]any{
				"input": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type": "input_text",
								"text": "What's in this image?",
							},
							map[string]any{
								"type":      "input_image",
								"image_url": "https://example.com/image.jpg",
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 1,
		},
		// The shapes MediaPartURL resolves to "": a Responses input_image whose
		// image_url is an object rather than a bare string, and one carrying no
		// image_url at all.
		{
			name: "responses input_image with a nested image_url is skipped",
			request: map[string]any{
				"input": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type":      "input_image",
								"image_url": map[string]any{"url": "https://example.com/image.jpg"},
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 0,
		},
		{
			name: "responses input_image with no image_url is skipped",
			request: map[string]any{
				"input": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type":   "input_image",
								"detail": "high",
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 0,
		},
		{
			name: "responses input as bare string has no items",
			request: map[string]any{
				"input": "hello",
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 0,
		},
		{
			// The mirror of the chat case below. The field to walk is chosen by
			// apiType, so a default arm that drifted to Responses would prime
			// chat content the Responses input union never carried.
			name: "responses request never reads a stray messages field",
			request: map[string]any{
				"input": "hello",
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type":      "image_url",
								"image_url": map[string]any{"url": "https://example.com/image.jpg"},
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 0,
		},
		{
			name: "chat completions request never reads a stray input field",
			request: map[string]any{
				"input": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type":      "input_image",
								"image_url": "https://example.com/image.jpg",
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeChatCompletions,
			expected: 0,
		},
		{
			// vLLM's chat-completions parser resolves input_image and image_url
			// through the same multimodal map, so an input_image on a chat
			// request is an image the model server will process and the encoder
			// has to be primed for it.
			name: "chat completions input_image part is extracted",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type":      "input_image",
								"image_url": "https://example.com/image.jpg",
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeChatCompletions,
			expected: 1,
			wantURLs: []string{"https://example.com/image.jpg"},
		},
		{
			name: "chat completions image_url alongside an input_image part",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type":      "image_url",
								"image_url": map[string]any{"url": "https://example.com/real.jpg"},
							},
							map[string]any{
								"type":      "input_image",
								"image_url": "https://example.com/second.jpg",
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeChatCompletions,
			expected: 2,
			wantURLs: []string{"https://example.com/real.jpg", "https://example.com/second.jpg"},
		},
		{
			name: "responses image_url part is not extracted",
			request: map[string]any{
				"input": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type":      "image_url",
								"image_url": map[string]any{"url": "https://example.com/image.jpg"},
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 0,
		},
		{
			name: "responses audio_url and video_url parts are not extracted",
			request: map[string]any{
				"input": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type":      "audio_url",
								"audio_url": map[string]any{"url": "https://example.com/speech.wav"},
							},
							map[string]any{
								"type":      "video_url",
								"video_url": map[string]any{"url": "https://example.com/clip.mp4"},
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 0,
		},
		{
			name: "responses input_image alongside a stray image_url part",
			request: map[string]any{
				"input": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type":      "image_url",
								"image_url": map[string]any{"url": "https://example.com/stray.jpg"},
							},
							map[string]any{
								"type":      "input_image",
								"image_url": "https://example.com/real.jpg",
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 1,
			wantURLs: []string{"https://example.com/real.jpg"},
		},
		{
			// The Responses input content union is input_text / input_image /
			// input_file, so priming an input_audio part against PathResponses
			// would fail the fanout for the whole request.
			name: "responses input_audio part is not extracted",
			request: map[string]any{
				"input": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type":        "input_audio",
								"input_audio": map[string]any{"data": "AAAA", "format": "wav"},
							},
						},
					},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 0,
		},
		// A function_call_output carries its parts under output, and vLLM forwards
		// that array as a tool message's content, so an image there reaches the
		// model. RejectStatefulResponsesFields already walks to this depth for
		// file_id, so an extraction fixed to content would pass the guard and
		// leave the prefiller looking up a hash nothing primed.
		{
			name: "responses function_call_output output array yields its image",
			request: map[string]any{
				"input": []any{
					map[string]any{
						"type":    "function_call_output",
						"call_id": "c1",
						"output": []any{
							map[string]any{"type": "input_text", "text": "here it is"},
							inputImageItem("https://example.com/tool-output.jpg"),
						},
					},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 1,
			wantURLs: []string{"https://example.com/tool-output.jpg"},
		},
		{
			name: "responses function_call_output with a string output yields nothing",
			request: map[string]any{
				"input": []any{
					map[string]any{"type": "function_call_output", "call_id": "c1", "output": "42"},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 0,
		},
		{
			// output is not a chat-completions field, so the gate keeps a chat
			// request from reading one a client sent by mistake.
			name: "chat completions request never reads an output field",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"type":   "function_call_output",
						"output": []any{imageURLItem("https://example.com/stray.jpg")},
					},
				},
			},
			apiType:  reqcommon.APITypeChatCompletions,
			expected: 0,
		},
		{
			name: "responses turn carrying both content and output yields both images",
			request: map[string]any{
				"input": []any{
					map[string]any{
						"role":    "user",
						"content": []any{inputImageItem("https://example.com/from-content.jpg")},
						"output":  []any{inputImageItem("https://example.com/from-output.jpg")},
					},
				},
			},
			apiType:  reqcommon.APITypeResponses,
			expected: 2,
			wantURLs: []string{"https://example.com/from-content.jpg", "https://example.com/from-output.jpg"},
		},
		{
			// Python's json reads a number outside float64 range as inf, so the
			// model server serves this body and processes the image. Decoding
			// the turn has to survive it, or the image reaches the prefiller
			// with nothing primed for it.
			name: "a turn carrying a number outside float64 range still yields its image",
			request: map[string]any{
				"messages": json.RawMessage(`[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.jpg"}}],"pinned":1e999}]`),
			},
			apiType:  reqcommon.APITypeChatCompletions,
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, err := json.Marshal(tt.request)
			assert.NoError(t, err)
			parsed, err := decodeRequestBody(body)
			assert.NoError(t, err)

			items := extractMMItems(log.Log, parsed, tt.apiType)
			assert.Equal(t, tt.expected, len(items), "unexpected number of MM items")
			if tt.wantURLs != nil {
				gotURLs := make([]string, 0, len(items))
				for _, item := range items {
					gotURLs = append(gotURLs, reqcommon.MediaPartURL(item))
				}
				assert.Equal(t, tt.wantURLs, gotURLs, "wrong parts extracted")
			}
		})
	}
}

// TestExtractMMItemsDropsAreLogged pins the drop count to default verbosity. A
// dropped part never becomes an item, so it never reaches the ec-nixl item
// count that the partial-coverage warning compares against: this line is the
// only signal that the encoder was not primed for content the client sent.
func TestExtractMMItemsDropsAreLogged(t *testing.T) {
	tests := []struct {
		name    string
		request map[string]any
		apiType reqcommon.APIType
		// Expected count on each line, zero meaning the line must not fire.
		wantPartDrops int
		wantTurnDrops int
		wantExtracted int
	}{
		{
			name:          "responses input_image with no fetchable URL",
			request:       map[string]any{"input": []any{userContent(map[string]any{"type": "input_image", "detail": "high"})}},
			apiType:       reqcommon.APITypeResponses,
			wantPartDrops: 1,
		},
		{
			// Not a drop: vLLM primes input_image on chat completions too.
			name:    "chat request carrying an input_image part stays quiet",
			request: map[string]any{"messages": []any{userContent(map[string]any{"type": "input_image", "image_url": "https://example.com/img.jpg"})}},
			apiType: reqcommon.APITypeChatCompletions,
		},
		{
			name:          "responses request carrying a chat image_url part",
			request:       map[string]any{"input": []any{userContent(map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/img.jpg"}})}},
			apiType:       reqcommon.APITypeResponses,
			wantPartDrops: 1,
		},
		{
			// Two drops beside one survivor: the smallest input that tells a
			// real count apart from one hardcoded to 1, and a real surviving
			// total apart from 0.
			name: "every drop is counted alongside what survived",
			request: map[string]any{"input": []any{userContent(
				inputImageItem("https://example.com/keep.jpg"),
				map[string]any{"type": "input_image", "detail": "high"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/drop.jpg"}},
			)}},
			apiType:       reqcommon.APITypeResponses,
			wantPartDrops: 2,
			wantExtracted: 1,
		},
		{
			name:          "a responses input item that does not decode as an object is counted",
			request:       map[string]any{"input": []any{"not-an-object"}},
			apiType:       reqcommon.APITypeResponses,
			wantTurnDrops: 1,
		},
		{
			// The turn loop is shared, so the count and its line hold for a chat
			// messages entry as well.
			name:          "a chat messages entry that does not decode as an object is counted",
			request:       map[string]any{"messages": []any{"not-an-object"}},
			apiType:       reqcommon.APITypeChatCompletions,
			wantTurnDrops: 1,
		},
		{
			// The two counts are independent: a turn that fails to decode and a
			// droppable part in a later turn both land in one request, so a single
			// counter would report three skipped parts where two were parts.
			name: "a bad turn and a dropped part are counted apart",
			request: map[string]any{"input": []any{
				"not-an-object",
				userContent(
					inputImageItem("https://example.com/keep.jpg"),
					map[string]any{"type": "input_image", "detail": "high"},
				),
			}},
			apiType:       reqcommon.APITypeResponses,
			wantPartDrops: 1,
			wantTurnDrops: 1,
			wantExtracted: 1,
		},
		// Deliberately uncounted, so that the line stays actionable: counting
		// these would fire it on ordinary text traffic, where most parts are a
		// type the encoder never primes.
		{
			name: "a content element that is not an object stays quiet",
			request: map[string]any{"input": []any{map[string]any{
				"role":    "user",
				"content": []any{"describe this", inputImageItem("https://example.com/img.jpg")},
			}}},
			apiType: reqcommon.APITypeResponses,
		},
		{
			name:    "a part with no type stays quiet",
			request: map[string]any{"input": []any{userContent(map[string]any{"text": "hi"})}},
			apiType: reqcommon.APITypeResponses,
		},
		{
			name:    "a part whose type is not a string stays quiet",
			request: map[string]any{"input": []any{userContent(map[string]any{"type": []any{"input_image"}, "image_url": "https://example.com/img.jpg"})}},
			apiType: reqcommon.APITypeResponses,
		},
		{
			name:    "a text part stays quiet",
			request: map[string]any{"input": []any{userContent(map[string]any{"type": "input_text", "text": "hi"})}},
			apiType: reqcommon.APITypeResponses,
		},
		{
			name:    "an input item with no content array stays quiet",
			request: map[string]any{"input": []any{map[string]any{"type": "function_call", "name": "f", "arguments": "{}"}}},
			apiType: reqcommon.APITypeResponses,
		},
		{
			name:    "nothing dropped stays quiet",
			request: map[string]any{"input": []any{userContent(inputImageItem("https://example.com/img.jpg"))}},
			apiType: reqcommon.APITypeResponses,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, err := json.Marshal(tt.request)
			require.NoError(t, err)
			parsed, err := decodeRequestBody(body)
			require.NoError(t, err)

			var logged []string
			logger := funcr.New(func(_, args string) {
				logged = append(logged, args)
			}, funcr.Options{Verbosity: 0})

			extractMMItems(logger, parsed, tt.apiType)

			// The numbers are the payload of each line: a counter stuck at 1, or
			// one reporting the wrong surviving total, renders identically for
			// every single-drop request. The two lines are asserted apart so a
			// turn that does not decode cannot be reported as a skipped part.
			assertLine := func(marker string, wantCount int) {
				t.Helper()
				idx := slices.IndexFunc(logged, func(e string) bool {
					return strings.Contains(e, marker)
				})
				assert.Equal(t, wantCount > 0, idx >= 0, "marker=%q logged=%v", marker, logged)
				if wantCount == 0 || idx < 0 {
					return
				}
				assert.Contains(t, logged[idx], fmt.Sprintf(`"count"=%d`, wantCount))
				assert.Contains(t, logged[idx], fmt.Sprintf(`"extracted"=%d`, tt.wantExtracted))
			}
			assertLine("skipped multimodal content parts", tt.wantPartDrops)
			assertLine("skipped turns that do not decode", tt.wantTurnDrops)
		})
	}
}

// TestMMItemsForFanoutLogsRequestID pins the drop line to a request. It is the
// only default-verbosity signal that a client's content went unprimed, so
// without a request ID an operator cannot tell which of the requests in flight
// lost an image.
func TestMMItemsForFanoutLogsRequestID(t *testing.T) {
	body, err := json.Marshal(map[string]any{"input": []any{userContent(
		inputImageItem("https://example.com/keep.jpg"),
		map[string]any{"type": "input_image", "detail": "high"},
	)}})
	require.NoError(t, err)
	parsed, err := decodeRequestBody(body)
	require.NoError(t, err)

	var logged []string
	srv := NewProxy(Config{Port: "0"})
	srv.logger = funcr.New(func(_, args string) {
		logged = append(logged, args)
	}, funcr.Options{Verbosity: logging.DEBUG})

	srv.mmItemsForFanout(parsed, "req-42", reqcommon.APITypeResponses)

	idx := slices.IndexFunc(logged, func(e string) bool {
		return strings.Contains(e, "skipped multimodal content parts")
	})
	require.GreaterOrEqual(t, idx, 0, "drop line not logged: %v", logged)
	assert.Contains(t, logged[idx], `"requestID"="req-42"`)
}

// userContent wraps content parts in a user-role turn, the shape both a
// chat messages element and a Responses input item take.
func userContent(parts ...map[string]any) map[string]any {
	return map[string]any{"role": "user", "content": parts}
}

// imageURLItem builds an image_url content item.
func imageURLItem(url string) map[string]any {
	return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}
}

// videoURLItem builds a video_url content item.
func videoURLItem(url string) map[string]any {
	return map[string]any{"type": "video_url", "video_url": map[string]any{"url": url}}
}

// audioURLItem builds an audio_url content item, the audio type that names a URL
// to fetch; inlineAudioItem covers the input_audio inline path instead.
func audioURLItem(url string) map[string]any {
	return map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": url}}
}

// inlineAudioItem builds an input_audio content item. Format is fixed to
// "wav" — no test currently exercises another format; add a parameter back
// when a caller needs one.
func inlineAudioItem(data string) map[string]any {
	return map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": data, "format": "wav"}}
}

// inputImageItem builds a Responses input_image content item. Unlike
// image_url, the URL is a bare string directly on the item.
func inputImageItem(url string) map[string]any {
	return map[string]any{"type": "input_image", "image_url": url}
}

// userMessageRequest wraps content items in a minimal chat-completions request,
// with messages held as raw bytes the way decodeRequestBody leaves them.
func userMessageRequest(items ...map[string]any) map[string]any {
	messages, _ := json.Marshal([]any{
		map[string]any{"role": "user", "content": items},
	})
	return map[string]any{"messages": json.RawMessage(messages)}
}

// responsesInputRequest wraps content items in a minimal Responses request,
// with input held as raw bytes the way decodeRequestBody leaves them.
func responsesInputRequest(items ...map[string]any) map[string]any {
	input, _ := json.Marshal([]any{
		map[string]any{"role": "user", "content": items},
	})
	return map[string]any{"input": json.RawMessage(input)}
}

// TestFanoutEncoderPath locks in the path the encoder is addressed on for
// each API type. The pipeline tests answer on any path, so without this an
// inverted selection in fanoutEncoder would send every chat request to the
// Responses endpoint and fail nothing.
func TestFanoutEncoderPath(t *testing.T) {
	tests := []struct {
		name     string
		request  map[string]any
		apiType  reqcommon.APIType
		wantPath string
	}{
		{
			name:     "chat completions request",
			request:  userMessageRequest(imageURLItem("https://example.com/image.jpg")),
			apiType:  reqcommon.APITypeChatCompletions,
			wantPath: reqcommon.PathChatCompletions,
		},
		{
			name:     "responses request",
			request:  responsesInputRequest(inputImageItem("https://example.com/image.jpg")),
			apiType:  reqcommon.APITypeResponses,
			wantPath: reqcommon.PathResponses,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotPaths []string
			var mu sync.Mutex
			encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				gotPaths = append(gotPaths, r.URL.Path)
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{}`))
			}))
			defer encoderBackend.Close()

			encoderURL, err := url.Parse(encoderBackend.URL)
			require.NoError(t, err)
			srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
			srv.logger = log.Log

			err = srv.fanoutEncoderPrimer(context.Background(), tt.request, []string{encoderURL.Host}, "test-req-id", tt.apiType)
			require.NoError(t, err)

			mu.Lock()
			defer mu.Unlock()
			assert.Equal(t, []string{tt.wantPath}, gotPaths)
		})
	}
}

// TestFanoutEncoderPrimerOnePerPart pins one encoder request per extracted
// content part. Repeated content is primed repeatedly: collapsing it would mean
// deciding which parts share a multimodal hash, and a wrong call there drops a
// part the prefiller then cannot find primed.
func TestFanoutEncoderPrimerOnePerPart(t *testing.T) {
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
		request       map[string]any
		apiType       reqcommon.APIType
		expectedCalls int32
	}{
		{
			name:          "distinct image URLs",
			request:       userMessageRequest(imageURLItem("https://example.com/img1.jpg"), imageURLItem("https://example.com/img2.jpg")),
			apiType:       reqcommon.APITypeChatCompletions,
			expectedCalls: 2,
		},
		{
			name:          "one image URL sent twice",
			request:       userMessageRequest(imageURLItem("https://example.com/same.jpg"), imageURLItem("https://example.com/same.jpg")),
			apiType:       reqcommon.APITypeChatCompletions,
			expectedCalls: 2,
		},
		{
			name:          "one video URL sent twice",
			request:       userMessageRequest(videoURLItem("https://example.com/same.mp4"), videoURLItem("https://example.com/same.mp4")),
			apiType:       reqcommon.APITypeChatCompletions,
			expectedCalls: 2,
		},
		{
			name:          "inline audio sent twice",
			request:       userMessageRequest(inlineAudioItem("aaa"), inlineAudioItem("aaa")),
			apiType:       reqcommon.APITypeChatCompletions,
			expectedCalls: 2,
		},
		// A differing modality and a differing uuid are separate entries to the
		// serving engine, so neither pair may collapse into one encoder request.
		{
			name: "one URL under two modalities",
			request: userMessageRequest(
				videoURLItem("https://example.com/same"),
				audioURLItem("https://example.com/same"),
			),
			apiType:       reqcommon.APITypeChatCompletions,
			expectedCalls: 2,
		},
		{
			name: "one URL under two client uuids",
			request: userMessageRequest(
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/same.jpg"}, "uuid": "a"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/same.jpg"}, "uuid": "b"},
			),
			apiType:       reqcommon.APITypeChatCompletions,
			expectedCalls: 2,
		},
		{
			name: "responses input_image URLs with different detail",
			request: responsesInputRequest(
				map[string]any{"type": "input_image", "image_url": "https://example.com/same.jpg", "detail": "low"},
				map[string]any{"type": "input_image", "image_url": "https://example.com/same.jpg", "detail": "high"},
			),
			apiType:       reqcommon.APITypeResponses,
			expectedCalls: 2,
		},
		{
			name:          "one responses input_image URL sent twice",
			request:       responsesInputRequest(inputImageItem("https://example.com/same.jpg"), inputImageItem("https://example.com/same.jpg")),
			apiType:       reqcommon.APITypeResponses,
			expectedCalls: 2,
		},
		{
			name:          "distinct responses input_image URLs",
			request:       responsesInputRequest(inputImageItem("https://example.com/img1.jpg"), inputImageItem("https://example.com/img2.jpg")),
			apiType:       reqcommon.APITypeResponses,
			expectedCalls: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requestCount.Store(0)
			err := srv.fanoutEncoderPrimer(context.Background(), tt.request, []string{encoderHostPort}, "test-req-id", tt.apiType)
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedCalls, requestCount.Load())
		})
	}
}

// TestFanoutEncoderForwardsMMProcessorKwargs drives a client body through the
// same decode the sidecar uses and asserts mm_processor_kwargs reaches the
// encoder on the wire, where the field is raw bytes rather than a decoded map.
func TestFanoutEncoderForwardsMMProcessorKwargs(t *testing.T) {
	const mmKwargs = `{"min_pixels":3136,"max_pixels":313600}`

	tests := []struct {
		name    string
		body    string
		apiType reqcommon.APIType
	}{
		{
			name:    "chat completions request",
			body:    `{"model":"m","mm_processor_kwargs":` + mmKwargs + `,"media_io_kwargs":{"video":{"num_frames":8}},"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.jpg"}}]}]}`,
			apiType: reqcommon.APITypeChatCompletions,
		},
		{
			name:    "responses request",
			body:    `{"model":"m","mm_processor_kwargs":` + mmKwargs + `,"media_io_kwargs":{"video":{"num_frames":8}},"input":[{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/image.jpg"}]}]}`,
			apiType: reqcommon.APITypeResponses,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBodies [][]byte
			var mu sync.Mutex
			encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				mu.Lock()
				gotBodies = append(gotBodies, raw)
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{}`))
			}))
			defer encoderBackend.Close()

			encoderURL, err := url.Parse(encoderBackend.URL)
			require.NoError(t, err)
			srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
			srv.logger = log.Log

			parsed, err := decodeRequestBody([]byte(tt.body))
			require.NoError(t, err)

			require.NoError(t, srv.fanoutEncoderPrimer(context.Background(), parsed,
				[]string{encoderURL.Host}, "test-req-id", tt.apiType))

			mu.Lock()
			defer mu.Unlock()
			require.Len(t, gotBodies, 1)

			var got map[string]any
			require.NoError(t, json.Unmarshal(gotBodies[0], &got))
			assert.Equal(t, map[string]any{"min_pixels": float64(3136), "max_pixels": float64(313600)},
				got[reqcommon.FieldMMProcessorKwargs])
			assert.Equal(t, map[string]any{"video": map[string]any{"num_frames": float64(8)}},
				got[reqcommon.FieldMediaIOKwargs])
		})
	}
}
