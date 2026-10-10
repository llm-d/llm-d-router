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
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

// TestSharedStorage_StreamingDecodeFirstHeaders covers the headers of a
// streamed decode-first request. An attempt that ends in cache_threshold is
// discarded, so only the decode after prefill may set the client's headers.
func TestSharedStorage_StreamingDecodeFirstHeaders(t *testing.T) {
	const (
		roleEvent      = `data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n"
		stopEvent      = `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}` + "\n\n"
		thresholdEvent = `data: {"choices":[{"delta":{},"finish_reason":"cache_threshold"}]}` + "\n\n"
	)
	tests := []struct {
		name        string
		firstEvents string
		wantAttempt string
	}{
		{name: "decode-first response", firstEvents: roleEvent + stopEvent, wantAttempt: "decode-first"},
		{name: "cache_threshold fallback", firstEvents: roleEvent + thresholdEvent, wantAttempt: "after-prefill"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{}`))
			}))
			defer prefill.Close()

			decodeURL, err := url.Parse("http://decoder:8000")
			require.NoError(t, err)
			srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: constants.KVConnectorSharedStorage})
			srv.logger = log.Log
			firstAttempt := make(chan struct{}, 1)
			firstAttempt <- struct{}{}
			srv.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				// Add, not Set: the reverse proxy adds upstream headers to the client's map.
				w.Header().Add("Content-Type", "text/event-stream")
				select {
				case <-firstAttempt:
					w.Header().Add("X-Decode-Attempt", "decode-first")
					_, _ = w.Write([]byte(tt.firstEvents))
				default:
					w.Header().Add("X-Decode-Attempt", "after-prefill")
					_, _ = w.Write([]byte(roleEvent + stopEvent))
				}
			})

			body := `{"model":"m","messages":[],"stream":true,"cache_hit_threshold":0.5}`
			req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(body))
			recorder := httptest.NewRecorder()
			srv.handleSharedStorage(recorder, req, strings.TrimPrefix(prefill.URL, "http://"), reqcommon.APITypeChatCompletions)

			require.Equal(t, http.StatusOK, recorder.Code)
			require.Equal(t, roleEvent+stopEvent, recorder.Body.String())
			header := recorder.Result().Header
			require.Equal(t, []string{tt.wantAttempt}, header.Values("X-Decode-Attempt"))
			require.Equal(t, []string{"text/event-stream"}, header.Values("Content-Type"))
		})
	}
}

// statefulResponsesTestBody is a /v1/responses body carrying the fields
// reqcommon.RejectStatefulResponsesFields refuses, shared by the tests that
// assert such a request is refused before it reaches any upstream.
const statefulResponsesTestBody = `{"model":"m","input":"hi","previous_response_id":"resp-123","conversation":"conv-123","background":true}`

// streamingDecodeFirstBody is a streaming chat-completions request that takes
// the decode-first path, shared by the tests that cover that path.
const streamingDecodeFirstBody = `{"model":"m","messages":[],"stream":true,"cache_hit_threshold":0.5}`

// requireStatefulResponsesRejected asserts the handler answered 400 naming the
// offending field and dispatched nothing upstream. previous_response_id is the
// first field RejectStatefulResponsesFields checks, so it is the one named for
// statefulResponsesTestBody.
func requireStatefulResponsesRejected(t *testing.T, recorder *httptest.ResponseRecorder, dispatched bool) {
	t.Helper()
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	require.Contains(t, recorder.Body.String(), reqcommon.FieldPreviousResponseID)
	require.False(t, dispatched, "request reached an upstream despite an unsupported field")
}

// TestSharedStorage_RejectsStatefulResponsesFields covers handleSharedStorage's
// default path (no cache_hit_threshold): the request is refused in readJSONBody,
// so neither the prefill nor the decode upstream is ever dispatched.
func TestSharedStorage_RejectsStatefulResponsesFields(t *testing.T) {
	var dispatched bool
	prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		dispatched = true
		w.WriteHeader(http.StatusOK)
	}))
	defer prefill.Close()

	decodeURL, err := url.Parse("http://decoder:8000")
	require.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: constants.KVConnectorSharedStorage})
	srv.logger = log.Log
	srv.allowlistValidator = &AllowlistValidator{}
	srv.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		dispatched = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop"}]}`))
	})

	req := httptest.NewRequest(http.MethodPost, reqcommon.PathResponses, strings.NewReader(statefulResponsesTestBody))
	req.Header.Set(routing.PrefillEndpointHeader, strings.TrimPrefix(prefill.URL, "http://"))
	recorder := httptest.NewRecorder()
	srv.disaggregatedPrefillHandler(reqcommon.APITypeResponses)(recorder, req)

	requireStatefulResponsesRejected(t, recorder, dispatched)
}

// signalingRecorder closes written after its first body write.
type signalingRecorder struct {
	*httptest.ResponseRecorder
	once    sync.Once
	written chan struct{}
}

func (r *signalingRecorder) Write(b []byte) (int, error) {
	defer r.once.Do(func() { close(r.written) })
	return r.ResponseRecorder.Write(b)
}

// TestSharedStorage_StreamingDecodeFirstAbort covers a streamed decode-first
// attempt that breaks mid-response, for example when the client disconnects.
// The reverse proxy then panics with http.ErrAbortHandler on the goroutine that
// runs the attempt. net/http only recovers that panic on the request
// goroutine, so it has to reach the caller there instead of exiting the process.
func TestSharedStorage_StreamingDecodeFirstAbort(t *testing.T) {
	const (
		roleEvent    = `data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n"
		contentEvent = `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}` + "\n\n"
	)
	tests := []struct {
		name string
		// relayed is closed once the start of the stream has reached the client.
		decoder  func(relayed <-chan struct{}) http.Handler
		wantBody string
	}{
		{
			name: "abort after the stream reached the client",
			decoder: func(relayed <-chan struct{}) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(http.StatusOK)
					_, _ = io.WriteString(w, roleEvent+contentEvent)
					select {
					case <-relayed:
					case <-time.After(5 * time.Second):
					}
					panic(http.ErrAbortHandler)
				})
			},
			wantBody: roleEvent + contentEvent,
		},
		{
			name: "abort before the first chunk was inspected",
			decoder: func(<-chan struct{}) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(http.StatusOK)
					_, _ = io.WriteString(w, roleEvent)
					panic(http.ErrAbortHandler)
				})
			},
			wantBody: roleEvent,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decodeURL, err := url.Parse("http://decoder:8000")
			require.NoError(t, err)
			srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: constants.KVConnectorSharedStorage})
			srv.logger = log.Log
			client := &signalingRecorder{ResponseRecorder: httptest.NewRecorder(), written: make(chan struct{})}
			srv.decoderProxy = tt.decoder(client.written)

			body := streamingDecodeFirstBody
			req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(body))
			require.PanicsWithValue(t, http.ErrAbortHandler, func() {
				srv.handleSharedStorage(client, req, "prefill:8000", reqcommon.APITypeChatCompletions)
			})
			require.Equal(t, http.StatusOK, client.Code)
			require.Equal(t, tt.wantBody, client.Body.String())
		})
	}
}

// commitRecorder records whether anything was written to the client.
// httptest.ResponseRecorder.Code defaults to 200 before WriteHeader, so that
// field cannot show that an abort left the response untouched.
type commitRecorder struct {
	*httptest.ResponseRecorder
	once      sync.Once
	written   chan struct{}
	committed bool
}

func (r *commitRecorder) WriteHeader(code int) {
	r.committed = true
	r.ResponseRecorder.WriteHeader(code)
}

func (r *commitRecorder) Write(b []byte) (int, error) {
	r.committed = true
	defer r.once.Do(func() {
		if r.written != nil {
			close(r.written)
		}
	})
	return r.ResponseRecorder.Write(b)
}

func (r *commitRecorder) Flush() {
	r.committed = true
	r.ResponseRecorder.Flush()
}

// TestSharedStorage_StreamingDecodeFirstErrorAbort covers an aborted decode
// that produced an error status, or no status at all. The abort has to be
// replayed on the request goroutine, and a missing status must not be flushed
// as an empty 200.
func TestSharedStorage_StreamingDecodeFirstErrorAbort(t *testing.T) {
	const (
		errEvent  = `data: {"error":{"message":"unavailable"}}` + "\n\n"
		errEvent2 = `data: {"error":{"message":"still down"}}` + "\n\n"
	)
	tests := []struct {
		name          string
		decoder       func(relayed <-chan struct{}) http.Handler
		wantStatus    int
		wantBody      string
		wantCommitted bool
	}{
		{
			name: "abort before any status was written",
			decoder: func(<-chan struct{}) http.Handler {
				return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
					panic(http.ErrAbortHandler)
				})
			},
		},
		{
			name: "abort after an error status and a partial body",
			decoder: func(<-chan struct{}) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(http.StatusServiceUnavailable)
					_, _ = io.WriteString(w, errEvent)
					panic(http.ErrAbortHandler)
				})
			},
			wantStatus:    http.StatusServiceUnavailable,
			wantBody:      errEvent,
			wantCommitted: true,
		},
		{
			name: "abort after the error stream reached the client",
			decoder: func(relayed <-chan struct{}) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(http.StatusServiceUnavailable)
					_, _ = io.WriteString(w, errEvent+errEvent2)
					select {
					case <-relayed:
					case <-time.After(5 * time.Second):
					}
					panic(http.ErrAbortHandler)
				})
			},
			wantStatus:    http.StatusServiceUnavailable,
			wantBody:      errEvent + errEvent2,
			wantCommitted: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decodeURL, err := url.Parse("http://decoder:8000")
			require.NoError(t, err)
			srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: constants.KVConnectorSharedStorage})
			srv.logger = log.Log
			client := &commitRecorder{ResponseRecorder: httptest.NewRecorder(), written: make(chan struct{})}
			srv.decoderProxy = tt.decoder(client.written)

			body := streamingDecodeFirstBody
			req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(body))
			require.PanicsWithValue(t, http.ErrAbortHandler, func() {
				srv.handleSharedStorage(client, req, "prefill:8000", reqcommon.APITypeChatCompletions)
			})
			require.Equal(t, tt.wantCommitted, client.committed)
			if tt.wantCommitted {
				require.Equal(t, tt.wantStatus, client.Code)
				require.Equal(t, tt.wantBody, client.Body.String())
			}
		})
	}
}

// TestSharedStorage_StreamingDecodeFirstErrorStatus covers a decode-first
// stream that fails with a real error status and does not abort. The status
// and body are forwarded and the handler returns.
func TestSharedStorage_StreamingDecodeFirstErrorStatus(t *testing.T) {
	const errEvent = `data: {"error":{"message":"unavailable"}}` + "\n\n"
	decodeURL, err := url.Parse("http://decoder:8000")
	require.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: constants.KVConnectorSharedStorage})
	srv.logger = log.Log
	srv.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, errEvent)
	})

	body := streamingDecodeFirstBody
	req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(body))
	client := httptest.NewRecorder()
	require.NotPanics(t, func() {
		srv.handleSharedStorage(client, req, "prefill:8000", reqcommon.APITypeChatCompletions)
	})
	require.Equal(t, http.StatusServiceUnavailable, client.Code)
	require.Equal(t, errEvent, client.Body.String())
}
