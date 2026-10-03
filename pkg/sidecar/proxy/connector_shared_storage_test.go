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

// statefulResponsesTestBody is a /v1/responses body carrying the fields
// reqcommon.RejectStatefulResponsesFields refuses, shared by the tests that
// assert such a request is refused before it reaches any upstream.
const statefulResponsesTestBody = `{"model":"m","input":"hi","previous_response_id":"resp-123","conversation":"conv-123","background":true}`

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

			body := `{"model":"m","messages":[],"stream":true,"cache_hit_threshold":0.5}`
			req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(body))
			require.PanicsWithValue(t, http.ErrAbortHandler, func() {
				srv.handleSharedStorage(client, req, "prefill:8000", reqcommon.APITypeChatCompletions)
			})
			require.Equal(t, http.StatusOK, client.Code)
			require.Equal(t, tt.wantBody, client.Body.String())
		})
	}
}
