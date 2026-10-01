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
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// routeOutcome is where createRoutes sends a request.
type routeOutcome int

const (
	// routeServed is the inference handler: it calls the prefiller EPP named.
	routeServed routeOutcome = iota
	// routeDecoder is the catch-all: it forwards the request to the decoder unread.
	routeDecoder
	// routeLocal is answered by the sidecar itself, reaching no backend.
	routeLocal
)

// pathRecorder is a backend that records the path of every request it receives.
type pathRecorder struct {
	mu    sync.Mutex
	paths []string
}

func (p *pathRecorder) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	p.mu.Lock()
	p.paths = append(p.paths, r.URL.Path)
	p.mu.Unlock()
	w.WriteHeader(http.StatusOK)
}

func (p *pathRecorder) got() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.paths...)
}

type routesUnderTest struct {
	handler      http.Handler
	decoder      *pathRecorder
	prefiller    *pathRecorder
	prefillerURL string
}

// newRoutesUnderTest builds createRoutes' handler for a NIXLv2 sidecar whose
// decoder and prefiller record the paths they receive.
func newRoutesUnderTest(t *testing.T) routesUnderTest {
	t.Helper()
	rt := routesUnderTest{decoder: &pathRecorder{}, prefiller: &pathRecorder{}}
	decoderServer := httptest.NewServer(rt.decoder)
	t.Cleanup(decoderServer.Close)
	prefillerServer := httptest.NewServer(rt.prefiller)
	t.Cleanup(prefillerServer.Close)
	rt.prefillerURL = prefillerServer.URL

	decoderURL, err := url.Parse(decoderServer.URL)
	require.NoError(t, err)
	s := NewProxy(Config{Port: "0", DecoderURL: decoderURL, KVConnector: KVConnectorNIXLV2})
	s.allowlistValidator = &AllowlistValidator{enabled: false}
	rt.handler = s.createRoutes()
	return rt
}

// TestCreateRoutes_Dispatch asserts where each method and path is dispatched.
// Every request carries the prefill header, so a served request reaches the
// prefiller and a catch-all request reaches only the decoder. wantPath is the
// path the backend receives.
func TestCreateRoutes_Dispatch(t *testing.T) {
	const body = `{"model":"m","input":"hi","prompt":"hi","messages":[{"role":"user","content":"hi"}],"max_tokens":1}`
	tests := []struct {
		method   string
		path     string
		want     routeOutcome
		wantPath string
	}{
		{http.MethodPost, "/v1/chat/completions", routeServed, "/v1/chat/completions"},
		{http.MethodPost, "/v1/completions", routeServed, "/v1/completions"},
		{http.MethodPost, "/v1/responses", routeServed, "/v1/responses"},
		{http.MethodPost, "/v1/messages", routeServed, "/v1/messages"},
		{http.MethodPost, "/inference/v1/generate", routeServed, "/inference/v1/generate"},
		{http.MethodPost, "/generate", routeServed, "/generate"},
		{http.MethodPost, "/v1/chat/completions/", routeServed, "/v1/chat/completions"},
		{http.MethodPost, "/v1/completions/", routeServed, "/v1/completions"},
		{http.MethodPost, "/v1/responses/", routeServed, "/v1/responses"},
		{http.MethodPost, "/v1/messages/", routeServed, "/v1/messages"},
		{http.MethodPost, "/inference/v1/generate/", routeServed, "/inference/v1/generate"},
		{http.MethodPost, "/generate/", routeServed, "/generate"},
		{http.MethodPost, "//v1/responses", routeServed, "/v1/responses"},
		{http.MethodPost, "/v1//chat/completions", routeServed, "/v1/chat/completions"},

		{http.MethodPost, "/v1/chat/completions/render", routeDecoder, "/v1/chat/completions/render"},
		{http.MethodPost, "/v1/chat/completions/chatcmpl_123", routeDecoder, "/v1/chat/completions/chatcmpl_123"},
		{http.MethodPost, "/v1/completions/render", routeDecoder, "/v1/completions/render"},
		{http.MethodPost, "/v1/messages/count_tokens", routeDecoder, "/v1/messages/count_tokens"},
		{http.MethodPost, "/v1/messages/batches", routeDecoder, "/v1/messages/batches"},
		{http.MethodGet, "/v1/messages/batches/mb_123/results", routeDecoder, "/v1/messages/batches/mb_123/results"},
		{http.MethodPost, "/v1/responses/resp_123/cancel", routeDecoder, "/v1/responses/resp_123/cancel"},
		{http.MethodPost, "/v1/responses/input_tokens", routeDecoder, "/v1/responses/input_tokens"},
		{http.MethodPost, "/v1/responses/compact", routeDecoder, "/v1/responses/compact"},
		{http.MethodGet, "/v1/responses/resp_123", routeDecoder, "/v1/responses/resp_123"},
		{http.MethodDelete, "/v1/responses/resp_123", routeDecoder, "/v1/responses/resp_123"},
		{http.MethodGet, "/v1/responses/resp_123/input_items", routeDecoder, "/v1/responses/resp_123/input_items"},
		{http.MethodPost, "/v1/conversations", routeDecoder, "/v1/conversations"},
		{http.MethodGet, "/v1/conversations/conv_1", routeDecoder, "/v1/conversations/conv_1"},
		{http.MethodDelete, "/v1/conversations/conv_1", routeDecoder, "/v1/conversations/conv_1"},
		{http.MethodPost, "/v1/conversations/conv_1/items", routeDecoder, "/v1/conversations/conv_1/items"},
		{http.MethodGet, "/v1/conversations/conv_1/items/item_1", routeDecoder, "/v1/conversations/conv_1/items/item_1"},
		{http.MethodDelete, "/v1/conversations/conv_1/items/item_1", routeDecoder, "/v1/conversations/conv_1/items/item_1"},
		{http.MethodGet, "/v1/chat/completions/chatcmpl_123", routeDecoder, "/v1/chat/completions/chatcmpl_123"},
		{http.MethodDelete, "/v1/chat/completions/chatcmpl_123", routeDecoder, "/v1/chat/completions/chatcmpl_123"},
		{http.MethodGet, "/v1/chat/completions/chatcmpl_123/messages", routeDecoder, "/v1/chat/completions/chatcmpl_123/messages"},
		{http.MethodPost, "/v1/embeddings", routeDecoder, "/v1/embeddings"},
		{http.MethodPost, "/v1/files", routeDecoder, "/v1/files"},
		{http.MethodGet, "/v1/files/file_1", routeDecoder, "/v1/files/file_1"},
		{http.MethodGet, "/v1/models", routeDecoder, "/v1/models"},
		{http.MethodGet, "/healthz", routeDecoder, "/healthz"},
		{http.MethodGet, "/v1/requests/req_1", routeDecoder, "/v1/requests/req_1"},

		{http.MethodGet, "/v1/chat/completions", routeDecoder, "/v1/chat/completions"},
		{http.MethodGet, "/v1/responses", routeDecoder, "/v1/responses"},
		{http.MethodGet, "/v1/responses/", routeDecoder, "/v1/responses"},
		{http.MethodDelete, "/v1/completions", routeDecoder, "/v1/completions"},
		{http.MethodHead, "/v1/messages", routeDecoder, "/v1/messages"},
		{http.MethodGet, "/inference/v1/generate", routeDecoder, "/inference/v1/generate"},
		{http.MethodPut, "/generate", routeDecoder, "/generate"},
		{http.MethodOptions, "/v1/chat/completions", routeDecoder, "/v1/chat/completions"},

		{http.MethodGet, "/health", routeLocal, ""},
	}
	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			rt := newRoutesUnderTest(t)
			req := httptest.NewRequest(tt.method, tt.path, strings.NewReader(body))
			req.Header.Set(routing.PrefillEndpointHeader, rt.prefillerURL)

			rec := httptest.NewRecorder()

			rt.handler.ServeHTTP(rec, req)

			switch tt.want {
			case routeServed:
				require.Equal(t, []string{tt.wantPath}, rt.prefiller.got(), "prefiller")
			case routeDecoder:
				require.Empty(t, rt.prefiller.got(), "prefiller")
				require.Equal(t, []string{tt.wantPath}, rt.decoder.got(), "decoder")
			case routeLocal:
				require.Equal(t, http.StatusOK, rec.Code)
				require.Empty(t, rt.prefiller.got(), "prefiller")
				require.Empty(t, rt.decoder.got(), "decoder")
			}
		})
	}
}

// TestCreateRoutes_RejectsStatefulResponsesFieldsOnEveryServedForm asserts the
// stateful-field guard runs on every form of PathResponses that is served,
// whether or not EPP selected a prefiller.
func TestCreateRoutes_RejectsStatefulResponsesFieldsOnEveryServedForm(t *testing.T) {
	for _, path := range []string{"/v1/responses", "/v1/responses/", "//v1/responses", "/v1/./responses", "/v1//responses/"} {
		for _, withPrefiller := range []bool{false, true} {
			name := path
			if withPrefiller {
				name += " with prefiller"
			}
			t.Run(name, func(t *testing.T) {
				rt := newRoutesUnderTest(t)
				req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(statefulResponsesTestBody))
				if withPrefiller {
					req.Header.Set(routing.PrefillEndpointHeader, rt.prefillerURL)
				}
				rec := httptest.NewRecorder()

				rt.handler.ServeHTTP(rec, req)

				dispatched := len(rt.decoder.got())+len(rt.prefiller.got()) > 0
				requireStatefulResponsesRejected(t, rec, dispatched)
			})
		}
	}
}
