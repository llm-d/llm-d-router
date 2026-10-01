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

package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// auxRouteStep is a stub step that registers an auxiliary route.
type auxRouteStep struct {
	stubStep
	registered bool
	gotID      string
}

func (s *auxRouteStep) RegisterRoutes(r chi.Router) {
	s.registered = true
	r.Get("/v1/requests/{id}", func(w http.ResponseWriter, r *http.Request) {
		s.gotID = chi.URLParam(r, "id")
		w.WriteHeader(http.StatusOK)
	})
}

func TestServerRegistersStepRoutes(t *testing.T) {
	step := &auxRouteStep{stubStep: stubStep{name: "aux"}}
	p := pipeline.New([]pipeline.Step{stubStep{name: "plain"}, step})
	srv, err := New(config.ServerConfig{}, p, gateway.NewWithTransport(nil, stubGatewayURL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !step.registered {
		t.Fatal("RegisterRoutes was not called on the implementing step")
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/requests/abc-123", nil)
	rec := httptest.NewRecorder()
	srv.httpServer.Handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 from step route, got %d", rec.Code)
	}
	if step.gotID != "abc-123" {
		t.Fatalf("expected path param in handler, got %q", step.gotID)
	}

	// The built-in routes are untouched.
	req = httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec = httptest.NewRecorder()
	srv.httpServer.Handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 from /healthz, got %d", rec.Code)
	}
}

// TestServerRouteDispatch asserts where each method and path is dispatched.
// wantPath is the path the pipeline or the gateway receives.
func TestServerRouteDispatch(t *testing.T) {
	const (
		served      = "served"
		passthrough = "passthrough"
		notAllowed  = "405"
		local       = "local"
	)
	tests := []struct {
		method   string
		path     string
		want     string
		wantPath string
	}{
		{http.MethodPost, "/v1/chat/completions", served, "/v1/chat/completions"},
		{http.MethodPost, "/v1/completions", served, "/v1/completions"},
		{http.MethodPost, "/inference/v1/generate", served, "/inference/v1/generate"},
		{http.MethodPost, "/v1/chat/completions/", served, "/v1/chat/completions"},
		{http.MethodPost, "/v1/completions/", served, "/v1/completions"},
		{http.MethodPost, "/inference/v1/generate/", served, "/inference/v1/generate"},
		{http.MethodPost, "//v1/chat/completions", served, "/v1/chat/completions"},
		{http.MethodPost, "/v1/chat//completions", served, "/v1/chat/completions"},

		{http.MethodPost, "/v1/responses", passthrough, "/v1/responses"},
		{http.MethodPost, "/v1/responses/", passthrough, "/v1/responses"},
		{http.MethodGet, "/v1/responses", passthrough, "/v1/responses"},
		{http.MethodHead, "/v1/messages", passthrough, "/v1/messages"},
		{http.MethodPut, "/generate", passthrough, "/generate"},
		{http.MethodPost, "/v1/messages", passthrough, "/v1/messages"},
		{http.MethodPost, "/v1/messages/", passthrough, "/v1/messages"},
		{http.MethodPost, "/generate", passthrough, "/generate"},
		{http.MethodPost, "/generate/", passthrough, "/generate"},
		{http.MethodPost, "/v1/chat/completions/render", passthrough, "/v1/chat/completions/render"},
		{http.MethodPost, "/v1/completions/render", passthrough, "/v1/completions/render"},
		{http.MethodPost, "/v1/messages/count_tokens", passthrough, "/v1/messages/count_tokens"},
		{http.MethodPost, "/v1/messages/batches", passthrough, "/v1/messages/batches"},
		{http.MethodGet, "/v1/messages/batches/mb_123/results", passthrough, "/v1/messages/batches/mb_123/results"},
		{http.MethodPost, "/v1/responses/input_tokens", passthrough, "/v1/responses/input_tokens"},
		{http.MethodPost, "/v1/responses/compact", passthrough, "/v1/responses/compact"},
		{http.MethodPost, "/v1/responses/resp_123/cancel", passthrough, "/v1/responses/resp_123/cancel"},
		{http.MethodGet, "/v1/responses/resp_123", passthrough, "/v1/responses/resp_123"},
		{http.MethodDelete, "/v1/responses/resp_123", passthrough, "/v1/responses/resp_123"},
		{http.MethodGet, "/v1/responses/resp_123/input_items", passthrough, "/v1/responses/resp_123/input_items"},
		{http.MethodPost, "/v1/conversations", passthrough, "/v1/conversations"},
		{http.MethodGet, "/v1/conversations/conv_1", passthrough, "/v1/conversations/conv_1"},
		{http.MethodDelete, "/v1/conversations/conv_1", passthrough, "/v1/conversations/conv_1"},
		{http.MethodPost, "/v1/conversations/conv_1/items", passthrough, "/v1/conversations/conv_1/items"},
		{http.MethodGet, "/v1/conversations/conv_1/items/item_1", passthrough, "/v1/conversations/conv_1/items/item_1"},
		{http.MethodDelete, "/v1/conversations/conv_1/items/item_1", passthrough, "/v1/conversations/conv_1/items/item_1"},
		{http.MethodGet, "/v1/chat/completions/chatcmpl_123", passthrough, "/v1/chat/completions/chatcmpl_123"},
		{http.MethodPost, "/v1/chat/completions/chatcmpl_123", passthrough, "/v1/chat/completions/chatcmpl_123"},
		{http.MethodDelete, "/v1/chat/completions/chatcmpl_123", passthrough, "/v1/chat/completions/chatcmpl_123"},
		{http.MethodGet, "/v1/chat/completions/chatcmpl_123/messages", passthrough, "/v1/chat/completions/chatcmpl_123/messages"},
		{http.MethodPost, "/v1/embeddings", passthrough, "/v1/embeddings"},
		{http.MethodPost, "/v1/files", passthrough, "/v1/files"},
		{http.MethodGet, "/v1/files/file_1", passthrough, "/v1/files/file_1"},
		{http.MethodGet, "/v1/models", passthrough, "/v1/models"},
		{http.MethodGet, "/health", passthrough, "/health"},
		{http.MethodGet, "/v1/requests/req_1", passthrough, "/v1/requests/req_1"},

		{http.MethodGet, "/v1/chat/completions", notAllowed, ""},
		{http.MethodGet, "/v1/chat/completions/", notAllowed, ""},
		{http.MethodDelete, "/v1/completions", notAllowed, ""},
		{http.MethodHead, "/inference/v1/generate", notAllowed, ""},
		{http.MethodPut, "/inference/v1/generate", notAllowed, ""},
		{http.MethodOptions, "/v1/chat/completions", notAllowed, ""},

		{http.MethodGet, "/healthz", local, ""},
		{http.MethodGet, "/readyz", local, ""},
	}
	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			upstream, cap := newCapturingUpstream(t, http.StatusOK, "")
			var pipelinePath string
			step := stubStep{name: "record", fn: func(_ context.Context, rc *pipeline.RequestContext) error {
				pipelinePath = rc.OriginalPath
				return nil
			}}
			srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(&http.Transport{}, upstream.URL))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			req := httptest.NewRequest(tt.method, tt.path, strings.NewReader(`{"model":"m"}`))

			rec := doPassthrough(t, srv, req)

			_, upstreamPath, _, _, _ := cap.get()
			switch tt.want {
			case served:
				if pipelinePath != tt.wantPath || upstreamPath != "" {
					t.Fatalf("pipeline path %q, gateway path %q; want pipeline path %q only", pipelinePath, upstreamPath, tt.wantPath)
				}
			case passthrough:
				if upstreamPath != tt.wantPath || pipelinePath != "" {
					t.Fatalf("gateway path %q, pipeline path %q; want gateway path %q only", upstreamPath, pipelinePath, tt.wantPath)
				}
			case notAllowed:
				if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodPost {
					t.Fatalf("got %d with Allow %q, want 405 with Allow POST", rec.Code, rec.Header().Get("Allow"))
				}
				if pipelinePath != "" || upstreamPath != "" {
					t.Fatalf("pipeline path %q, gateway path %q; want neither reached", pipelinePath, upstreamPath)
				}
			case local:
				if rec.Code != http.StatusOK || pipelinePath != "" || upstreamPath != "" {
					t.Fatalf("got %d, pipeline path %q, gateway path %q; want 200 with neither reached", rec.Code, pipelinePath, upstreamPath)
				}
			}
		})
	}
}
