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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// auxRouteStep is a stub step that registers an auxiliary route.
type auxRouteStep struct {
	stubStep
	registered bool
}

func (s *auxRouteStep) RegisterRoutes(r chi.Router) {
	s.registered = true
	r.Get("/v1/requests/{id}", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(chi.URLParam(r, "id")))
	})
}

func TestServerRegistersStepRoutes(t *testing.T) {
	step := &auxRouteStep{stubStep: stubStep{name: "aux"}}
	p := pipeline.New([]pipeline.Step{stubStep{name: "plain"}, step})
	srv, err := New(config.ServerConfig{}, p)
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
	if rec.Body.String() != "abc-123" {
		t.Fatalf("expected path param in body, got %q", rec.Body.String())
	}

	// The built-in routes are untouched.
	req = httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec = httptest.NewRecorder()
	srv.httpServer.Handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 from /healthz, got %d", rec.Code)
	}
}
