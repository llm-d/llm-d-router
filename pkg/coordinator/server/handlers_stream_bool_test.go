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
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// A non-bool stream is re-marshalled into the worker body unchanged; a worker
// that coerces it to true would stream while the coordinator ran the request as
// non-streaming. The coordinator rejects it as a bad request instead of running
// the pipeline.
func TestHandleInference_RejectsNonBoolStream(t *testing.T) {
	step := &capturingStep{}
	srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(&http.Transport{}, stubGatewayURL))
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m","stream":"true"}`))
	rec := httptest.NewRecorder()
	srv.handleInference(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400 for a non-bool stream, got %d (%s)", rec.Code, strings.TrimSpace(rec.Body.String()))
	}
	if step.captured != nil {
		t.Fatal("pipeline ran for a request that should have been rejected")
	}
}

// A real boolean stream is accepted and classified as streaming.
func TestHandleInference_AcceptsBoolStream(t *testing.T) {
	step := &capturingStep{}
	srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(&http.Transport{}, stubGatewayURL))
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m","stream":true}`))
	rec := httptest.NewRecorder()
	srv.handleInference(rec, req)

	if step.captured == nil {
		t.Fatalf("pipeline did not run for a valid request (status %d)", rec.Code)
	}
	if !step.captured.Stream {
		t.Fatal("request with stream:true was not classified as streaming")
	}
}
