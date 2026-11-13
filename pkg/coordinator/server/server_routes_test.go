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
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"net/textproto"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
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
		{http.MethodPost, "/v1/responses", served, "/v1/responses"},
		{http.MethodPost, "/v1/chat/completions/", served, "/v1/chat/completions"},
		{http.MethodPost, "/v1/completions/", served, "/v1/completions"},
		{http.MethodPost, "/inference/v1/generate/", served, "/inference/v1/generate"},
		{http.MethodPost, "/v1/responses/", served, "/v1/responses"},
		{http.MethodPost, "/v1/messages", served, "/v1/messages"},
		{http.MethodPost, "/v1/messages/", served, "/v1/messages"},
		{http.MethodPost, "//v1/chat/completions", served, "/v1/chat/completions"},
		{http.MethodPost, "/v1/chat//completions", served, "/v1/chat/completions"},
		{http.MethodPost, "/v1/messages/.", served, "/v1/messages"},
		{http.MethodPost, "/v1/chat%2Fcompletions", served, "/v1/chat/completions"},
		{http.MethodPost, "/v1/messages%2F", served, "/v1/messages"},
		{http.MethodPost, "/sglang.api.v1.SglangService/Generate", served, "/sglang.api.v1.SglangService/Generate"},
		{http.MethodPost, "/sglang.api.v1.SglangService/Generate/", served, "/sglang.api.v1.SglangService/Generate"},
		{http.MethodPost, "/sglang.api.v1.SglangService%2FGenerate", served, "/sglang.api.v1.SglangService/Generate"},
		{http.MethodPost, "/vllm.Inference/Generate", served, "/vllm.Inference/Generate"},
		{http.MethodPost, "/vllm.Inference/Generate/", served, "/vllm.Inference/Generate"},
		{http.MethodPost, "/vllm.Inference%2FGenerate", served, "/vllm.Inference/Generate"},
		{http.MethodPost, "/vllm.Inference/GenerateStream", served, "/vllm.Inference/GenerateStream"},
		{http.MethodPost, "/vllm.Inference/GenerateStream/", served, "/vllm.Inference/GenerateStream"},
		{http.MethodPost, "/vllm.Inference%2FGenerateStream", served, "/vllm.Inference/GenerateStream"},

		{http.MethodPost, "/sglang.api.v1.SglangService/HealthCheck", passthrough, "/sglang.api.v1.SglangService/HealthCheck"},
		{http.MethodPost, "/vllm.Control/GetServerInfo", passthrough, "/vllm.Control/GetServerInfo"},
		{http.MethodPut, "/generate", passthrough, "/generate"},
		{http.MethodPost, "/generate", passthrough, "/generate"},
		{http.MethodPost, "/generate/", passthrough, "/generate/"},
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
		{http.MethodPatch, "/v1/files/file_1", passthrough, "/v1/files/file_1"},
		{http.MethodGet, "/v1/models", passthrough, "/v1/models"},
		{http.MethodGet, "/health", passthrough, "/health"},
		{http.MethodGet, "/v1/requests/req_1", passthrough, "/v1/requests/req_1"},

		{http.MethodGet, "/v1/chat/completions/", passthrough, "/v1/chat/completions/"},

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
			var reqBody io.Reader = strings.NewReader(`{"model":"m"}`)
			if strings.Contains(tt.path, "SglangService") || strings.Contains(tt.path, "vllm.") {
				reqBody = bytes.NewReader(make([]byte, grpcframe.HeaderLen))
			}
			req := httptest.NewRequest(tt.method, tt.path, reqBody)

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
			case local:
				if rec.Code != http.StatusOK || pipelinePath != "" || upstreamPath != "" {
					t.Fatalf("got %d, pipeline path %q, gateway path %q; want 200 with neither reached", rec.Code, pipelinePath, upstreamPath)
				}
			}
		})
	}
}

// TestServerRouteMethodNotAllowed asserts the 405 a registered route answers
// for any other method, before the request reaches the pipeline or the gateway.
func TestServerRouteMethodNotAllowed(t *testing.T) {
	tests := []struct {
		method    string
		path      string
		wantAllow string
	}{
		{http.MethodGet, "/v1/chat/completions", http.MethodPost},
		{http.MethodDelete, "/v1/completions", http.MethodPost},
		{http.MethodHead, "/inference/v1/generate", http.MethodPost},
		{http.MethodPut, "/inference/v1/generate", http.MethodPost},
		{http.MethodOptions, "/v1/chat/completions", http.MethodPost},
		{http.MethodGet, "/v1/responses", http.MethodPost},
		{http.MethodHead, "/v1/messages", http.MethodPost},
		{http.MethodGet, "/sglang.api.v1.SglangService/Generate", http.MethodPost},
		{http.MethodGet, "/vllm.Inference/Generate", http.MethodPost},
		{http.MethodGet, "/vllm.Inference/GenerateStream", http.MethodPost},
		{http.MethodPost, "/healthz", http.MethodGet},
		{http.MethodHead, "/readyz", http.MethodGet},
		{"PROPFIND", "/v1/files", ""},
	}
	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			upstream, cap := newCapturingUpstream(t, http.StatusOK, "")
			reached := false
			step := stubStep{name: "record", fn: func(context.Context, *pipeline.RequestContext) error {
				reached = true
				return nil
			}}
			srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(&http.Transport{}, upstream.URL))
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			rec := doPassthrough(t, srv, httptest.NewRequest(tt.method, tt.path, nil))

			if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != tt.wantAllow {
				t.Fatalf("got %d with Allow %q, want 405 with Allow %s", rec.Code, rec.Header().Get("Allow"), tt.wantAllow)
			}
			if _, upstreamPath, _, _, _ := cap.get(); reached || upstreamPath != "" {
				t.Fatalf("pipeline reached %v, gateway path %q; want neither", reached, upstreamPath)
			}
		})
	}
}

// TestCanonicalizeInferencePath asserts which requests the middleware rewrites
// and that the caller's request is left as sent.
func TestCanonicalizeInferencePath(t *testing.T) {
	routes := chi.NewRouter()
	routes.Post("/v1/responses", func(http.ResponseWriter, *http.Request) {})
	routes.Post(sglangapiv1.SglangService_Generate_FullMethodName, func(http.ResponseWriter, *http.Request) {})
	routes.Post(vllmpb.Inference_Generate_FullMethodName, func(http.ResponseWriter, *http.Request) {})
	routes.Post(vllmpb.Inference_GenerateStream_FullMethodName, func(http.ResponseWriter, *http.Request) {})
	routes.Post("/v1/requests/{id}/cancel", func(http.ResponseWriter, *http.Request) {})
	tests := []struct {
		name        string
		method      string
		target      string
		wantPath    string
		wantRawPath string
	}{
		{name: "trailing slash", method: http.MethodPost, target: "/v1/responses/?x=1", wantPath: "/v1/responses"},
		{name: "encoded slash", method: http.MethodPost, target: "/v1/responses%2F", wantPath: "/v1/responses"},
		{name: "exact path", method: http.MethodPost, target: "/v1/responses", wantPath: "/v1/responses"},
		{name: "sglang grpc generate trailing slash", method: http.MethodPost, target: "/sglang.api.v1.SglangService/Generate/", wantPath: "/sglang.api.v1.SglangService/Generate"},
		{name: "sglang grpc healthcheck unserved", method: http.MethodPost, target: "/sglang.api.v1.SglangService/HealthCheck/", wantPath: "/sglang.api.v1.SglangService/HealthCheck/"},
		{name: "vllm grpc generate trailing slash", method: http.MethodPost, target: "/vllm.Inference/Generate/", wantPath: "/vllm.Inference/Generate"},
		{name: "vllm grpc generate stream trailing slash", method: http.MethodPost, target: "/vllm.Inference/GenerateStream/", wantPath: "/vllm.Inference/GenerateStream"},
		{name: "vllm grpc control unserved", method: http.MethodPost, target: "/vllm.Control/GetServerInfo/", wantPath: "/vllm.Control/GetServerInfo/"},
		{name: "other method", method: http.MethodGet, target: "/v1/responses/", wantPath: "/v1/responses/"},
		{name: "unserved path", method: http.MethodPost, target: "/generate/", wantPath: "/generate/"},
		{name: "step route", method: http.MethodPost, target: "/v1/requests/abc/cancel/", wantPath: "/v1/requests/abc/cancel/"},
		{name: "sub-resource", method: http.MethodPost, target: "/v1/responses/compact/", wantPath: "/v1/responses/compact/"},
		{name: "unserved escaped path", method: http.MethodPost, target: "/v1/files/a%2Fb", wantPath: "/v1/files/a/b", wantRawPath: "/v1/files/a%2Fb"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got *http.Request
			h := canonicalizeInferencePath(routes)(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) { got = r }))
			req := httptest.NewRequest(tt.method, tt.target, nil)
			origPath, origRawPath := req.URL.Path, req.URL.RawPath

			h.ServeHTTP(httptest.NewRecorder(), req)

			if got.URL.Path != tt.wantPath || got.URL.RawPath != tt.wantRawPath {
				t.Errorf("next saw (Path %q, RawPath %q), want (%q, %q)", got.URL.Path, got.URL.RawPath, tt.wantPath, tt.wantRawPath)
			}
			if req.URL.Path != origPath || req.URL.RawPath != origRawPath {
				t.Errorf("caller's request was modified to (Path %q, RawPath %q)", req.URL.Path, req.URL.RawPath)
			}
			if got.URL.RawQuery != req.URL.RawQuery {
				t.Errorf("query = %q, want %q", got.URL.RawQuery, req.URL.RawQuery)
			}
		})
	}
}

// TestServerRejectsStatefulResponsesFieldsOnEveryServedForm asserts the
// stateful-field guard runs on every form of PathResponses that is served.
func TestServerRejectsStatefulResponsesFieldsOnEveryServedForm(t *testing.T) {
	const body = `{"model":"m","input":"hi","previous_response_id":"resp-123"}`
	for _, path := range []string{"/v1/responses", "/v1/responses/", "//v1/responses", "/v1/./responses", "/v1//responses/", "/v1%2Fresponses", "/v1/responses%2F"} {
		t.Run(path, func(t *testing.T) {
			upstream, cap := newCapturingUpstream(t, http.StatusOK, "")
			reached := false
			step := stubStep{name: "record", fn: func(context.Context, *pipeline.RequestContext) error {
				reached = true
				return nil
			}}
			srv, err := New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(&http.Transport{}, upstream.URL))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))

			rec := doPassthrough(t, srv, req)

			if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), reqcommon.FieldPreviousResponseID) {
				t.Fatalf("got %d %q, want 400 naming %s", rec.Code, rec.Body.String(), reqcommon.FieldPreviousResponseID)
			}
			if _, upstreamPath, _, _, _ := cap.get(); reached || upstreamPath != "" {
				t.Fatalf("pipeline reached %v, gateway path %q; want neither", reached, upstreamPath)
			}
		})
	}
}

func TestServerDrain_InFlightResponse(t *testing.T) {
	tests := []struct {
		name       string
		earlyHints bool
		respond    func(http.ResponseWriter) error
		wantStatus int
		wantBody   string
	}{
		{
			name: "explicit headers",
			respond: func(w http.ResponseWriter) error {
				w.WriteHeader(http.StatusCreated)
				_, err := w.Write([]byte("done"))
				return err
			},
			wantStatus: http.StatusCreated,
			wantBody:   "done",
		},
		{
			name: "implicit headers",
			respond: func(w http.ResponseWriter) error {
				_, err := w.Write([]byte("done"))
				return err
			},
			wantStatus: http.StatusOK,
			wantBody:   "done",
		},
		{
			name: "flush",
			respond: func(w http.ResponseWriter) error {
				return http.NewResponseController(w).Flush()
			},
			wantStatus: http.StatusOK,
		},
		{
			name:       "empty response",
			wantStatus: http.StatusOK,
		},
		{
			name:       "informational headers",
			earlyHints: true,
			respond: func(w http.ResponseWriter) error {
				w.WriteHeader(http.StatusCreated)
				return nil
			},
			wantStatus: http.StatusCreated,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var srv *Server
			step := stubStep{name: "respond", fn: func(_ context.Context, rc *pipeline.RequestContext) error {
				if err := http.NewResponseController(rc.ResponseWriter).SetWriteDeadline(time.Time{}); err != nil {
					return err
				}
				if tt.earlyHints {
					rc.ResponseWriter.WriteHeader(http.StatusEarlyHints)
				}
				srv.StartDrain()
				if tt.respond != nil {
					return tt.respond(rc.ResponseWriter)
				}
				return nil
			}}
			var err error
			srv, err = New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(nil, stubGatewayURL))
			if err != nil {
				t.Fatal(err)
			}
			ts := httptest.NewServer(srv.httpServer.Handler)
			defer ts.Close()

			resp, err := ts.Client().Post(ts.URL+reqcommon.PathChatCompletions, "application/json", strings.NewReader(`{"model":"m","stream":false}`))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != tt.wantStatus || string(body) != tt.wantBody {
				t.Fatalf("response = %d %q, want %d %q", resp.StatusCode, body, tt.wantStatus, tt.wantBody)
			}
			if !resp.Close {
				t.Fatal("response completed during drain must close its connection")
			}
		})
	}
}

func TestCloseWhenDraining_CommittedResponse(t *testing.T) {
	srv := &Server{}
	rec := httptest.NewRecorder()
	handler := srv.closeWhenDraining(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		srv.StartDrain()
		if _, err := w.Write([]byte("done")); err != nil {
			t.Error(err)
		}
	}))
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if got := rec.Header().Get("Connection"); got != "" {
		t.Fatalf("committed response header changed to Connection: %s", got)
	}
	if rec.Body.String() != "done" {
		t.Fatalf("body = %q, want done", rec.Body.String())
	}
}

func TestCloseWhenDraining_SwitchingProtocols(t *testing.T) {
	srv := &Server{}
	srv.StartDrain()
	rec := httptest.NewRecorder()
	handler := srv.closeWhenDraining(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Connection", "Upgrade")
		w.WriteHeader(http.StatusSwitchingProtocols)
	}))
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusSwitchingProtocols {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusSwitchingProtocols)
	}
	if got := rec.Header().Get("Connection"); got != "Upgrade" {
		t.Fatalf("Connection = %q, want Upgrade", got)
	}
}

func TestServerDrain_HTTP2InformationalResponse(t *testing.T) {
	for _, tt := range []struct {
		name   string
		status int
	}{
		{name: "continue", status: http.StatusContinue},
		{name: "early hints", status: http.StatusEarlyHints},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var srv *Server
			step := stubStep{name: "respond", fn: func(_ context.Context, rc *pipeline.RequestContext) error {
				srv.StartDrain()
				rc.ResponseWriter.WriteHeader(tt.status)
				rc.ResponseWriter.WriteHeader(http.StatusCreated)
				_, err := rc.ResponseWriter.Write([]byte("done"))
				return err
			}}
			var err error
			srv, err = New(config.ServerConfig{}, pipeline.New([]pipeline.Step{step}), gateway.NewWithTransport(nil, stubGatewayURL))
			if err != nil {
				t.Fatal(err)
			}
			ts := httptest.NewUnstartedServer(srv.httpServer.Handler)
			ts.EnableHTTP2 = true
			ts.StartTLS()
			defer ts.Close()

			var gotStatus int
			var gotConnection string
			trace := &httptrace.ClientTrace{
				Got1xxResponse: func(status int, header textproto.MIMEHeader) error {
					gotStatus = status
					gotConnection = header.Get("Connection")
					return nil
				},
			}
			ctx := httptrace.WithClientTrace(t.Context(), trace)
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+reqcommon.PathChatCompletions, strings.NewReader(`{"model":"m","stream":false}`))
			if err != nil {
				t.Fatal(err)
			}
			resp, err := ts.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if resp.ProtoMajor != 2 || resp.StatusCode != http.StatusCreated || string(body) != "done" {
				t.Fatalf("response = %s %d %q, want HTTP/2 201 done", resp.Proto, resp.StatusCode, body)
			}
			if gotStatus != tt.status || gotConnection != "" {
				t.Fatalf("informational response = %d with Connection %q, want %d without Connection", gotStatus, gotConnection, tt.status)
			}
		})
	}
}
