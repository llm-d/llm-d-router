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
	"fmt"
	"math"
	"net"
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	ctrl "sigs.k8s.io/controller-runtime"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

var serverLog = ctrl.Log.WithName("server")

var (
	loggedRequestHeaders  = []string{"Content-Type", reqcommon.RequestIDHeaderKey, gateway.EPPPhaseHeader, "Prefer"}
	loggedResponseHeaders = []string{"Content-Type", reqcommon.RequestIDHeaderKey}
)

func pickHeaders(h http.Header, names []string) map[string]string {
	out := make(map[string]string, len(names))
	for _, n := range names {
		v := h.Get(n)
		if v == "" {
			continue
		}
		if n == reqcommon.RequestIDHeaderKey && !validRequestID.MatchString(v) {
			v = "<redacted>"
		}
		out[n] = v
	}
	return out
}

func logRequestResponse(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log := serverLog.V(logutil.DEBUG)
		if !log.Enabled() {
			next.ServeHTTP(w, r)
			return
		}
		log.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"headers", pickHeaders(r.Header, loggedRequestHeaders))
		ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
		next.ServeHTTP(ww, r)
		log.Info("response",
			"method", r.Method,
			"path", r.URL.Path,
			"status", ww.Status(),
			"headers", pickHeaders(ww.Header(), loggedResponseHeaders))
	})
}

type Server struct {
	httpServer         *http.Server
	pipeline           *pipeline.Pipeline
	maxRequestBodySize int64
}

func New(cfg config.ServerConfig, p *pipeline.Pipeline) *Server {
	maxBodySize := cfg.MaxRequestBodySize
	if maxBodySize == 0 {
		// Zero means unset; Viper fills this from the config default in
		// production. Direct callers (e.g. tests) that leave it unset get
		// the same default.
		maxBodySize = config.DefaultMaxRequestBodySize
	}
	if maxBodySize < 0 {
		panic(fmt.Sprintf("server: MaxRequestBodySize must be positive, got %d", maxBodySize))
	}
	if maxBodySize > (math.MaxInt64-1)/config.BytesPerMB {
		// maxRequestBodySize*1024*1024+1 is used as the io.LimitReader sentinel;
		// an MB value that overflows int64 when converted to bytes would cause
		// LimitReader to receive a negative limit and return immediate EOF.
		panic(fmt.Sprintf("server: MaxRequestBodySize must be at most %d MB, got %d", int64((math.MaxInt64-1)/config.BytesPerMB), maxBodySize))
	}
	s := &Server{pipeline: p, maxRequestBodySize: maxBodySize}

	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Recoverer)
	r.Use(logRequestResponse)

	r.Post(gateway.PathChatCompletions, s.handleInference)
	r.Post(gateway.PathCompletions, s.handleInference)
	r.Get("/healthz", s.handleHealth)
	r.Get("/readyz", s.handleHealth)

	s.httpServer = &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      r,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}

	return s
}

func (s *Server) ListenAndServe() error {
	return s.httpServer.ListenAndServe()
}

func (s *Server) Serve(l net.Listener) error {
	return s.httpServer.Serve(l)
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}
