package server

import (
	"context"
	"net"
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	ctrl "sigs.k8s.io/controller-runtime"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
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
	httpServer *http.Server
	pipeline   *pipeline.Pipeline
}

func New(cfg config.ServerConfig, p *pipeline.Pipeline) *Server {
	s := &Server{pipeline: p}

	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Recoverer)
	r.Use(logRequestResponse)

	r.Post(gateway.PathChatCompletions, s.handleChatCompletions)
	r.Post(gateway.PathCompletions, s.handleCompletions)
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
