package server

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

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

	r.Post("/v1/chat/completions", s.handleChatCompletions)
	r.Post("/v1/completions", s.handleCompletions)
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
