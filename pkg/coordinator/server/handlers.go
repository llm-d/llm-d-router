package server

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.handleInference(w, r)
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	s.handleInference(w, r)
}

const maxRequestBodySize = 64 << 20 // 64 MB

func (s *Server) handleInference(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBodySize+1))
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	if len(body) > maxRequestBodySize {
		http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
		return
	}

	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}

	stream, _ := parsed["stream"].(bool)
	model, _ := parsed["model"].(string)

	flusher, ok := w.(http.Flusher)
	if !ok && stream {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	reqCtx := &pipeline.RequestContext{
		RequestID:        uuid.New().String(),
		OriginalPath:     r.URL.Path,
		OriginalBody:     body,
		Body:             parsed,
		Model:            model,
		Stream:           stream,
		KVTransferParams: make(map[string]any),
		ResponseWriter:   w,
		Flusher:          flusher,
		StartTime:        time.Now(),
	}

	if err := s.pipeline.Execute(r.Context(), reqCtx); err != nil {
		slog.Error("pipeline execution failed", "request_id", reqCtx.RequestID, "error", err)
		http.Error(w, err.Error(), http.StatusBadGateway)
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}
