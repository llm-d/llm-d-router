package server

import (
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/coordinator/pkg/logging"
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

	logger := ctrl.Log.WithName("handler").WithValues("request_id", reqCtx.RequestID)
	ctx := logging.IntoContext(r.Context(), logger)

	logger.V(logging.DEFAULT).Info("received request", "path", r.URL.Path, "model", model, "stream", stream)

	if err := s.pipeline.Execute(ctx, reqCtx); err != nil {
		logger.Error(err, "pipeline execution failed")
		http.Error(w, err.Error(), http.StatusBadGateway)
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}
