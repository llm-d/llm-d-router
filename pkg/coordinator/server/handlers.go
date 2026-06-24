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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"time"

	"github.com/google/uuid"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/coordinator/pkg/pipeline"
)

const maxRequestBodySize = 64 << 20 // 64 MB

// validRequestID bounds a client-supplied x-request-id to alphanumerics and
// dashes, at most 128 characters. handleInference replaces a header that fails
// the match with a generated UUID before it reaches error responses; the
// request-logging middleware redacts it. This keeps attacker-controlled content
// out of both surfaces.
var validRequestID = regexp.MustCompile(`^[a-zA-Z0-9\-]{1,128}$`)

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
	if parsed == nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}

	stream, _ := parsed["stream"].(bool)
	model, _ := parsed["model"].(string)

	requestID := r.Header.Get(reqcommon.RequestIDHeaderKey)
	clientRequestID := requestID
	requestIDReplaced := !validRequestID.MatchString(requestID)
	if requestIDReplaced {
		requestID = uuid.New().String()
	}

	reqCtx := &pipeline.RequestContext{
		RequestID:        requestID,
		OriginalPath:     r.URL.Path,
		OriginalHeaders:  r.Header.Clone(),
		OriginalBody:     body,
		Body:             parsed,
		Model:            model,
		Stream:           stream,
		KVTransferParams: make(map[string]any),
		ResponseWriter:   w,
		StartTime:        time.Now(),
	}

	logger := ctrl.Log.WithName("handler").WithValues(reqcommon.RequestIDHeaderKey, reqCtx.RequestID)
	ctx := log.IntoContext(r.Context(), logger)

	if requestIDReplaced && clientRequestID != "" {
		// Log the rejected length, never the raw value, to avoid reflecting
		// attacker-controlled content into the log.
		logger.V(logutil.DEFAULT).Info("replaced invalid client request ID", "rejectedLength", len(clientRequestID))
	}

	logger.V(logutil.DEFAULT).Info("received request", "path", r.URL.Path, "model", model, "stream", stream)

	if err := s.pipeline.Execute(ctx, reqCtx); err != nil {
		logger.Error(err, "pipeline execution failed")
		status, msg := classifyPipelineError(err, reqCtx.RequestID)
		http.Error(w, msg, status)
	}
}

// classifyPipelineError maps a pipeline error to a client-facing status and
// message. Invalid client input is 400. An upstream 4xx is forwarded with its
// status (the request was the root cause); any other upstream status, and every
// other error, is a 502 gateway fault. Upstream response bodies stay in the
// server log (logged by the caller) and are never sent to the client.
func classifyPipelineError(err error, requestID string) (int, string) {
	if errors.Is(err, pipeline.ErrBadRequest) {
		return http.StatusBadRequest, fmt.Sprintf("bad request (request_id: %s)", requestID)
	}
	var upstream *pipeline.UpstreamError
	if errors.As(err, &upstream) && upstream.StatusCode >= 400 && upstream.StatusCode < 500 {
		return upstream.StatusCode, fmt.Sprintf("%s rejected the request: HTTP %d (request_id: %s)", upstream.Step, upstream.StatusCode, requestID)
	}
	return http.StatusBadGateway, fmt.Sprintf("internal error (request_id: %s)", requestID)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}
