/*
Copyright 2025 The llm-d Authors.

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

package proxy

import (
	"encoding/json"
	"errors"
	"maps"
	"math/rand/v2"
	"net/http"
	"os"
	"slices"
	"strconv"
	"time"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/sglangwire"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

var (
	sglangBootstrapPort int
)

func init() {
	// Default SGLang bootstrap port
	sglangBootstrapPort = 8998

	// Override from environment variable if set
	if portStr := os.Getenv("SGLANG_BOOTSTRAP_PORT"); portStr != "" {
		if port, err := strconv.Atoi(portStr); err == nil {
			sglangBootstrapPort = port
		}
	}
}

func (s *Server) handleSGLang(w http.ResponseWriter, r *http.Request, prefillPodHostPort string) {
	if (r.URL != nil && r.URL.Path == sglangapiv1.SglangService_Generate_FullMethodName) ||
		grpcframe.IsContentType(r.Header.Get(reqcommon.HeaderContentType)) {
		s.handleSGLangGRPC(w, r, prefillPodHostPort)
		return
	}

	logger := log.FromContext(r.Context())
	logger.V(logging.DEBUG).Info("running SGLang protocol", "url", prefillPodHostPort)

	original, requestData, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}
	toolFields, _ := captureInferenceToolCallingRequest(r, reqcommon.DetectAPIType(r.URL.Path), original)

	roomID := s.generateSGLangRoomID()
	// SGLang keys its P/D transfer state and log lines by the bootstrap room.
	trace.SpanFromContext(r.Context()).SetAttributes(semconv.LLMDPDProxyBootstrapRoom(roomID))

	// Inject bootstrap info for both prefill and decode
	bootstrapInfo := s.addSGLangBootstrapInfo(logger, requestData, prefillPodHostPort, roomID)

	body, err := json.Marshal(bootstrapInfo)
	if err != nil {
		if err := errorJSONInvalid(w, r, err); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Send concurrent prefill and decode requests
	s.runConcurrentPD(w, r, body, body, prefillPodHostPort, constants.KVConnectorSGLang, nil, toolFields)
}

func (s *Server) handleSGLangGRPC(w http.ResponseWriter, r *http.Request, prefillPodHostPort string) {
	logger := log.FromContext(r.Context())
	logger.V(logging.DEBUG).Info("running SGLang gRPC protocol", "url", prefillPodHostPort)

	defer r.Body.Close()
	// ReadMessage rejects a declared length over the limit before it
	// allocates, and data after the one frame of a unary request.
	frame, err := grpcframe.ReadMessage(r.Body, grpcframe.MaxPayloadLen)
	if err != nil {
		logger.V(logging.DEBUG).Info("invalid gRPC request frame", "error", err)
		code, msg := codes.InvalidArgument, "invalid GenerateRequest frame"
		switch {
		case errors.Is(err, grpcframe.ErrCompressed):
			code, msg = codes.Unimplemented, err.Error()
		case errors.Is(err, grpcframe.ErrTooLarge):
			code, msg = codes.ResourceExhausted, err.Error()
		}
		grpcframe.WriteError(w, code, msg)
		return
	}
	scanned, err := sglangwire.ScanGenerateRequest(frame[grpcframe.HeaderLen:])
	if err != nil {
		logger.V(logging.DEBUG).Info("invalid GenerateRequest protobuf", "error", err)
		grpcframe.WriteError(w, codes.InvalidArgument, "invalid GenerateRequest")
		return
	}

	roomID := s.generateSGLangRoomID()
	trace.SpanFromContext(r.Context()).SetAttributes(semconv.LLMDPDProxyBootstrapRoom(roomID))

	bootstrapHost := extractHost(prefillPodHostPort)
	head, tail, err := sglangwire.LegEnvelope(len(scanned.Payload), &sglangwire.Bootstrap{
		Host: bootstrapHost,
		Port: int64(sglangBootstrapPort),
		Room: roomID,
	})
	if err != nil {
		logger.Error(err, "failed to encode GenerateRequest")
		grpcframe.WriteError(w, codes.ResourceExhausted, err.Error())
		return
	}
	logger.V(logging.TRACE).Info("bootstrap info added",
		"bootstrap_host", bootstrapHost,
		"bootstrap_port", sglangBootstrapPort,
		"bootstrap_room", roomID)

	body := slices.Concat(head, scanned.Payload, tail)
	s.runConcurrentPD(w, r, body, body, prefillPodHostPort, constants.KVConnectorSGLang, nil, nil)
}

func (s *Server) addSGLangBootstrapInfo(logger logr.Logger, requestData map[string]interface{}, prefillHostPort string, roomID int64) map[string]interface{} {
	modifiedRequest := maps.Clone(requestData)

	// Generate bootstrap host from prefill host
	bootstrapHost := extractHost(prefillHostPort)

	// Add bootstrap information
	modifiedRequest[reqcommon.FieldBootstrapHost] = bootstrapHost
	modifiedRequest[reqcommon.FieldBootstrapPort] = sglangBootstrapPort
	modifiedRequest[reqcommon.FieldBootstrapRoom] = roomID

	logger.V(logging.TRACE).Info("bootstrap info added",
		"bootstrap_host", bootstrapHost,
		"bootstrap_port", sglangBootstrapPort,
		"bootstrap_room", roomID)

	return modifiedRequest
}

func (s *Server) generateSGLangRoomID() int64 {
	return time.Now().UnixNano() + int64(rand.IntN(1000)) //#nosec G404 -- non-crypto use, a room ID disambiguator
}
