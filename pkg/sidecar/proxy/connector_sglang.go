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
	"maps"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"sigs.k8s.io/controller-runtime/pkg/log"
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
		if err := errorJSONInvalid(err, w); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Send concurrent prefill and decode requests
	s.runConcurrentPD(w, r, body, body, prefillPodHostPort, constants.KVConnectorSGLang, nil, toolFields)
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
