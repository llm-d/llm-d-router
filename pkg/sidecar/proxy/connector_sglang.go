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
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
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
	s.logger.V(logging.DEBUG).Info("running SGLang protocol", "url", prefillPodHostPort)

	// Make Request
	requestData, err := s.parseSGLangRequest(r)

	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	roomID := s.generateSGLangRoomID()

	// Inject bootstrap info for both prefill and decode
	bootstrapInfo := s.addSGLangBootstrapInfo(requestData, prefillPodHostPort, roomID)

	body, err := json.Marshal(bootstrapInfo)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Send concurrent prefill and decode requests
	s.runConcurrentPD(w, r, body, body, prefillPodHostPort, KVConnectorSGLang, nil)
}

func (s *Server) addSGLangBootstrapInfo(requestData map[string]interface{}, prefillHostPort string, roomID int64) map[string]interface{} {
	modifiedRequest := make(map[string]interface{})
	for k, v := range requestData {
		modifiedRequest[k] = v
	}

	// Generate bootstrap host from prefill host
	bootstrapHost := extractHost(prefillHostPort)

	// Add bootstrap information
	modifiedRequest[requestFieldBootstrapHost] = bootstrapHost
	modifiedRequest[requestFieldBootstrapPort] = sglangBootstrapPort
	modifiedRequest[requestFieldBootstrapRoom] = roomID

	s.logger.V(logging.TRACE).Info("bootstrap info added",
		"bootstrap_host", bootstrapHost,
		"bootstrap_port", sglangBootstrapPort,
		"bootstrap_room", roomID)

	return modifiedRequest
}

func (s *Server) parseSGLangRequest(r *http.Request) (map[string]interface{}, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read request body: %w", err)
	}

	var requestData map[string]interface{}
	if err := json.Unmarshal(body, &requestData); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	return requestData, nil
}

func (s *Server) generateSGLangRoomID() int64 {
	return time.Now().UnixNano() + int64(rand.IntN(1000))
}
