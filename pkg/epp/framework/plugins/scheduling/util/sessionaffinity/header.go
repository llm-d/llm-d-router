/*
Copyright 2026 The Kubernetes Authors.

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

// Package sessionaffinity provides the session header helpers shared by the
// session-affinity scorer and filter plugins.
package sessionaffinity

import (
	"context"
	"encoding/base64"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// DefaultHeader is the default request/response header carrying the session
// token.
const DefaultHeader = "x-session-token"

// NormalizeHeader lowercases and trims the configured session header name,
// falling back to DefaultHeader when empty. Request headers are lowercased at
// ingestion, so the configured name must be lowercased to match.
func NormalizeHeader(name string) string {
	header := strings.ToLower(strings.TrimSpace(name))
	if header == "" {
		return DefaultHeader
	}
	return header
}

// DecodePodName decodes a base64-encoded session token into a pod
// NamespacedName string. It returns "" when the token is empty or cannot be
// decoded.
func DecodePodName(ctx context.Context, token string) string {
	if token == "" {
		return ""
	}
	decoded, err := base64.StdEncoding.DecodeString(token)
	if err != nil {
		log.FromContext(ctx).Error(err, "Error decoding session header")
		return ""
	}
	return string(decoded)
}

// WriteResponseHeader encodes targetPod into sessionHeader on the response sent
// to the client; pluginType labels the originating plugin in logs.
// TODO: this should be using a cookie and ensure not overriding any other
// cookie values if present.
// Tracked in https://github.com/llm-d/llm-d-router/issues/28
func WriteResponseHeader(ctx context.Context, pluginType, sessionHeader string, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	if response == nil || targetPod == nil {
		reqID := "undefined"
		if response != nil {
			reqID = response.RequestID
		}
		log.FromContext(ctx).V(logutil.DEBUG).Info("Session affinity - skip response header because response or targetPod is nil", "plugin", pluginType, "req id", reqID)
		return
	}

	if response.Headers == nil { // TODO should always be populated?
		response.Headers = make(map[string]string)
	}

	response.Headers[sessionHeader] = base64.StdEncoding.EncodeToString([]byte(targetPod.NamespacedName.String()))
}

// ResolvePodToWrite looks up the target pod from the scheduling results if profileName is set.
// When profileName is empty, targetPod (the primary/decode pod) is returned.
// When profileName is set, the function returns the profile's endpoint or nil
// if the profile was not scheduled (e.g. decode-only requests skip prefill).
func ResolvePodToWrite(request *scheduling.InferenceRequest, profileName string, targetPod *datalayer.EndpointMetadata) *datalayer.EndpointMetadata {
	if profileName == "" {
		return targetPod
	}
	if request != nil && request.SchedulingResult != nil {
		if result := request.SchedulingResult.ProfileResults[profileName]; result != nil && len(result.TargetEndpoints) > 0 && result.TargetEndpoints[0] != nil {
			if md := result.TargetEndpoints[0].GetMetadata(); md != nil {
				return md
			}
		}
	}
	return nil
}
