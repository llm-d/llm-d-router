/*
Copyright 2025 The Kubernetes Authors.
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

package error

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/llm-d/llm-d-router/pkg/common/envoy"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// RequestDroppedReasonHeaderKey is the HTTP response header that communicates the specific
// reason the EPP dropped a request.
const RequestDroppedReasonHeaderKey = "x-llm-d-request-dropped-reason"

// RequestDroppedReason is the reason a request was rejected before dispatch or evicted after dispatch.
type RequestDroppedReason string

const (
	// Rejected — request never reached an inference server.
	RequestDroppedReasonSaturated        RequestDroppedReason = "rejected-saturated"
	RequestDroppedReasonNoEndpoints      RequestDroppedReason = "rejected-no-endpoints"
	RequestDroppedReasonTTLExpired       RequestDroppedReason = "rejected-ttl-expired"
	RequestDroppedReasonContextCancelled RequestDroppedReason = "rejected-context-cancelled"
	RequestDroppedReasonShuttingDown     RequestDroppedReason = "rejected-shutting-down"
	RequestDroppedReasonInternal         RequestDroppedReason = "rejected-internal"

	// Evicted — request was dispatched to an inference server and then killed.
	// The generic "evicted" reason is the current default used by ImmediateResponseEvictor.Evict().
	// The specific sub-reasons are forward-looking for when EvictN() callers specify why they're evicting.
	RequestDroppedReasonEvicted              RequestDroppedReason = "evicted"
	RequestDroppedReasonEvictedQueuePressure RequestDroppedReason = "evicted-queue-pressure"
	RequestDroppedReasonEvictedPriority      RequestDroppedReason = "evicted-priority"
)

// Error is an error struct for errors returned by the epp/bbr server.
type Error struct {
	Code string
	Msg  string
	// Headers are set on the error response. The response body is JSON, so a
	// content-type entry is replaced by application/json.
	Headers map[string]string
}

const (
	Unknown            = "Unknown"
	BadRequest         = "BadRequest"
	Unauthorized       = "Unauthorized"
	Forbidden          = "Forbidden"
	NotFound           = "NotFound"
	PreconditionFailed = "PreconditionFailed"
	Internal           = "Internal"
	ServiceUnavailable = "ServiceUnavailable"
	ModelServerError   = "ModelServerError"
	ResourceExhausted  = "ResourceExhausted"
)

// Error returns a string version of the error.
func (e Error) Error() string {
	return fmt.Sprintf("inference error: %s - %s", e.Code, e.Msg)
}

// CanonicalCode returns the error's ErrorCode.
func CanonicalCode(err error) string {
	e, ok := err.(Error)
	if ok {
		return e.Code
	}
	return Unknown
}

// openAIErrorResponse is the error envelope of the OpenAI-compatible APIs.
type openAIErrorResponse struct {
	Error openAIError `json:"error"`
}

type openAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    int    `json:"code"`
}

// anthropicErrorResponse is the error envelope of the Anthropic Messages API.
type anthropicErrorResponse struct {
	Type  string         `json:"type"`
	Error anthropicError `json:"error"`
}

type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// wireCode is how a canonical code appears on the wire: the HTTP status and the
// error type of each API's envelope.
type wireCode struct {
	httpCode  envoyTypePb.StatusCode
	openAI    string
	anthropic string
}

var wireCodes = map[string]wireCode{
	BadRequest:         {envoyTypePb.StatusCode_BadRequest, "invalid_request_error", "invalid_request_error"},
	Unauthorized:       {envoyTypePb.StatusCode_Unauthorized, "authentication_error", "authentication_error"},
	Forbidden:          {envoyTypePb.StatusCode_Forbidden, "permission_error", "permission_error"},
	NotFound:           {envoyTypePb.StatusCode_NotFound, "not_found_error", "not_found_error"},
	PreconditionFailed: {envoyTypePb.StatusCode_PreconditionFailed, "invalid_request_error", "invalid_request_error"},
	ResourceExhausted:  {envoyTypePb.StatusCode_TooManyRequests, "rate_limit_error", "rate_limit_error"},
	Internal:           {envoyTypePb.StatusCode_InternalServerError, "server_error", "api_error"},
	ServiceUnavailable: {envoyTypePb.StatusCode_ServiceUnavailable, "server_error", "overloaded_error"},
}

// errorBody serializes the error in the envelope of the API the request was sent to.
// Messages quote request fragments, so HTML characters are written verbatim.
func errorBody(apiType reqcommon.APIType, e Error, code wireCode) ([]byte, error) {
	var envelope any = openAIErrorResponse{
		Error: openAIError{Message: e.Msg, Type: code.openAI, Code: int(code.httpCode)},
	}
	if apiType == reqcommon.APITypeMessages {
		envelope = anthropicErrorResponse{
			Type:  "error",
			Error: anthropicError{Type: code.anthropic, Message: e.Msg},
		}
	}

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(envelope); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}

// BuildErrResponse maps an error to an Envoy ImmediateResponse with the appropriate
// HTTP status code and a JSON body in the error envelope of apiType. If the error
// code is not recognized, it returns a gRPC error instead of an ImmediateResponse.
func BuildErrResponse(err error, apiType reqcommon.APIType) (*extProcPb.ProcessingResponse, error) {
	code, ok := wireCodes[CanonicalCode(err)]
	if !ok {
		return nil, status.Errorf(status.Code(err), "failed to handle request: %v", err)
	}

	e, _ := err.(Error)
	body, encodeErr := errorBody(apiType, e, code)
	if encodeErr != nil {
		return nil, status.Errorf(codes.Internal, "failed to encode error response: %v", encodeErr)
	}

	headers := make(map[string]string, len(e.Headers)+1)
	maps.Copy(headers, e.Headers)
	headers[reqcommon.HeaderContentType] = reqcommon.ContentTypeJSON
	return envoy.BuildImmediateResponse(code.httpCode, headers, body), nil
}
