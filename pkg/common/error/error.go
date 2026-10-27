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

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
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
	Code    string
	Msg     string
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

// httpCodes maps a canonical code to the HTTP status it is answered with. The
// envelope derives its error type from the status (reqcommon.ErrorEnvelope).
var httpCodes = map[string]envoyTypePb.StatusCode{
	BadRequest:         envoyTypePb.StatusCode_BadRequest,
	Unauthorized:       envoyTypePb.StatusCode_Unauthorized,
	Forbidden:          envoyTypePb.StatusCode_Forbidden,
	NotFound:           envoyTypePb.StatusCode_NotFound,
	PreconditionFailed: envoyTypePb.StatusCode_PreconditionFailed,
	ResourceExhausted:  envoyTypePb.StatusCode_TooManyRequests,
	Internal:           envoyTypePb.StatusCode_InternalServerError,
	ServiceUnavailable: envoyTypePb.StatusCode_ServiceUnavailable,
}

// errorBody serializes the error in the envelope of the API the request was sent to.
// Messages quote request fragments, so HTML characters are written verbatim.
func errorBody(apiType reqcommon.APIType, e Error, httpCode envoyTypePb.StatusCode) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(reqcommon.ErrorEnvelope(apiType, int(httpCode), e.Msg)); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}

// BuildErrResponse maps an error to an Envoy ImmediateResponse with the appropriate
// HTTP status code and a JSON body in the error envelope of apiType. If the error
// code is not recognized, it returns a gRPC error instead of an ImmediateResponse.
func BuildErrResponse(err error, apiType reqcommon.APIType) (*extProcPb.ProcessingResponse, error) {
	httpCode, ok := httpCodes[CanonicalCode(err)]
	if !ok {
		return nil, status.Errorf(status.Code(err), "failed to handle request: %v", err)
	}

	e, _ := err.(Error)
	body, encodeErr := errorBody(apiType, e, httpCode)
	if encodeErr != nil {
		return nil, status.Errorf(codes.Internal, "failed to encode error response: %v", encodeErr)
	}

	contentType := &configPb.HeaderValueOption{
		Header: &configPb.HeaderValue{Key: reqcommon.HeaderContentType, RawValue: []byte(reqcommon.ContentTypeJSON)},
	}
	setHeaders := append([]*configPb.HeaderValueOption{contentType}, envoy.GenerateHeadersMutation(e.Headers)...)

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extProcPb.ImmediateResponse{
				Status:  &envoyTypePb.HttpStatus{Code: httpCode},
				Body:    body,
				Headers: &extProcPb.HeaderMutation{SetHeaders: setHeaders},
			},
		},
	}, nil
}
