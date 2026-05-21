/*
Copyright 2025 The Kubernetes Authors.

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
	"fmt"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc/status"
)

// RequestDroppedReasonHeaderKey is the HTTP response header that communicates the specific
// reason a request was dropped by flow control.
const RequestDroppedReasonHeaderKey = "x-llm-d-request-dropped-reason"

// RequestDroppedReason is the reason a request was rejected before dispatch or evicted after dispatch.
type RequestDroppedReason string

const (
	// Rejected — request never reached an inference server.
	RequestDroppedReasonSaturated        RequestDroppedReason = "rejected-saturated"
	RequestDroppedReasonTTLExpired       RequestDroppedReason = "rejected-ttl-expired"
	RequestDroppedReasonContextCancelled RequestDroppedReason = "rejected-context-cancelled"

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

// BuildErrResponse maps an error to an Envoy ImmediateResponse with the appropriate
// HTTP status code and error message body. If the error code is not recognized,
// it returns a gRPC error instead of an ImmediateResponse.
func BuildErrResponse(err error) (*extProcPb.ProcessingResponse, error) {
	var httpCode envoyTypePb.StatusCode

	switch CanonicalCode(err) {
	case BadRequest:
		httpCode = envoyTypePb.StatusCode_BadRequest
	case Unauthorized:
		httpCode = envoyTypePb.StatusCode_Unauthorized
	case Forbidden:
		httpCode = envoyTypePb.StatusCode_Forbidden
	case NotFound:
		httpCode = envoyTypePb.StatusCode_NotFound
	case ResourceExhausted:
		httpCode = envoyTypePb.StatusCode_TooManyRequests
	case Internal:
		httpCode = envoyTypePb.StatusCode_InternalServerError
	case ServiceUnavailable:
		httpCode = envoyTypePb.StatusCode_ServiceUnavailable
	default:
		return nil, status.Errorf(status.Code(err), "failed to handle request: %v", err)
	}

	ir := &extProcPb.ImmediateResponse{
		Status: &envoyTypePb.HttpStatus{
			Code: httpCode,
		},
	}

	if err.Error() != "" {
		ir.Body = []byte(err.Error())
	}

	if e, ok := err.(Error); ok && len(e.Headers) > 0 {
		setHeaders := make([]*configPb.HeaderValueOption, 0, len(e.Headers))
		for k, v := range e.Headers {
			setHeaders = append(setHeaders, &configPb.HeaderValueOption{
				Header: &configPb.HeaderValue{
					Key:      k,
					RawValue: []byte(v),
				},
			})
		}
		ir.Headers = &extProcPb.HeaderMutation{
			SetHeaders: setHeaders,
		}
	}

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: ir,
		},
	}, nil
}
