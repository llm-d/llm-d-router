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

package envoy

import (
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc/status"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
)

// BuildImmediateResponse builds an ImmediateResponse: Envoy answers the caller
// with httpCode, headers and body, and forwards nothing upstream. An empty
// header map adds no header mutation.
func BuildImmediateResponse(httpCode envoyTypePb.StatusCode, headers map[string]string, body []byte) *extProcPb.ProcessingResponse {
	ir := &extProcPb.ImmediateResponse{
		Status: &envoyTypePb.HttpStatus{
			Code: httpCode,
		},
		Body: body,
	}
	if len(headers) > 0 {
		ir.Headers = &extProcPb.HeaderMutation{
			SetHeaders: GenerateHeadersMutation(headers),
		}
	}
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: ir,
		},
	}
}

// BuildErrResponse maps an error to an Envoy ImmediateResponse with the appropriate
// HTTP status code, the error message as body, and the headers the error carries.
// If the error code is not recognized, it returns a gRPC error instead of an
// ImmediateResponse.
func BuildErrResponse(err error) (*extProcPb.ProcessingResponse, error) {
	var httpCode envoyTypePb.StatusCode

	switch errcommon.CanonicalCode(err) {
	case errcommon.BadRequest:
		httpCode = envoyTypePb.StatusCode_BadRequest
	case errcommon.Unauthorized:
		httpCode = envoyTypePb.StatusCode_Unauthorized
	case errcommon.Forbidden:
		httpCode = envoyTypePb.StatusCode_Forbidden
	case errcommon.NotFound:
		httpCode = envoyTypePb.StatusCode_NotFound
	case errcommon.PreconditionFailed:
		httpCode = envoyTypePb.StatusCode_PreconditionFailed
	case errcommon.ResourceExhausted:
		httpCode = envoyTypePb.StatusCode_TooManyRequests
	case errcommon.Internal:
		httpCode = envoyTypePb.StatusCode_InternalServerError
	case errcommon.ServiceUnavailable:
		httpCode = envoyTypePb.StatusCode_ServiceUnavailable
	default:
		return nil, status.Errorf(status.Code(err), "failed to handle request: %v", err)
	}

	var headers map[string]string
	if e, ok := err.(errcommon.Error); ok {
		headers = e.Headers
	}
	return BuildImmediateResponse(httpCode, headers, []byte(err.Error())), nil
}
