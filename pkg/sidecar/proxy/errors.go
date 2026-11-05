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
	"fmt"
	"net/http"
	"strings"

	"google.golang.org/grpc/codes"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

var errInvalidJSON = errors.New("invalid JSON")

// errPrefillAborted reports a prefill response whose body failed to arrive
// after its status did.
var errPrefillAborted = errors.New("prefill response interrupted")

// JSON errors can contain payload values and client-supplied field paths.
func jsonErrorForLog(err error) error {
	var syntaxError *json.SyntaxError
	if errors.As(err, &syntaxError) {
		return fmt.Errorf("invalid JSON syntax at byte %d", syntaxError.Offset)
	}
	var typeError *json.UnmarshalTypeError
	if errors.As(err, &typeError) {
		return fmt.Errorf("invalid JSON value at byte %d", typeError.Offset)
	}
	return err
}

func errorJSONInvalid(w http.ResponseWriter, r *http.Request, err error) error {
	return sendError(w, r, http.StatusBadRequest, err.Error())
}

// errorBadGateway answers a failed call to another pod (prefill, decoder,
// encoder, bootstrap). The raw error can carry internal routing details such
// as dial addresses, so callers log it and the client gets the fixed message
// the coordinator answers upstream connection failures with.
func errorBadGateway(w http.ResponseWriter, r *http.Request) error {
	reqID := ""
	if r != nil {
		reqID = r.Header.Get(reqcommon.RequestIDHeaderKey)
	}
	return sendError(w, r, http.StatusBadGateway, reqcommon.InternalErrorMessage(reqID))
}

func errorInternalServerError(w http.ResponseWriter, r *http.Request, err error) error {
	return sendError(w, r, http.StatusInternalServerError, err.Error())
}

// sendError writes msg in the error envelope of the API r was sent to. A path
// that is not an inference route, which only the decoder catch-all serves, and
// a request or URL net/http never produces both get the OpenAI envelope. gRPC
// requests receive a trailers-only gRPC status response.
func sendError(w http.ResponseWriter, r *http.Request, status int, msg string) error {
	if isGRPCRequest(r) {
		grpcframe.WriteError(w, httpStatusToGRPCCode(status), msg)
		return nil
	}
	path := ""
	if r != nil && r.URL != nil {
		path = r.URL.Path
	}
	return reqcommon.WriteAPIError(w, reqcommon.DetectAPIType(path), status, msg)
}

// grpcMethodRoutes maps served gRPC method paths to their request API types.
var grpcMethodRoutes = map[string]reqcommon.APIType{
	vllmpb.Inference_Generate_FullMethodName:       reqcommon.APITypeVLLMGenerate,
	vllmpb.Inference_GenerateStream_FullMethodName: reqcommon.APITypeVLLMGenerate,
}

func isVLLMGRPCPath(p string) bool {
	_, ok := grpcMethodRoutes[strings.TrimSuffix(p, "/")]
	return ok
}

func isVLLMGRPCRequest(r *http.Request) bool {
	return r != nil && r.URL != nil && isVLLMGRPCPath(r.URL.Path)
}

func isGRPCRequest(r *http.Request) bool {
	if r == nil {
		return false
	}
	if grpcframe.IsContentType(r.Header.Get(reqcommon.HeaderContentType)) {
		return true
	}
	return isVLLMGRPCRequest(r)
}

func httpStatusToGRPCCode(status int) codes.Code {
	switch status {
	case http.StatusBadRequest:
		return codes.InvalidArgument
	case http.StatusUnauthorized:
		return codes.Unauthenticated
	case http.StatusForbidden:
		return codes.PermissionDenied
	case http.StatusNotFound:
		return codes.NotFound
	case http.StatusTooManyRequests:
		return codes.ResourceExhausted
	case http.StatusBadGateway, http.StatusServiceUnavailable:
		return codes.Unavailable
	case http.StatusGatewayTimeout:
		return codes.DeadlineExceeded
	default:
		return codes.Internal
	}
}
