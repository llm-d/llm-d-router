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
	"net/http"
)

var errInvalidJSON = errors.New("invalid JSON")

// vLLM error response
type errorResponse struct {
	Object  string `json:"object"`
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    int    `json:"code"`
}

var decoderServiceUnavailableResponseJSON []byte

func init() {
	response := errorResponse{
		Object:  "error",
		Message: "The decode node is not ready. Please check that the vLLM service is running and the port configuration is correct.",
		Type:    "ServiceUnavailable",
		Code:    http.StatusServiceUnavailable,
	}
	decoderServiceUnavailableResponseJSON, _ = json.Marshal(response)
}

func errorJSONInvalid(err error, w http.ResponseWriter) error {
	return sendError(err, "BadRequestError", http.StatusBadRequest, w)
}

func errorBadGateway(err error, w http.ResponseWriter) error {
	return sendError(err, "BadGateway", http.StatusBadGateway, w)
}

func errorInternalServerError(err error, w http.ResponseWriter) error {
	return sendError(err, "InternalServerError", http.StatusInternalServerError, w)
}

// sendError simulates vLLM errors
//
// Example:
//
//	 {
//		  "object": "error",
//		  "message": "[{'type': 'json_invalid', 'loc': ('body', 167), 'msg': 'JSON decode error', 'input': {}, 'ctx': {'error': 'Invalid control character at'}}]",
//		  "type": "BadRequestError",
//		  "param": null,
//		  "code": 400
//	 }
func sendError(err error, errorType string, code int, w http.ResponseWriter) error {
	er := errorResponse{
		Object:  "error",
		Message: err.Error(),
		Type:    errorType,
		Code:    code,
	}

	b, err := json.Marshal(er)
	if err != nil {
		return err
	}

	w.WriteHeader(code)
	_, err = w.Write(b)
	return err
}
