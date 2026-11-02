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

package request

import (
	"encoding/json"
	"net/http"
)

// OpenAIErrorResponse is the error envelope of the OpenAI-compatible APIs.
type OpenAIErrorResponse struct {
	Error OpenAIError `json:"error"`
}

// OpenAIError carries the message, the error type and the HTTP status of a
// rejection.
type OpenAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    int    `json:"code"`
}

// OpenAIErrorType returns the error type the OpenAI envelope carries for an
// HTTP status.
func OpenAIErrorType(status int) string {
	switch {
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusForbidden:
		return "permission_error"
	case status == http.StatusNotFound:
		return "not_found_error"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status >= http.StatusInternalServerError:
		return "server_error"
	default:
		return "invalid_request_error"
	}
}

// AnthropicErrorResponse is the error envelope of the Anthropic Messages API.
type AnthropicErrorResponse struct {
	Type  string         `json:"type"`
	Error AnthropicError `json:"error"`
}

// AnthropicError carries the error type and the message of a rejection.
type AnthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// AnthropicErrorType returns the error type the Anthropic envelope carries for
// an HTTP status.
func AnthropicErrorType(status int) string {
	switch {
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusForbidden:
		return "permission_error"
	case status == http.StatusNotFound:
		return "not_found_error"
	case status == http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status == http.StatusServiceUnavailable:
		return "overloaded_error"
	case status >= http.StatusInternalServerError:
		return "api_error"
	default:
		return "invalid_request_error"
	}
}

// ErrorEnvelope returns the error envelope of the API a request was sent to:
// the Anthropic envelope for Messages, the OpenAI envelope for every other API.
func ErrorEnvelope(apiType APIType, status int, msg string) any {
	if apiType == APITypeMessages {
		return AnthropicErrorResponse{Type: "error", Error: AnthropicError{Type: AnthropicErrorType(status), Message: msg}}
	}
	return OpenAIErrorResponse{Error: OpenAIError{Message: msg, Type: OpenAIErrorType(status), Code: status}}
}

// WriteAPIError rejects an HTTP request with the error envelope of apiType (see
// ErrorEnvelope). It sets the status and the nosniff header that http.Error
// sets, with a JSON body in place of the plain text. Error messages quote
// request fragments, so HTML characters are written verbatim.
func WriteAPIError(w http.ResponseWriter, apiType APIType, status int, msg string) error {
	w.Header().Set(HeaderContentType, ContentTypeJSON)
	w.Header().Set(HeaderContentTypeOptions, ContentTypeOptionsNoSniff)
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return enc.Encode(ErrorEnvelope(apiType, status, msg))
}
