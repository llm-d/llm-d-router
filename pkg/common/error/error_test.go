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
	"errors"
	"maps"
	"testing"

	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestError_Error(t *testing.T) {
	tests := []struct {
		name string
		err  Error
		want string
	}{
		{
			name: "BadRequest error",
			err: Error{
				Code: BadRequest,
				Msg:  "invalid model name",
			},
			want: "inference error: BadRequest - invalid model name",
		},
		{
			name: "Internal error",
			err: Error{
				Code: Internal,
				Msg:  "unexpected condition",
			},
			want: "inference error: Internal - unexpected condition",
		},
		{
			name: "ServiceUnavailable error",
			err: Error{
				Code: ServiceUnavailable,
				Msg:  "service unavailable",
			},
			want: "inference error: ServiceUnavailable - service unavailable",
		},
		{
			name: "ModelServerError",
			err: Error{
				Code: ModelServerError,
				Msg:  "connection timeout",
			},
			want: "inference error: ModelServerError - connection timeout",
		},
		{
			name: "ResourceExhausted error",
			err: Error{
				Code: ResourceExhausted,
				Msg:  "no available pods",
			},
			want: "inference error: ResourceExhausted - no available pods",
		},
		{
			name: "Unknown error",
			err: Error{
				Code: Unknown,
				Msg:  "something went wrong",
			},
			want: "inference error: Unknown - something went wrong",
		},
		{
			name: "Empty message",
			err: Error{
				Code: BadRequest,
				Msg:  "",
			},
			want: "inference error: BadRequest - ",
		},
		{
			name: "Empty code",
			err: Error{
				Code: "",
				Msg:  "error occurred",
			},
			want: "inference error:  - error occurred",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.want {
				t.Errorf("Error.Error() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCanonicalCode(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "Error type with BadRequest code",
			err: Error{
				Code: BadRequest,
				Msg:  "invalid input",
			},
			want: BadRequest,
		},
		{
			name: "Error type with Internal code",
			err: Error{
				Code: Internal,
				Msg:  "server error",
			},
			want: Internal,
		},
		{
			name: "Error type with ServiceUnavailable code",
			err: Error{
				Code: ServiceUnavailable,
				Msg:  "Service unavailable error",
			},
			want: ServiceUnavailable,
		},
		{
			name: "Error type with ModelServerError code",
			err: Error{
				Code: ModelServerError,
				Msg:  "model unavailable",
			},
			want: ModelServerError,
		},
		{
			name: "Error type with ResourceExhausted code",
			err: Error{
				Code: ResourceExhausted,
				Msg:  "no resources",
			},
			want: ResourceExhausted,
		},
		{
			name: "Error type with Unknown code",
			err: Error{
				Code: Unknown,
				Msg:  "unknown error",
			},
			want: Unknown,
		},
		{
			name: "Error type with empty code",
			err: Error{
				Code: "",
				Msg:  "no code provided",
			},
			want: "",
		},
		{
			name: "Non-Error type",
			err:  errors.New("standard go error"),
			want: Unknown,
		},
		{
			name: "Nil error",
			err:  nil,
			want: Unknown,
		},
		{
			name: "Custom error type that is not Error",
			err:  customError{msg: "custom error"},
			want: Unknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CanonicalCode(tt.err); got != tt.want {
				t.Errorf("CanonicalCode() = %v, want %v", got, tt.want)
			}
		})
	}
}

// customError is a helper type for testing non-Error error types
type customError struct {
	msg string
}

func (e customError) Error() string {
	return e.msg
}

func TestErrorConstants(t *testing.T) {
	// Verify that error constants match their expected string values
	tests := map[string]string{
		Unknown:            "Unknown",
		BadRequest:         "BadRequest",
		Internal:           "Internal",
		ServiceUnavailable: "ServiceUnavailable",
		ModelServerError:   "ModelServerError",
		ResourceExhausted:  "ResourceExhausted",
	}

	for constant, expected := range tests {
		if constant != expected {
			t.Errorf("Constant value %q != expected %q", constant, expected)
		}
	}
}

func TestBuildErrResponse(t *testing.T) {
	tests := []struct {
		name           string
		err            error
		apiType        reqcommon.APIType
		wantHTTPStatus envoyTypePb.StatusCode
		wantBody       string
		wantGRPCErr    bool
		wantHeaders    map[string]string
	}{
		{
			name:           "BadRequest returns 400",
			err:            Error{Code: BadRequest, Msg: "invalid model name"},
			wantHTTPStatus: envoyTypePb.StatusCode_BadRequest,
			wantBody:       `{"error":{"message":"invalid model name","type":"invalid_request_error","code":400}}`,
		},
		{
			name:           "Unauthorized returns 401",
			err:            Error{Code: Unauthorized, Msg: "missing token"},
			wantHTTPStatus: envoyTypePb.StatusCode_Unauthorized,
			wantBody:       `{"error":{"message":"missing token","type":"authentication_error","code":401}}`,
		},
		{
			name:           "Forbidden returns 403",
			err:            Error{Code: Forbidden, Msg: "unsafe content blocked"},
			wantHTTPStatus: envoyTypePb.StatusCode_Forbidden,
			wantBody:       `{"error":{"message":"unsafe content blocked","type":"permission_error","code":403}}`,
		},
		{
			name:           "NotFound returns 404",
			err:            Error{Code: NotFound, Msg: "model not found"},
			wantHTTPStatus: envoyTypePb.StatusCode_NotFound,
			wantBody:       `{"error":{"message":"model not found","type":"not_found_error","code":404}}`,
		},
		{
			name:           "PreconditionFailed returns 412",
			err:            Error{Code: PreconditionFailed, Msg: "revision mismatch"},
			wantHTTPStatus: envoyTypePb.StatusCode_PreconditionFailed,
			wantBody:       `{"error":{"message":"revision mismatch","type":"invalid_request_error","code":412}}`,
		},
		{
			name:           "ResourceExhausted returns 429",
			err:            Error{Code: ResourceExhausted, Msg: "no capacity"},
			wantHTTPStatus: envoyTypePb.StatusCode_TooManyRequests,
			wantBody:       `{"error":{"message":"no capacity","type":"rate_limit_error","code":429}}`,
		},
		{
			name:           "Internal returns 500",
			err:            Error{Code: Internal, Msg: "unexpected failure"},
			wantHTTPStatus: envoyTypePb.StatusCode_InternalServerError,
			wantBody:       `{"error":{"message":"unexpected failure","type":"server_error","code":500}}`,
		},
		{
			name:           "ServiceUnavailable returns 503",
			err:            Error{Code: ServiceUnavailable, Msg: "no endpoints"},
			wantHTTPStatus: envoyTypePb.StatusCode_ServiceUnavailable,
			wantBody:       `{"error":{"message":"no endpoints","type":"server_error","code":503}}`,
		},
		{
			name:           "message is JSON-escaped",
			err:            Error{Code: BadRequest, Msg: `json: cannot unmarshal string into Go struct field "messages"`},
			wantHTTPStatus: envoyTypePb.StatusCode_BadRequest,
			wantBody:       `{"error":{"message":"json: cannot unmarshal string into Go struct field \"messages\"","type":"invalid_request_error","code":400}}`,
		},
		{
			name:           "HTML characters are not escaped",
			err:            Error{Code: BadRequest, Msg: "invalid character '<' looking for beginning of value"},
			wantHTTPStatus: envoyTypePb.StatusCode_BadRequest,
			wantBody:       `{"error":{"message":"invalid character '<' looking for beginning of value","type":"invalid_request_error","code":400}}`,
		},
		{
			name:           "Messages API BadRequest uses the Anthropic envelope",
			err:            Error{Code: BadRequest, Msg: "must have valid messages field"},
			apiType:        reqcommon.APITypeMessages,
			wantHTTPStatus: envoyTypePb.StatusCode_BadRequest,
			wantBody:       `{"type":"error","error":{"type":"invalid_request_error","message":"must have valid messages field"}}`,
		},
		{
			name:           "Messages API Internal maps to api_error",
			err:            Error{Code: Internal, Msg: "no pods available in datastore"},
			apiType:        reqcommon.APITypeMessages,
			wantHTTPStatus: envoyTypePb.StatusCode_InternalServerError,
			wantBody:       `{"type":"error","error":{"type":"api_error","message":"no pods available in datastore"}}`,
		},
		{
			name:           "Messages API ServiceUnavailable maps to overloaded_error",
			err:            Error{Code: ServiceUnavailable, Msg: "no endpoints"},
			apiType:        reqcommon.APITypeMessages,
			wantHTTPStatus: envoyTypePb.StatusCode_ServiceUnavailable,
			wantBody:       `{"type":"error","error":{"type":"overloaded_error","message":"no endpoints"}}`,
		},
		{
			name:        "plain error returns gRPC error",
			err:         errors.New("unknown problem"),
			wantGRPCErr: true,
		},
		{
			name:           "headers include content-type",
			err:            Error{Code: ResourceExhausted, Msg: "no capacity", Headers: map[string]string{RequestDroppedReasonHeaderKey: string(RequestDroppedReasonSaturated)}},
			wantHTTPStatus: envoyTypePb.StatusCode_TooManyRequests,
			wantBody:       `{"error":{"message":"no capacity","type":"rate_limit_error","code":429}}`,
			wantHeaders:    map[string]string{RequestDroppedReasonHeaderKey: string(RequestDroppedReasonSaturated)},
		},
		{
			name:           "content-type in the error's headers is replaced",
			err:            Error{Code: ResourceExhausted, Msg: "no capacity", Headers: map[string]string{"content-type": "text/plain"}},
			wantHTTPStatus: envoyTypePb.StatusCode_TooManyRequests,
			wantBody:       `{"error":{"message":"no capacity","type":"rate_limit_error","code":429}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := BuildErrResponse(tt.err, tt.apiType)

			if tt.wantGRPCErr {
				if err == nil {
					t.Fatal("expected gRPC error, got nil")
				}
				if resp != nil {
					t.Errorf("expected nil response on gRPC error, got %v", resp)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp == nil {
				t.Fatal("expected response, got nil")
			}

			ir := resp.GetImmediateResponse()
			if ir == nil {
				t.Fatal("expected ImmediateResponse, got nil")
			}
			if ir.GetStatus().GetCode() != tt.wantHTTPStatus {
				t.Errorf("HTTP status = %v, want %v", ir.GetStatus().GetCode(), tt.wantHTTPStatus)
			}
			if got := string(ir.GetBody()); got != tt.wantBody {
				t.Errorf("body = %s, want %s", got, tt.wantBody)
			}

			setHeaders := ir.GetHeaders().GetSetHeaders()
			gotHeaders := make(map[string]string, len(setHeaders))
			for _, h := range setHeaders {
				gotHeaders[h.GetHeader().GetKey()] = string(h.GetHeader().GetRawValue())
			}
			wantHeaders := map[string]string{"content-type": "application/json"}
			maps.Copy(wantHeaders, tt.wantHeaders)
			if len(setHeaders) != len(wantHeaders) || !maps.Equal(gotHeaders, wantHeaders) {
				t.Errorf("headers = %v, want %v", gotHeaders, wantHeaders)
			}
		})
	}
}
