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
	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

// BuildImmediateResponse builds an ImmediateResponse: Envoy answers the caller
// with httpCode, setHeaders and body, and forwards nothing upstream.
func BuildImmediateResponse(httpCode envoyTypePb.StatusCode, setHeaders []*corev3.HeaderValueOption, body []byte) *extProcPb.ProcessingResponse {
	ir := &extProcPb.ImmediateResponse{
		Status: &envoyTypePb.HttpStatus{Code: httpCode},
		Body:   body,
	}
	if len(setHeaders) > 0 {
		ir.Headers = &extProcPb.HeaderMutation{SetHeaders: setHeaders}
	}
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{ImmediateResponse: ir},
	}
}
