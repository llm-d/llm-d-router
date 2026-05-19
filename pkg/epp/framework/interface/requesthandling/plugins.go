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

package requesthandling

import (
	"context"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// Parser defines the interface for parsing payload(requests and responses).
type Parser interface {
	fwkplugin.Plugin
	// ParseRequest parses the request body and headers and returns the parsed result.
	// There are three outcomes based on the return values:
	// 1. err != nil: The request is invalid or cannot be parsed. The framework will fail the request early.
	// 2. err == nil and result.Skip == true: The request is valid but should not be processed by the scheduling
	//    framework (e.g., non-inference requests). The framework will fall back to routing the request to a random
	//    endpoint and skip subsequent interception phases.
	// 3. err == nil and result.Skip == false: The request is valid and will be processed by the scheduling framework.
	ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*ParseResult, error)

	// ParseResponse parses the response payload.
	// For streaming responses , this method is invoked multiple times (once per chunk),
	// where 'endOfStream' is set to true only for the final chunk.
	// For non-streaming responses, this method is invoked exactly once with the full
	// buffered response body and 'endOfStream' set to true.
	ParseResponse(ctx context.Context, body []byte, headers map[string]string, endofStream bool) (*ParsedResponse, error)

	// SupportedAppProtocols returns the list of supported protocols.
	// Returning an empty list means it supports all protocols.
	SupportedAppProtocols() []v1.AppProtocol
}

// ParseResult contains the result of parsing the request.
type ParseResult struct {
	// Body contains the parsed inference request body.
	Body *InferenceRequestBody
	// Skip indicates whether to skip the remaining processing phases in the EPP.
	// This should only be used for non-inference related requests.
	// When set to true, the request will bypass the scheduling director and will be
	// routed to a random endpoint. The EPP will also stop intercepting the stream
	// for this request (e.g., response headers and body will not be processed).
	//
	// Example: In a gRPC parser, if the request path does not match any known
	// methods (e.g., neither Generate nor Embed for vLLM), setting Skip to true
	// allows the request to proceed to a fallback endpoint without failing.
	Skip bool
}

type ParsedResponse struct {
	// Usage is only populate when the raw response has usage.
	Usage *Usage
}
