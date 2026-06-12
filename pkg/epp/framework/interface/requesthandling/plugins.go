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
	// 2. err == nil and result.SkipResponseProcessing == true: The request is valid but EPP should stop intercepting the stream
	//    after the request phase. The scheduling director will still route the request, but subsequent
	//    response interception phases will be skipped.
	// 3. err == nil and result.SkipResponseProcessing == false: The request is valid and will be processed by the scheduling framework.
	ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*ParseResult, error)

	// ParseResponse parses the response payload.
	// For streaming responses , this method is invoked multiple times (once per chunk),
	// where 'endOfStream' is set to true only for the final chunk.
	// For non-streaming responses, this method is invoked exactly once with the full
	// buffered response body and 'endOfStream' set to true.
	ParseResponse(ctx context.Context, body []byte, headers map[string]string, endofStream bool) (*ParsedResponse, error)

	// Claims returns the paths and protocols claimed by this parser.
	Claims() Claims
}

// Claims defines the matching criteria for a parser.
type Claims struct {
	Paths     []string         // path patterns this parser claims (e.g., "chat/completions")
	Protocols []v1.AppProtocol // protocols this parser supports (e.g., "h2c")
}

// ParseResult contains the result of parsing the request.
type ParseResult struct {
	// Body contains the parsed inference request body.
	Body *InferenceRequestBody
	// SkipResponseProcessing indicates whether to skip EPP stream interception for this request.
	// When set to true, the request will still go through the scheduling director
	// (allowing routing decisions, profiles, and admission control to run),
	// but the EPP will stop intercepting the stream after the request phase completes
	// (e.g., response headers and body will not be processed).
	//
	// This allows fallback or non-standard requests to be routed using the configured
	// policies without paying the overhead of response-phase interception.
	SkipResponseProcessing bool
}

type ParsedResponse struct {
	// Usage is only populate when the raw response has usage.
	Usage *Usage
}
