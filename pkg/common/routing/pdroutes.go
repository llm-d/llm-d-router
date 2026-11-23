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

package routing

import (
	"slices"
	"strings"

	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// PDHTTPPaths are the HTTP inference paths the sidecar serves with
// disaggregated prefill. The sidecar forwards any other path to its decoder.
var PDHTTPPaths = []string{
	reqcommon.PathChatCompletions,
	reqcommon.PathCompletions,
	reqcommon.PathMessages,
	reqcommon.PathResponses,
	reqcommon.PathVLLMGenerate,
	reqcommon.PathSGLangGenerate,
}

// PDVLLMGRPCMethods maps the vLLM gRPC methods the sidecar serves with
// disaggregated prefill to their request API type.
var PDVLLMGRPCMethods = map[string]reqcommon.APIType{
	vllmpb.Inference_Generate_FullMethodName:       reqcommon.APITypeVLLMGenerate,
	vllmpb.Inference_GenerateStream_FullMethodName: reqcommon.APITypeVLLMGenerate,
}

// HasPDRoute reports whether the sidecar serves a request on path with
// disaggregated prefill. An HTTP path matches as reqcommon.MatchAPIType matches
// it, ignoring a query, a trailing slash and a provider prefix. The SGLang
// Generate method counts although the sidecar serves it only with the sglang
// KV connector, because the caller cannot see the sidecar's flags.
func HasPDRoute(path string) bool {
	if apiType, ok := reqcommon.MatchAPIType(path); ok {
		return slices.Contains(PDHTTPPaths, apiType.Path())
	}
	method, _, _ := strings.Cut(path, "?")
	method = strings.TrimSuffix(method, "/")
	if _, ok := PDVLLMGRPCMethods[method]; ok {
		return true
	}
	return method == sglangapiv1.SglangService_Generate_FullMethodName
}
