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

package disagg

import (
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// hasMultimodalContent reports whether the tokenized prompt carries any
// multimodal features. Detection is protocol-agnostic: it relies on the
// token-producer plugin having populated PromptTokens.MultiModalFeatures.
func hasMultimodalContent(request *scheduling.InferenceRequest) bool {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return false
	}
	for _, p := range request.Body.TokenizedRequest.Prompts {
		if len(p.MultiModalFeatures) > 0 {
			return true
		}
	}
	return false
}
