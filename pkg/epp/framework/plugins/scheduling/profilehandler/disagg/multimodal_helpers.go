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
