package disagg

import (
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// hasMultimodalContent reports whether the tokenized prompt carries any
// multimodal features. Detection is protocol-agnostic: it relies on the
// token-producer plugin having populated TokenizedPrompt.MultiModalFeatures.
func hasMultimodalContent(request *scheduling.InferenceRequest) bool {
	if request == nil || request.Body == nil || request.Body.TokenizedPrompt == nil {
		return false
	}
	for _, perPrompt := range request.Body.TokenizedPrompt.MultiModalFeatures {
		if len(perPrompt) > 0 {
			return true
		}
	}
	return false
}
