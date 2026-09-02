package disagg

import (
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

// hasMultimodalContent reports whether the tokenized prompt carries any
// multimodal features. Detection is protocol-agnostic: it relies on the
// token-producer plugin having populated PromptTokens.MultiModalFeatures.
func hasMultimodalContent(request *scheduling.InferenceRequest) bool {
	if request == nil {
		return false
	}
	tp, ok := scheduling.ReadRequestAttribute[*scheduling.TokenizedRequest](request, tokenproducer.TokenizedPromptDataKey)
	if !ok {
		return false
	}
	for _, p := range tp.Prompts {
		if len(p.MultiModalFeatures) > 0 {
			return true
		}
	}
	return false
}
