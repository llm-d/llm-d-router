package disagg

import (
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// hasMultimodalContent returns true if the request contains any image, video, or audio content blocks.
func hasMultimodalContent(request *scheduling.InferenceRequest) bool {
	if request == nil || request.Body == nil || request.Body.ChatCompletions == nil {
		return false
	}
	for _, msg := range request.Body.ChatCompletions.Messages {
		// See https://github.com/vllm-project/vllm/blob/main/docs/features/multimodal_inputs.md#online-serving
		for _, block := range msg.Content.Structured {
			if block.Type == "image_url" || block.Type == "video_url" || block.Type == "input_audio" {
				return true
			}
		}
	}
	return false
}
