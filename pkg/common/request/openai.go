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

package request

import "maps"

const imageURLPartType = "image_url"

// ImageParts returns image_url content parts in message order.
// The returned parts reference the input maps.
func ImageParts(body map[string]any) []map[string]any {
	messages, _ := body["messages"].([]any)
	var parts []map[string]any
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		content, ok := msgMap["content"].([]any)
		if !ok {
			continue
		}
		for _, part := range content {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}
			if partMap["type"] == imageURLPartType {
				parts = append(parts, partMap)
			}
		}
	}
	return parts
}

// SingleImageChatRequest builds a non-streaming, single-output-token request
// containing the selected image. An out-of-range index produces an empty URL.
// The image_url value is shared with the selected input part.
func SingleImageChatRequest(model string, imageParts []map[string]any, index int) map[string]any {
	var imageURL any = map[string]any{"url": ""}
	if index >= 0 && index < len(imageParts) {
		imageURL = imageParts[index][imageURLPartType]
	}
	body := map[string]any{
		"model": model,
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": imageURLPartType, imageURLPartType: imageURL},
				},
			},
		},
	}
	PrimeSingleTokenRequest(body)
	return body
}

// SingleTokenChatRequest copies a chat body into a non-streaming,
// single-output-token request. Nested values remain shared with the input
// and must not be mutated through the returned body.
func SingleTokenChatRequest(body map[string]any) map[string]any {
	body = maps.Clone(body)
	PrimeSingleTokenRequest(body)
	return body
}

// SingleTokenCompletionRequest builds a non-streaming, single-output-token
// request with a text or token prompt.
func SingleTokenCompletionRequest(model string, prompt any) map[string]any {
	body := map[string]any{"model": model, "prompt": prompt}
	PrimeSingleTokenRequest(body)
	return body
}
