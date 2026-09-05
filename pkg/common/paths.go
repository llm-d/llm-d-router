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

package common

// API request paths shared by the coordinator, EPP, and sidecar.
const (
	ChatCompletionsPath     = "/v1/chat/completions"
	CompletionsPath         = "/v1/completions"
	AudioSpeechPath         = "/v1/audio/speech"
	AudioTranscriptionsPath = "/v1/audio/transcriptions"
	ImagesGenerationsPath   = "/v1/images/generations"
	EmbeddingsPath          = "/v1/embeddings"
	InferencePath           = "/v1/inference"
	// GeneratePath is vLLM's token-in generate endpoint.
	GeneratePath = "/inference/v1/generate"
)
