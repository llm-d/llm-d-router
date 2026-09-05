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

const (
	// ModelArchLabel is the pod label indicating the model architecture category.
	ModelArchLabel = "llm-d.ai/model-arch"

	// Model architecture values.
	ModelArchOmniLLM        = "omni-llm"
	ModelArchDiffusion      = "diffusion"
	ModelArchAutoRegressTTS = "autoregressive-tts"
	ModelArchEncoderDecSTT  = "encoder-decoder-stt"
	ModelArchAutoRegressLLM = "autoregressive-llm"

	// EnvoyPathHeader is the pseudo-header Envoy uses to pass the request path.
	EnvoyPathHeader = ":path"
)

// PathToModelArch maps API request paths to compatible model architecture labels.
var PathToModelArch = map[string][]string{
	AudioSpeechPath:         {ModelArchOmniLLM, ModelArchAutoRegressTTS},
	AudioTranscriptionsPath: {ModelArchEncoderDecSTT},
	ImagesGenerationsPath:   {ModelArchDiffusion},
}
