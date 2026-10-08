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

import "testing"

// The rule over every part type this package names, on both APIs that carry
// parts. Asserted as a full cross product rather than per case, because the
// coordinator's walkers and the sidecar's encoder fan-out now pair media with
// multimodal entries on this one answer: a part type that changed sides on
// either API would shift that pairing rather than fail outright.
func TestPartModality(t *testing.T) {
	for _, tc := range []struct {
		partType     string
		chatModality Modality // "" means not media on chat completions
		respModality Modality // "" means not media on Responses
	}{
		// vLLM's chat parser primes input_image and image_url through the same
		// content part map, so an image on a chat request is named either way.
		{PartTypeImageURL, ModalityImage, ""},
		{PartTypeInputImage, ModalityImage, ModalityImage},
		// Audio and video are chat-completions only: the Responses input
		// content union is input_text / input_image / input_file.
		{PartTypeAudioURL, ModalityAudio, ""},
		{PartTypeInputAudio, ModalityAudio, ""},
		{PartTypeVideoURL, ModalityVideo, ""},
		// In the Responses union but not media a walk collects, and the
		// part types that name no media on either API.
		{PartTypeInputFile, "", ""},
		{PartTypeComputerScreenshot, "", ""},
		{"text", "", ""},
		{"input_text", "", ""},
		{"image_embeds", "", ""},
		{"", "", ""},
	} {
		t.Run(tc.partType, func(t *testing.T) {
			for _, api := range []struct {
				apiType APIType
				want    Modality
			}{
				{APITypeChatCompletions, tc.chatModality},
				{APITypeResponses, tc.respModality},
			} {
				got, ok := PartModality(tc.partType, api.apiType)
				if ok != (api.want != "") {
					t.Errorf("PartModality(%q, %v) ok = %v, want %v", tc.partType, api.apiType, ok, api.want != "")
				}
				if got != api.want {
					t.Errorf("PartModality(%q, %v) = %q, want %q", tc.partType, api.apiType, got, api.want)
				}
			}
		})
	}
}

// A media part type reports no modality on an API that carries no part arrays
// at all, so a token-in or legacy-completions body cannot pick up a part a
// client never sent through a content array.
func TestPartModality_NonPartAPIs(t *testing.T) {
	for _, apiType := range []APIType{APITypeCompletions, APITypeVLLMGenerate, APITypeSGLangGenerate, APITypeMessages} {
		t.Run(apiType.String(), func(t *testing.T) {
			// Only the Responses carve-out is keyed off apiType, so every other
			// API answers as chat completions does. That is deliberate: these
			// APIs reach no part walk (steps.promptItems returns nothing for
			// them), and a modality here would be read by no caller.
			if got, ok := PartModality(PartTypeAudioURL, apiType); !ok || got != ModalityAudio {
				t.Errorf("PartModality(audio_url, %v) = %q/%v, want audio/true", apiType, got, ok)
			}
		})
	}
}
