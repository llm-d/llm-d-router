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

// The rule over every part type this package names, asserted as a full table:
// the coordinator's walkers and the sidecar's encoder fan-out pair media with
// multimodal entries on this one answer, so a part type that changed sides
// would shift that pairing rather than fail outright.
//
// The answer is the same on chat completions and Responses. A Messages request
// names media with its own block types (TestMessagesBlockModality). Beyond
// that, which API carried the part decides only which arrays an item holds
// (ItemPartArrays) and whether a part walk runs at all (steps.promptItems).
func TestPartModality(t *testing.T) {
	for _, tc := range []struct {
		partType string
		want     Modality // "" means the type names no media
	}{
		// vLLM's chat parser primes input_image and image_url through the same
		// content part map, so an image is named either way.
		{PartTypeImageURL, ModalityImage},
		{PartTypeInputImage, ModalityImage},
		{PartTypeAudioURL, ModalityAudio},
		{PartTypeInputAudio, ModalityAudio},
		{PartTypeVideoURL, ModalityVideo},
		// Part types that name no media a walk collects.
		{PartTypeInputFile, ""},
		{PartTypeComputerScreenshot, ""},
		{"text", ""},
		{"input_text", ""},
		{"image_embeds", ""},
		{"", ""},
	} {
		t.Run(tc.partType, func(t *testing.T) {
			got, ok := PartModality(tc.partType)
			if ok != (tc.want != "") {
				t.Errorf("PartModality(%q) ok = %v, want %v", tc.partType, ok, tc.want != "")
			}
			if got != tc.want {
				t.Errorf("PartModality(%q) = %q, want %q", tc.partType, got, tc.want)
			}
		})
	}
}

// The coordinator's walkers and the sidecar's encoder fan-out both read a
// Messages request's media through this rule.
func TestMessagesBlockModality(t *testing.T) {
	for _, tc := range []struct {
		blockType string
		want      Modality // "" means the block names no media
	}{
		{BlockTypeImage, ModalityImage},
		{BlockTypeToolResult, ""},
		{"text", ""},
		// A chat part type is not a Messages block.
		{PartTypeImageURL, ""},
		{PartTypeInputAudio, ""},
		{"", ""},
	} {
		t.Run(tc.blockType, func(t *testing.T) {
			got, ok := MessagesBlockModality(tc.blockType)
			if ok != (tc.want != "") {
				t.Errorf("MessagesBlockModality(%q) ok = %v, want %v", tc.blockType, ok, tc.want != "")
			}
			if got != tc.want {
				t.Errorf("MessagesBlockModality(%q) = %q, want %q", tc.blockType, got, tc.want)
			}
		})
	}
}
