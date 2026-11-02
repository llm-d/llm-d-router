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

// Modality identifies the kind of multimodal content a prompt carries. Values
// match the model server's multimodal hash keys, so one label names a modality
// in a body's mm_hashes / mm_placeholders / kwargs_data maps, on a
// pipeline.MultimodalEntry, in the EPP's features, and in a metric label. It
// lives here because the coordinator, the sidecar and the EPP framework all key
// on it; the framework contract aliases these names
// (pkg/epp/framework/interface/requesthandling).
//
// This is a vocabulary, not a validated enum. The constants name the modalities
// this repo's pipeline produces, but a token-in request may carry a feature map
// keyed by anything its model server understands (see
// steps.modalitiesInFeatures), so a reader cannot assume one of these three.
type Modality string

const (
	ModalityImage Modality = "image"
	ModalityAudio Modality = "audio"
	ModalityVideo Modality = "video"
)

// PartModality reports the Modality a content part of type partType names, with
// ok false for a part type that names no media. This is the only place that
// decides which types do, and the answer does not depend on which API carries
// the part: vLLM's chat parser primes input_image and image_url through the same
// content part map, so an image is named either way.
//
// Callers may handle ok == false however they like -- the coordinator's walkers
// pass over the part, the sidecar's fan-out logs a drop -- but may not disagree
// about which parts exist, which is why they share this.
func PartModality(partType string) (Modality, bool) {
	switch partType {
	case PartTypeImageURL, PartTypeInputImage:
		return ModalityImage, true
	case PartTypeAudioURL, PartTypeInputAudio:
		return ModalityAudio, true
	case PartTypeVideoURL:
		return ModalityVideo, true
	}
	return "", false
}

// MessagesBlockModality reports the Modality an Anthropic Messages content block
// of type blockType names, with ok false for a block that names no media. It is
// PartModality's counterpart for a Messages request, whose blocks are not chat
// parts: vLLM's Anthropic conversion loads image blocks only.
func MessagesBlockModality(blockType string) (Modality, bool) {
	if blockType == BlockTypeImage {
		return ModalityImage, true
	}
	return "", false
}
