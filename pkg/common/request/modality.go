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
// match the model server's multimodal hash keys, so the same label names a
// modality in a request body's mm_hashes, mm_placeholders and kwargs_data maps,
// on a pipeline.MultimodalEntry, in the EPP's multimodal features, and in a
// metric label.
//
// The vocabulary lives here because the coordinator, the sidecar and the EPP
// framework all key on it and all already import this package; the framework
// contract aliases these names rather than declaring its own
// (pkg/epp/framework/interface/requesthandling).
//
// The three constants name the modalities this repo's pipeline produces, not
// the only values the type holds: a token-in request may carry a feature map
// keyed by anything its model server understands, and the coordinator turns
// those keys into entries as it finds them (see steps.modalitiesInFeatures).
// So the type is a vocabulary, not a validated enum, and a reader of an entry
// cannot assume one of these three.
type Modality string

const (
	ModalityImage Modality = "image"
	ModalityAudio Modality = "audio"
	ModalityVideo Modality = "video"
)

// PartModality reports the Modality a content part of type partType names on a
// request to apiType. ok is false for a part type that names no media on that
// API, and this is the only place that decides which do.
//
// A chat-completions request may name an image either way: vLLM's chat parser
// primes input_image and image_url through the same content part map, so an
// input_image on a chat request reaches the model and has to be collected. It
// alone also carries the audio and video part types, inline or by URL.
//
// A Responses request names an image input_image only, and names no audio or
// video at all: its input content union is input_text / input_image /
// input_file, so a Responses request carrying any of the others fails the model
// server's input validation before a worker sees it. An entry built for such a
// part would carry placeholder tokens no worker ever produces, and an encoder
// primed with one would fail the fanout first.
//
// What a caller does with ok == false is its own business, and the callers
// differ: the coordinator's walkers pass over the part, while the sidecar's
// encoder fan-out counts a part type that names media on some other API as a
// skipped part so the drop is visible in a log line. What they may not do is
// disagree about which parts exist at all, which is why they share this.
func PartModality(partType string, apiType APIType) (Modality, bool) {
	if partType == PartTypeInputImage {
		return ModalityImage, true
	}
	if apiType == APITypeResponses {
		return "", false
	}
	switch partType {
	case PartTypeImageURL:
		return ModalityImage, true
	case PartTypeAudioURL, PartTypeInputAudio:
		return ModalityAudio, true
	case PartTypeVideoURL:
		return ModalityVideo, true
	}
	return "", false
}
