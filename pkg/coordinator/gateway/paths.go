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

package gateway

import "strings"

const (
	PathChatCompletions = "/v1/chat/completions"
	PathCompletions     = "/v1/completions"
	PathResponses       = "/v1/responses"
	DefaultGeneratePath = "/inference/v1/generate"

	EPPProfileHeader  = "EPP-Profile"
	ContentTypeHeader = "Content-Type"
	ContentTypeJSON   = "application/json"

	PhaseEncode  = "encode"
	PhasePrefill = "prefill"
	PhaseDecode  = "decode"
)

type RequestFormat int

const (
	FormatGenerate RequestFormat = iota
	FormatCompletions
	FormatChatCompletions
	FormatResponses
)

func (f RequestFormat) String() string {
	switch f {
	case FormatGenerate:
		return DefaultGeneratePath
	case FormatCompletions:
		return PathCompletions
	case FormatChatCompletions:
		return PathChatCompletions
	case FormatResponses:
		return PathResponses
	default:
		return "unknown"
	}
}

// DetectFormat classifies an inbound request path by substring match. The chi
// router registers PathChatCompletions, PathCompletions, PathResponses, and
// DefaultGeneratePath; DetectFormat maps the first three by name and
// everything else, including DefaultGeneratePath itself, to FormatGenerate.
// There is no error return because an unrecognized path is not a failure: it
// maps to the generate format by design.
//
// A step that decides which body field to read based on wire format (chat
// completions' "messages" versus Responses' "input") must gate on this result
// rather than on which field happens to be present in the body: an unrelated
// route's request could carry a same-shaped stray field, and key presence
// alone cannot tell that apart from the field this request actually means.
func DetectFormat(path string) RequestFormat {
	if strings.Contains(path, PathChatCompletions) {
		return FormatChatCompletions
	}
	if strings.Contains(path, PathCompletions) {
		return FormatCompletions
	}
	if strings.Contains(path, PathResponses) {
		return FormatResponses
	}
	return FormatGenerate
}

func PathForFormat(format RequestFormat) string {
	switch format {
	case FormatChatCompletions:
		return PathChatCompletions
	case FormatCompletions:
		return PathCompletions
	case FormatResponses:
		return PathResponses
	default:
		return DefaultGeneratePath
	}
}
