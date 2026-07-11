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
	DefaultGeneratePath = "/inference/v1/generate"

	EPPPhaseHeader    = "EPP-Phase"
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
)

func (f RequestFormat) String() string {
	switch f {
	case FormatGenerate:
		return DefaultGeneratePath
	case FormatCompletions:
		return PathCompletions
	case FormatChatCompletions:
		return PathChatCompletions
	default:
		return "unknown"
	}
}

// DetectFormat classifies an inbound request path. The chi router registers
// only PathChatCompletions and PathCompletions, so in production path is always
// one of those two; the FormatGenerate fallback covers only callers that pass an
// arbitrary path. There is no error return because an unrecognized path is not a
// failure: it maps to the generate format by design.
func DetectFormat(path string) RequestFormat {
	if strings.Contains(path, PathChatCompletions) {
		return FormatChatCompletions
	}
	if strings.Contains(path, PathCompletions) {
		return FormatCompletions
	}
	return FormatGenerate
}

func PathForFormat(format RequestFormat) string {
	switch format {
	case FormatChatCompletions:
		return PathChatCompletions
	case FormatCompletions:
		return PathCompletions
	default:
		return DefaultGeneratePath
	}
}
