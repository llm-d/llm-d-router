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
