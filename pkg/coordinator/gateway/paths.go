package gateway

import "strings"

const (
	PathChatCompletions = "/v1/chat/completions"
	PathCompletions     = "/v1/completions"
	DefaultGeneratePath = "/inference/v1/generate"

	EPPPhaseHeader = "EPP-Phase"

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
