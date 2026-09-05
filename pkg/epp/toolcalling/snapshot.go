/*
Copyright 2025 The Kubernetes Authors.

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

package toolcalling

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strconv"

	"go.opentelemetry.io/otel/attribute"
)

const (
	HeaderSnapshotHash      = "x-llm-d-tool-snapshot-hash"
	HeaderToolPresent       = "x-llm-d-tool-present"
	HeaderToolChoiceKind    = "x-llm-d-tool-choice-kind"
	HeaderToolDefsCount     = "x-llm-d-tool-definitions-count"
	HeaderParallelToolCalls = "x-llm-d-tool-parallel-calls"

	ChoiceUnset    = "unset"
	ChoiceNone     = "none"
	ChoiceAuto     = "auto"
	ChoiceRequired = "required"
	ChoiceNamed    = "named"
	ChoiceUnknown  = "unknown"

	ParallelUnset   = "unset"
	ParallelTrue    = "true"
	ParallelFalse   = "false"
	ParallelUnknown = "unknown"

	PreservedTrue    = "true"
	PreservedFalse   = "false"
	PreservedUnknown = "unknown"

	hashTruncLen = 16
)

// ToolCallingSnapshot captures the normalized shape of tool-calling
// parameters in an inference request, without exposing high-cardinality
// content such as tool names, schemas, or function descriptions.
type ToolCallingSnapshot struct {
	Present              bool
	ToolChoicePresent    bool
	ToolChoiceKind       string
	ToolDefinitionsCount int
	ToolDefinitionsHash  string
	ParallelToolCalls    string
	SnapshotHash         string
}

// ExtractFromPayloadMap reads tool-calling fields from a parsed request
// body map and returns a normalized snapshot. Returns nil when none of
// the tool-calling keys are present.
func ExtractFromPayloadMap(m map[string]any) *ToolCallingSnapshot {
	_, hasTools := m["tools"]
	_, hasToolChoice := m["tool_choice"]
	_, hasParallel := m["parallel_tool_calls"]

	if !hasTools && !hasToolChoice && !hasParallel {
		return nil
	}

	s := &ToolCallingSnapshot{}

	if hasTools {
		if tools, ok := m["tools"].([]any); ok && len(tools) > 0 {
			s.Present = true
			s.ToolDefinitionsCount = len(tools)
			s.ToolDefinitionsHash = computeToolDefinitionsHash(tools)
		}
	}

	s.ToolChoiceKind = normalizeToolChoice(m)
	s.ToolChoicePresent = s.ToolChoiceKind != ChoiceUnset

	s.ParallelToolCalls = normalizeParallelToolCalls(m)

	s.SnapshotHash = computeSnapshotHash(s)

	return s
}

func normalizeToolChoice(m map[string]any) string {
	v, ok := m["tool_choice"]
	if !ok || v == nil {
		return ChoiceUnset
	}

	switch tc := v.(type) {
	case string:
		switch tc {
		case "none":
			return ChoiceNone
		case "auto":
			return ChoiceAuto
		case "required":
			return ChoiceRequired
		default:
			return ChoiceUnknown
		}
	case map[string]any:
		return ChoiceNamed
	default:
		return ChoiceUnknown
	}
}

func normalizeParallelToolCalls(m map[string]any) string {
	v, ok := m["parallel_tool_calls"]
	if !ok {
		return ParallelUnset
	}

	b, ok := v.(bool)
	if !ok {
		return ParallelUnknown
	}
	if b {
		return ParallelTrue
	}
	return ParallelFalse
}

func computeSnapshotHash(s *ToolCallingSnapshot) string {
	raw := "present=" + strconv.FormatBool(s.Present) +
		"|choice_present=" + strconv.FormatBool(s.ToolChoicePresent) +
		"|choice_kind=" + s.ToolChoiceKind +
		"|defs_count=" + strconv.Itoa(s.ToolDefinitionsCount) +
		"|defs_hash=" + s.ToolDefinitionsHash +
		"|parallel=" + s.ParallelToolCalls

	h := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(h[:])[:hashTruncLen]
}

func computeToolDefinitionsHash(tools []any) string {
	type canonicalTool struct {
		Name       string `json:"name"`
		Parameters any    `json:"parameters,omitempty"`
	}

	var extracted []canonicalTool
	for _, t := range tools {
		tm, ok := t.(map[string]any)
		if !ok {
			continue
		}
		fn, ok := tm["function"].(map[string]any)
		if !ok {
			continue
		}
		ct := canonicalTool{}
		if name, ok := fn["name"].(string); ok {
			ct.Name = name
		}
		ct.Parameters = fn["parameters"]
		extracted = append(extracted, ct)
	}

	if len(extracted) == 0 {
		return ""
	}

	sort.Slice(extracted, func(i, j int) bool {
		return extracted[i].Name < extracted[j].Name
	})

	b, err := json.Marshal(extracted)
	if err != nil {
		return ""
	}

	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])[:hashTruncLen]
}

// PreservationStatus compares inbound and outbound snapshots.
func PreservationStatus(inbound, outbound *ToolCallingSnapshot) string {
	if inbound == nil || outbound == nil {
		return PreservedUnknown
	}
	if inbound.SnapshotHash == outbound.SnapshotHash {
		return PreservedTrue
	}
	return PreservedFalse
}

// ToHeaders converts the snapshot to propagation headers.
func ToHeaders(s *ToolCallingSnapshot) map[string]string {
	if s == nil {
		return nil
	}
	return map[string]string{
		HeaderSnapshotHash:      s.SnapshotHash,
		HeaderToolPresent:       strconv.FormatBool(s.Present),
		HeaderToolChoiceKind:    s.ToolChoiceKind,
		HeaderToolDefsCount:     strconv.Itoa(s.ToolDefinitionsCount),
		HeaderParallelToolCalls: s.ParallelToolCalls,
	}
}

// FromHeaders reconstructs a snapshot from propagation headers.
// Returns nil if the snapshot hash header is absent.
func FromHeaders(headers map[string]string) *ToolCallingSnapshot {
	hash, ok := headers[HeaderSnapshotHash]
	if !ok || hash == "" {
		return nil
	}

	s := &ToolCallingSnapshot{
		SnapshotHash:      hash,
		ToolChoiceKind:    headers[HeaderToolChoiceKind],
		ParallelToolCalls: headers[HeaderParallelToolCalls],
	}

	if v, err := strconv.ParseBool(headers[HeaderToolPresent]); err == nil {
		s.Present = v
	}
	if v, err := strconv.Atoi(headers[HeaderToolDefsCount]); err == nil {
		s.ToolDefinitionsCount = v
	}
	s.ToolChoicePresent = s.ToolChoiceKind != ChoiceUnset

	return s
}

// SpanAttributes returns OTel span event attributes for the snapshot.
func SpanAttributes(s *ToolCallingSnapshot, boundary string) []attribute.KeyValue {
	if s == nil {
		return nil
	}
	return []attribute.KeyValue{
		attribute.String("boundary", boundary),
		attribute.Bool("tool_calling.present", s.Present),
		attribute.Bool("tool_calling.tool_choice.present", s.ToolChoicePresent),
		attribute.String("tool_calling.tool_choice.kind", s.ToolChoiceKind),
		attribute.Int("tool_calling.tool_definitions.count", s.ToolDefinitionsCount),
		attribute.String("tool_calling.tool_definitions.hash", s.ToolDefinitionsHash),
		attribute.String("tool_calling.parallel_tool_calls", s.ParallelToolCalls),
		attribute.String("tool_calling.snapshot_hash", s.SnapshotHash),
	}
}
