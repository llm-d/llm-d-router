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
	"maps"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestExtractFromPayloadMap_NoToolFields(t *testing.T) {
	m := map[string]any{"model": "llama3", "messages": []any{}}
	require.Nil(t, ExtractFromPayloadMap(m))
}

func TestExtractFromPayloadMap_ToolsOnly(t *testing.T) {
	m := map[string]any{
		"tools": []any{
			map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":       "get_weather",
					"parameters": map[string]any{"type": "object"},
				},
			},
		},
	}
	s := ExtractFromPayloadMap(m)
	require.NotNil(t, s)
	require.True(t, s.Present)
	require.Equal(t, 1, s.ToolDefinitionsCount)
	require.NotEmpty(t, s.ToolDefinitionsHash)
	require.Equal(t, ChoiceUnset, s.ToolChoiceKind)
	require.False(t, s.ToolChoicePresent)
	require.Equal(t, ParallelUnset, s.ParallelToolCalls)
	require.NotEmpty(t, s.SnapshotHash)
}

func TestExtractFromPayloadMap_ToolChoiceVariants(t *testing.T) {
	tests := []struct {
		name     string
		choice   any
		wantKind string
	}{
		{"auto", "auto", ChoiceAuto},
		{"none", "none", ChoiceNone},
		{"required", "required", ChoiceRequired},
		{"named", map[string]any{"type": "function", "function": map[string]any{"name": "f"}}, ChoiceNamed},
		{"unknown_string", "something_else", ChoiceUnknown},
		{"unknown_type", 42, ChoiceUnknown},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := map[string]any{
				"tools":       []any{map[string]any{"function": map[string]any{"name": "f"}}},
				"tool_choice": tt.choice,
			}
			s := ExtractFromPayloadMap(m)
			require.NotNil(t, s)
			require.Equal(t, tt.wantKind, s.ToolChoiceKind)
			require.True(t, s.ToolChoicePresent)
		})
	}
}

func TestExtractFromPayloadMap_ParallelToolCalls(t *testing.T) {
	tests := []struct {
		name  string
		value any
		want  string
	}{
		{"true", true, ParallelTrue},
		{"false", false, ParallelFalse},
		{"unknown", "yes", ParallelUnknown},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := map[string]any{
				"tools":               []any{map[string]any{"function": map[string]any{"name": "f"}}},
				"parallel_tool_calls": tt.value,
			}
			s := ExtractFromPayloadMap(m)
			require.NotNil(t, s)
			require.Equal(t, tt.want, s.ParallelToolCalls)
		})
	}
}

func TestExtractFromPayloadMap_NilToolChoice(t *testing.T) {
	m := map[string]any{
		"tools":       []any{map[string]any{"function": map[string]any{"name": "f"}}},
		"tool_choice": nil,
	}
	s := ExtractFromPayloadMap(m)
	require.NotNil(t, s)
	require.Equal(t, ChoiceUnset, s.ToolChoiceKind)
	require.False(t, s.ToolChoicePresent)
}

func TestHashStability(t *testing.T) {
	m := map[string]any{
		"tools": []any{
			map[string]any{"function": map[string]any{"name": "a", "parameters": map[string]any{"type": "object"}}},
			map[string]any{"function": map[string]any{"name": "b", "parameters": map[string]any{"type": "object"}}},
		},
		"tool_choice":         "auto",
		"parallel_tool_calls": true,
	}
	s1 := ExtractFromPayloadMap(m)
	s2 := ExtractFromPayloadMap(m)
	require.Equal(t, s1.SnapshotHash, s2.SnapshotHash)
	require.Equal(t, s1.ToolDefinitionsHash, s2.ToolDefinitionsHash)
}

func TestHashChangesOnToolDefinitionChange(t *testing.T) {
	m1 := map[string]any{
		"tools": []any{
			map[string]any{"function": map[string]any{"name": "get_weather"}},
		},
	}
	m2 := map[string]any{
		"tools": []any{
			map[string]any{"function": map[string]any{"name": "get_stock_price"}},
		},
	}
	s1 := ExtractFromPayloadMap(m1)
	s2 := ExtractFromPayloadMap(m2)
	require.NotEqual(t, s1.ToolDefinitionsHash, s2.ToolDefinitionsHash)
	require.NotEqual(t, s1.SnapshotHash, s2.SnapshotHash)
}

func TestHashChangesOnToolChoiceChange(t *testing.T) {
	base := map[string]any{
		"tools": []any{map[string]any{"function": map[string]any{"name": "f"}}},
	}
	m1 := copyMap(base)
	m1["tool_choice"] = "auto"
	m2 := copyMap(base)
	m2["tool_choice"] = ChoiceNone

	s1 := ExtractFromPayloadMap(m1)
	s2 := ExtractFromPayloadMap(m2)
	require.NotEqual(t, s1.SnapshotHash, s2.SnapshotHash)
}

func TestPreservationStatus(t *testing.T) {
	tools := map[string]any{
		"tools":       []any{map[string]any{"function": map[string]any{"name": "f"}}},
		"tool_choice": "auto",
	}

	t.Run("equal", func(t *testing.T) {
		s1 := ExtractFromPayloadMap(tools)
		s2 := ExtractFromPayloadMap(tools)
		require.Equal(t, PreservedTrue, PreservationStatus(s1, s2))
	})

	t.Run("different_choice", func(t *testing.T) {
		m2 := copyMap(tools)
		m2["tool_choice"] = ChoiceNone
		s1 := ExtractFromPayloadMap(tools)
		s2 := ExtractFromPayloadMap(m2)
		require.Equal(t, PreservedFalse, PreservationStatus(s1, s2))
	})

	t.Run("nil_inbound", func(t *testing.T) {
		s2 := ExtractFromPayloadMap(tools)
		require.Equal(t, PreservedUnknown, PreservationStatus(nil, s2))
	})

	t.Run("nil_outbound", func(t *testing.T) {
		s1 := ExtractFromPayloadMap(tools)
		require.Equal(t, PreservedUnknown, PreservationStatus(s1, nil))
	})

	t.Run("both_nil", func(t *testing.T) {
		require.Equal(t, PreservedUnknown, PreservationStatus(nil, nil))
	})
}

func TestHeaderRoundTrip(t *testing.T) {
	m := map[string]any{
		"tools": []any{
			map[string]any{"function": map[string]any{"name": "a", "parameters": map[string]any{"type": "object"}}},
			map[string]any{"function": map[string]any{"name": "b"}},
		},
		"tool_choice":         "required",
		"parallel_tool_calls": false,
	}
	original := ExtractFromPayloadMap(m)
	require.NotNil(t, original)

	headers := ToHeaders(original)
	require.Len(t, headers, 5)

	restored := FromHeaders(headers)
	require.NotNil(t, restored)

	require.Equal(t, original.Present, restored.Present)
	require.Equal(t, original.ToolChoicePresent, restored.ToolChoicePresent)
	require.Equal(t, original.ToolChoiceKind, restored.ToolChoiceKind)
	require.Equal(t, original.ToolDefinitionsCount, restored.ToolDefinitionsCount)
	require.Equal(t, original.ParallelToolCalls, restored.ParallelToolCalls)
	require.Equal(t, original.SnapshotHash, restored.SnapshotHash)
}

func TestFromHeaders_NoHash(t *testing.T) {
	require.Nil(t, FromHeaders(map[string]string{}))
	require.Nil(t, FromHeaders(map[string]string{HeaderToolPresent: "true"}))
}

func TestToHeaders_Nil(t *testing.T) {
	require.Nil(t, ToHeaders(nil))
}

func TestSpanAttributes(t *testing.T) {
	m := map[string]any{
		"tools":       []any{map[string]any{"function": map[string]any{"name": "f"}}},
		"tool_choice": "auto",
	}
	s := ExtractFromPayloadMap(m)
	attrs := SpanAttributes(s, "epp_inbound")
	require.Len(t, attrs, 8)
	require.Nil(t, SpanAttributes(nil, "epp_inbound"))
}

func TestToolDefinitionsHashOrder(t *testing.T) {
	m1 := map[string]any{
		"tools": []any{
			map[string]any{"function": map[string]any{"name": "b"}},
			map[string]any{"function": map[string]any{"name": "a"}},
		},
	}
	m2 := map[string]any{
		"tools": []any{
			map[string]any{"function": map[string]any{"name": "a"}},
			map[string]any{"function": map[string]any{"name": "b"}},
		},
	}
	s1 := ExtractFromPayloadMap(m1)
	s2 := ExtractFromPayloadMap(m2)
	require.Equal(t, s1.ToolDefinitionsHash, s2.ToolDefinitionsHash)
}

func copyMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	maps.Copy(out, m)
	return out
}
