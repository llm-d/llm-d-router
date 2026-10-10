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

package toolcalling

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestCompareRequestFields(t *testing.T) {
	tests := []struct {
		name       string
		api        reqcommon.APIType
		before     map[string]any
		after      map[string]any
		field      Field
		wantStatus FieldStatusValue
	}{
		{
			name: "reordered object keys are preserved",
			api:  reqcommon.APITypeChatCompletions,
			before: map[string]any{
				"tools": []any{map[string]any{
					"type": "function",
					"function": map[string]any{
						"name":       "private_tool_name",
						"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
					},
				}},
			},
			after: map[string]any{
				"tools": []any{map[string]any{
					"function": map[string]any{
						"parameters": map[string]any{"properties": map[string]any{"city": map[string]any{"type": "string"}}, "type": "object"},
						"name":       "private_tool_name",
					},
					"type": "function",
				}},
			},
			field:      FieldTools,
			wantStatus: FieldStatusPreserved,
		},
		{
			name:       "tool choice value changed",
			api:        reqcommon.APITypeChatCompletions,
			before:     map[string]any{"tool_choice": "required"},
			after:      map[string]any{"tool_choice": "auto"},
			field:      FieldToolChoice,
			wantStatus: FieldStatusChanged,
		},
		{
			name:       "field removed is dropped",
			api:        reqcommon.APITypeChatCompletions,
			before:     map[string]any{"parallel_tool_calls": true},
			after:      map[string]any{},
			field:      FieldParallelToolCalls,
			wantStatus: FieldStatusDropped,
		},
		{
			name:       "absent to null is changed",
			api:        reqcommon.APITypeMessages,
			before:     map[string]any{},
			after:      map[string]any{"tool_choice": nil},
			field:      FieldToolChoice,
			wantStatus: FieldStatusChanged,
		},
		{
			name:       "explicit null preserved",
			api:        reqcommon.APITypeMessages,
			before:     map[string]any{"tool_choice": nil},
			after:      map[string]any{"tool_choice": nil},
			field:      FieldToolChoice,
			wantStatus: FieldStatusPreserved,
		},
		{
			name:       "response format compared for chat completions",
			api:        reqcommon.APITypeChatCompletions,
			before:     map[string]any{"response_format": map[string]any{"type": "json_object"}},
			after:      map[string]any{"response_format": map[string]any{"type": "text"}},
			field:      FieldResponseFormat,
			wantStatus: FieldStatusChanged,
		},
		{
			name:       "array order is significant",
			api:        reqcommon.APITypeChatCompletions,
			before:     map[string]any{"tools": []any{map[string]any{"name": "a"}, map[string]any{"name": "b"}}},
			after:      map[string]any{"tools": []any{map[string]any{"name": "b"}, map[string]any{"name": "a"}}},
			field:      FieldTools,
			wantStatus: FieldStatusChanged,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before, err := captureRequest(tt.api, tt.before)
			require.NoError(t, err)
			after, err := captureRequest(tt.api, tt.after)
			require.NoError(t, err)

			results, err := CompareRequests(before, after)
			require.NoError(t, err)
			require.Equal(t, tt.wantStatus, resultStatus(t, results, tt.field))
		})
	}
}

func TestCompareRequests_ExplicitNullDiffersFromAbsent(t *testing.T) {
	absent, err := captureRequest(reqcommon.APITypeChatCompletions, map[string]any{})
	require.NoError(t, err)

	withNull, err := captureRequest(reqcommon.APITypeChatCompletions, map[string]any{"response_format": nil})
	require.NoError(t, err)

	results, err := CompareRequests(absent, withNull)
	require.NoError(t, err)
	require.Equal(t, FieldStatusChanged, resultStatus(t, results, FieldResponseFormat))
}

func TestCompareRequestJSONNumbers(t *testing.T) {
	tests := []struct {
		name   string
		before string
		after  string
		want   FieldStatusValue
	}{
		{name: "large integer changed", before: "9007199254740992", after: "9007199254740993", want: FieldStatusChanged},
		{name: "large negative integer changed", before: "-9007199254740992", after: "-9007199254740993", want: FieldStatusChanged},
		{name: "precise decimal changed", before: "0.100000000000000005", after: "0.100000000000000006", want: FieldStatusChanged},
		{name: "integer and decimal equivalent", before: "1", after: "1.0", want: FieldStatusPreserved},
		{name: "integer and exponent equivalent", before: "1000", after: "1e3", want: FieldStatusPreserved},
		{name: "fraction and exponent equivalent", before: "0.0010", after: "1E-3", want: FieldStatusPreserved},
		{name: "negative numbers equivalent", before: "-12.50", after: "-125e-1", want: FieldStatusPreserved},
		{name: "negative zero equivalent", before: "-0.0", after: "0e1000", want: FieldStatusPreserved},
		{name: "beyond float range equivalent", before: "1e400", after: "10e399", want: FieldStatusPreserved},
		{name: "large exponent equivalent", before: "1e1000000000", after: "10e999999999", want: FieldStatusPreserved},
		{name: "large exponent changed", before: "1e1000000000", after: "1e1000000001", want: FieldStatusChanged},
		{name: "exponent beyond int64 equivalent", before: "1e9223372036854775808", after: "10e9223372036854775807", want: FieldStatusPreserved},
		{name: "number and string differ", before: "1", after: `"1"`, want: FieldStatusChanged},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := func(value string) []byte {
				return []byte(`{"tools":[{"type":"function","function":{"name":"example","parameters":{"enum":[` + value + `]}}}]}`)
			}
			before, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, body(tt.before))
			require.NoError(t, err)
			after, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, body(tt.after))
			require.NoError(t, err)
			statuses, err := CompareRequests(before, after)
			require.NoError(t, err)
			require.Equal(t, tt.want, resultStatus(t, statuses, FieldTools))
		})
	}
}

func TestCompareRequestNumbersAcrossPayloadTypes(t *testing.T) {
	before, err := captureRequest(reqcommon.APITypeMessages, map[string]any{
		"tools": []any{map[string]any{"input_schema": map[string]any{"maximum": int64(9007199254740993)}}},
	})
	require.NoError(t, err)
	after, err := CaptureRequestJSON(reqcommon.APITypeMessages, []byte(`{"tools":[{"input_schema":{"maximum":9007199254740993.0}}]}`))
	require.NoError(t, err)
	statuses, err := CompareRequests(before, after)
	require.NoError(t, err)
	require.Equal(t, FieldStatusPreserved, resultStatus(t, statuses, FieldTools))
}

func TestCompareRequests_AbsentFieldsAreNotObserved(t *testing.T) {
	before, err := captureRequest(reqcommon.APITypeChatCompletions, map[string]any{})
	require.NoError(t, err)
	after, err := captureRequest(reqcommon.APITypeChatCompletions, map[string]any{})
	require.NoError(t, err)

	results, err := CompareRequests(before, after)
	require.NoError(t, err)
	require.False(t, resultFor(t, results, FieldToolChoice).Observed)
	require.Equal(t, FieldStatusPreserved, resultFor(t, results, FieldToolChoice).Status)
}

func TestRejectedStatusesOnlyIncludeExplicitPresentFields(t *testing.T) {
	snapshot, err := captureRequest(reqcommon.APITypeChatCompletions, map[string]any{
		"tools":       []any{},
		"tool_choice": "required",
	})
	require.NoError(t, err)

	for _, tt := range []struct {
		name   string
		fields []Field
		want   []Field
	}{
		{name: "no explicit rejection"},
		{name: "only tools", fields: []Field{FieldTools}, want: []Field{FieldTools}},
		{name: "only choice", fields: []Field{FieldToolChoice}, want: []Field{FieldToolChoice}},
		{name: "duplicates count once", fields: []Field{FieldTools, FieldTools}, want: []Field{FieldTools}},
		{name: "absent field", fields: []Field{FieldParallelToolCalls}},
		{name: "unsupported field", fields: []Field{"private_unknown_field"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			results := RejectedFieldStatuses(snapshot, tt.fields...)
			require.Len(t, results, len(tt.want))
			for i, field := range tt.want {
				require.Equal(t, FieldStatus{Field: field, Status: FieldStatusRejected, Observed: true}, results[i])
			}
		})
	}

	messages, err := captureRequest(reqcommon.APITypeMessages, map[string]any{"parallel_tool_calls": true})
	require.NoError(t, err)
	require.Empty(t, RejectedFieldStatuses(messages, FieldParallelToolCalls))
}

func TestCaptureRequestJSON_MalformedBody(t *testing.T) {
	_, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, []byte(`{"tools":[`))
	require.Error(t, err)
}

func TestCaptureRequestJSONRejectsTrailingData(t *testing.T) {
	for _, body := range []string{`{"tools":[]} {}`, `{"tools":[]} 1`, `{"tools":[]} invalid`, "\f{}", "{}\u0085", "\u00a0{}"} {
		_, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, []byte(body))
		require.Error(t, err)
	}
}

func TestCaptureRequestJSONFieldLookup(t *testing.T) {
	for _, tt := range []struct {
		name    string
		body    string
		present bool
	}{
		{name: "absent", body: `{"messages":[{"content":"hello"}]}`},
		{name: "nested field", body: `{"messages":[{"tools":[]}]}`},
		{name: "field name in content", body: `{"messages":[{"content":"tools tool_choice"}]}`},
		{name: "case sensitive", body: `{"TOOLS":[]}`},
		{name: "escaped field name", body: `{"to\u006fls":[]}`, present: true},
		{name: "literal Unicode escape in key", body: `{"to\\u006fls":[]}`},
		{name: "escaped duplicate last wins", body: `{"tool_choice":"required","tool\u005fchoice":"auto"}`, present: true},
		{name: "explicit null", body: `{"tools":null}`, present: true},
		{name: "structured output only", body: `{"response_format":{"type":"json_object"}}`},
		{name: "null structured output", body: `{"response_format":null}`},
		{name: "parallel calls only", body: `{"parallel_tool_calls":false}`, present: true},
		{name: "duplicate field", body: `{"tool_choice":"required","tool_choice":"auto"}`, present: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			snapshot, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, []byte(tt.body))
			require.NoError(t, err)
			require.Equal(t, tt.present, snapshot.Summary().ToolCallingPresent)
			if tt.name == "duplicate field" || tt.name == "escaped duplicate last wins" {
				require.Equal(t, ToolChoiceAuto, snapshot.Summary().ToolChoiceKind)
			}
		})
	}
	for _, body := range []string{`null`, `[]`, `42`, `{"messages":[}`, `{"model":"m"} {}`} {
		_, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, []byte(body))
		require.Error(t, err)
	}
}

func TestCaptureRequestJSONNonToolAllocations(t *testing.T) {
	for _, content := range []string{"hello", "tools tool_choice parallel_tool_calls response_format", `\u0061`, `quote: \"tools\" and backslash: \\`} {
		for _, size := range []int{0, 64 * 1024, 1024 * 1024} {
			t.Run(fmt.Sprintf("%s/%d", content, size), func(t *testing.T) {
				body := []byte(`{"metadata":{"tools":[]},"messages":[{"content":"` + content + strings.Repeat("x", size) + `"}]}`)
				allocations := testing.AllocsPerRun(20, func() {
					snapshot, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, body)
					if err != nil || snapshot.Summary().ToolCallingPresent {
						t.Fatal("expected a valid non-tool snapshot", err)
					}
				})
				require.LessOrEqual(t, allocations, float64(2), "non-tool values should not be copied or decoded")
			})
		}
	}
}

func FuzzCaptureRequestJSON(f *testing.F) {
	for _, body := range []string{
		`{}`, `null`, `[]`, `{"tools":[`, `{"tools":[]} {}`, "\f{}", "{}\u0085", "\u00a0{}",
		`{"messages":[{"content":"tools \u0061 \"tools\" \\"}]}`,
		`{"to\u006fls":[{"parameters":{"b":2,"a":1e999}}],"tool_choice":"auto"}`,
		`{"tools":[],"to\u006fls":null,"tool_choice":"none","tool_choice":"auto"}`,
		`{"TOOLS":[],"ignored":[{},[1,null,true,"}"]],"response_format":null}`,
		`{"tools":[{"name":"` + string([]byte{0xff}) + `"}]}`,
		`{"a":-1.2e+30,"b":true,"c":false,"d":null,"tools":[]}`,
		"{ \n\t\"ignored\" : [ {\"x\": [false, null, {\"text\": \"[{}]\\\"\"}]} ], \r\n\"tool_choice\" : \"auto\" }",
	} {
		f.Add([]byte(body))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		trimmed := bytes.TrimSpace(body)
		validObject := len(trimmed) > 0 && trimmed[0] == '{' && json.Valid(body)
		for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages, reqcommon.APITypeResponses} {
			got, err := CaptureRequestJSON(api, body)
			if !validObject {
				require.Error(t, err)
				continue
			}
			require.NoError(t, err)
			decoder := json.NewDecoder(bytes.NewReader(body))
			decoder.UseNumber()
			var decoded map[string]any
			require.NoError(t, decoder.Decode(&decoded))
			want, err := captureRequest(api, decoded)
			require.NoError(t, err)
			require.Equal(t, want.Summary(), got.Summary())
			statuses, err := CompareRequests(want, got)
			require.NoError(t, err)
			for _, status := range statuses {
				require.Equal(t, FieldStatusPreserved, status.Status)
				require.Equal(t, want.fields[status.Field].present, status.Observed)
			}
		}
	})
}

func BenchmarkCaptureRequestJSONNonTool(b *testing.B) {
	for _, size := range []int{1024, 64 * 1024, 1024 * 1024} {
		body := []byte(`{"messages":[{"content":"` + strings.Repeat("x", size) + `"}]}`)
		b.Run(fmt.Sprintf("%d_bytes", size), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, body); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkCaptureRequestJSON(b *testing.B) {
	for _, size := range []int{1024, 64 * 1024, 1024 * 1024} {
		prompt := strings.Repeat("x", size)
		for _, tt := range []struct {
			name string
			body string
		}{
			{name: "non_tool", body: `{"messages":[{"content":"` + prompt + `"}]}`},
			{name: "prompt_field_words", body: `{"messages":[{"content":"tools tool_choice parallel_tool_calls response_format ` + prompt + `"}]}`},
			{name: "prompt_unicode_escape", body: `{"messages":[{"content":"\u0061` + prompt + `"}]}`},
			{name: "nested_fields", body: `{"messages":[{"tools":[],"content":"` + prompt + `"}]}`},
			{name: "tool", body: `{"messages":[{"content":"` + prompt + `"}],"tools":[{"type":"function","function":{"name":"private","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"tool_choice":"auto"}`},
			{name: "escaped_tool_key", body: `{"messages":[{"content":"` + prompt + `"}],"to\u006fls":[{"name":"private","parameters":{"type":"object"}}]}`},
		} {
			b.Run(fmt.Sprintf("%s/%d", tt.name, size), func(b *testing.B) {
				body := []byte(tt.body)
				b.ReportAllocs()
				for b.Loop() {
					if _, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, body); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func TestCaptureRequestJSONRetainsRawFields(t *testing.T) {
	body := []byte(`{"tools":[{"function":{"parameters":{"properties":{"city":{"type":"string"}},"type":"object"},"name":"private"},"type":"function"}],"tool_choice":"auto","response_format":{"type":"json_object"}}`)
	snapshot, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, body)
	require.NoError(t, err)
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &raw))
	for _, field := range []Field{FieldTools, FieldToolChoice, FieldResponseFormat} {
		require.Equal(t, []byte(raw[string(field)]), snapshot.fields[field].value, "capture should retain raw JSON without re-encoding")
	}
	copy(body, bytes.Repeat([]byte{'x'}, len(body)))
	require.Equal(t, []byte(raw["tools"]), snapshot.fields[FieldTools].value, "captured fields must own their bytes")
}

func TestCompareRequestJSONRepresentations(t *testing.T) {
	for _, tt := range []struct {
		name   string
		before string
		after  string
		want   FieldStatusValue
	}{
		{name: "reordered nested keys", before: `[{"name":"n","parameters":{"b":2,"a":1}}]`, after: `[{"parameters":{"a":1,"b":2},"name":"n"}]`, want: FieldStatusPreserved},
		{name: "whitespace and escaped values", before: `[{"name":"\u006e"}]`, after: `[ { "name" : "n" } ]`, want: FieldStatusPreserved},
		{name: "duplicate nested key last wins", before: `[{"parameters":{"a":0,"a":1}}]`, after: `[{"parameters":{"a":1}}]`, want: FieldStatusPreserved},
		{name: "number differs from string", before: `[{"value":1}]`, after: `[{"value":"1"}]`, want: FieldStatusChanged},
		{name: "missing nested key differs from null", before: `[{}]`, after: `[{"value":null}]`, want: FieldStatusChanged},
		{name: "array order differs", before: `[1,2]`, after: `[2,1]`, want: FieldStatusChanged},
		{name: "container types differ", before: `[]`, after: `{}`, want: FieldStatusChanged},
	} {
		t.Run(tt.name, func(t *testing.T) {
			before, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, []byte(`{"tools":`+tt.before+`}`))
			require.NoError(t, err)
			after, err := CaptureRequestJSON(reqcommon.APITypeChatCompletions, []byte(`{"tools":`+tt.after+`}`))
			require.NoError(t, err)
			statuses, err := CompareRequests(before, after)
			require.NoError(t, err)
			require.Equal(t, tt.want, resultStatus(t, statuses, FieldTools))
		})
	}
}

func TestCaptureRequestJSONMatchesDecodedCapture(t *testing.T) {
	for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages, reqcommon.APITypeResponses} {
		for _, body := range []string{
			`{"tools":null,"tool_choice":null,"response_format":null}`,
			`{"tools":[],"tool_choice":"required","parallel_tool_calls":false}`,
			`{"tools":[null,true,1e999,"x",{}],"tool_choice":{"type":"function","name":"private","function":{"name":"private"}}}`,
			`{"tools":42,"tool_choice":1e999}`,
			`{"tools":{},"tool_choice":{"type":"tool","name":"private","extra":1e999}}`,
			`{"tools":[{},{}],"tools":[{}],"tool_choice":"required","tool_choice":"auto"}`,
			`{"tools":[{},{},{},{},{},{},{},{},{},{},{}],"response_format":{"b":2,"a":1}}`,
		} {
			t.Run(api.String()+"/"+body, func(t *testing.T) {
				decoder := json.NewDecoder(strings.NewReader(body))
				decoder.UseNumber()
				var decoded map[string]any
				require.NoError(t, decoder.Decode(&decoded))
				want, err := captureRequest(api, decoded)
				require.NoError(t, err)
				got, err := CaptureRequestJSON(api, []byte(body))
				require.NoError(t, err)
				require.Equal(t, want.Summary(), got.Summary())
				statuses, err := CompareRequests(want, got)
				require.NoError(t, err)
				for _, status := range statuses {
					require.Equal(t, FieldStatusPreserved, status.Status)
				}
			})
		}
	}
}

func TestCompareRequests_MessagesOnlyIncludesSupportedFields(t *testing.T) {
	before, err := captureRequest(reqcommon.APITypeMessages, map[string]any{
		"tools":           []any{},
		"tool_choice":     map[string]any{"type": "auto"},
		"response_format": map[string]any{"type": "json_object"},
	})
	require.NoError(t, err)
	after, err := captureRequest(reqcommon.APITypeMessages, map[string]any{
		"tools":           []any{},
		"tool_choice":     map[string]any{"type": "auto"},
		"response_format": map[string]any{"type": "text"},
	})
	require.NoError(t, err)

	results, err := CompareRequests(before, after)
	require.NoError(t, err)
	require.Len(t, results, 2)
	for _, result := range results {
		require.NotEqual(t, FieldResponseFormat, result.Field)
	}
}

func TestCompareRequests_RejectsDifferentSurfaces(t *testing.T) {
	chat, err := captureRequest(reqcommon.APITypeChatCompletions, map[string]any{})
	require.NoError(t, err)
	messages, err := captureRequest(reqcommon.APITypeMessages, map[string]any{})
	require.NoError(t, err)

	_, err = CompareRequests(chat, messages)
	require.Error(t, err)
}

func TestCaptureRequest_CopiesFieldValues(t *testing.T) {
	body := map[string]any{"tool_choice": "auto"}
	before, err := captureRequest(reqcommon.APITypeChatCompletions, body)
	require.NoError(t, err)
	body["tool_choice"] = "required"
	after, err := captureRequest(reqcommon.APITypeChatCompletions, body)
	require.NoError(t, err)

	results, err := CompareRequests(before, after)
	require.NoError(t, err)
	require.Equal(t, FieldStatusChanged, resultStatus(t, results, FieldToolChoice))
}

func TestRequestSummaryUsesBoundedValuesOnly(t *testing.T) {
	snapshot, err := captureRequest(reqcommon.APITypeChatCompletions, map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{"name": "sentinel_private_name", "parameters": map[string]any{"description": "sentinel_private_schema"}}},
		},
		"tool_choice": map[string]any{"type": "function", "function": map[string]any{"name": "sentinel_private_name"}},
	})
	require.NoError(t, err)

	summary := snapshot.Summary()
	require.True(t, summary.ToolCallingPresent)
	require.True(t, summary.ToolChoicePresent)
	require.Equal(t, ToolChoiceNamed, summary.ToolChoiceKind)
	require.Equal(t, ToolCountBucketOne, summary.ToolCountBucket)

	encoded, err := json.Marshal(summary)
	require.NoError(t, err)
	require.NotContains(t, string(encoded), "sentinel_private")
}

func TestRequestSummaryPresenceIncludesOnlySupportedToolFields(t *testing.T) {
	for _, tt := range []struct {
		name        string
		surface     reqcommon.APIType
		body        map[string]any
		wantPresent bool
		wantChoice  bool
	}{
		{name: "absent fields", surface: reqcommon.APITypeChatCompletions},
		{name: "structured output only", surface: reqcommon.APITypeChatCompletions, body: map[string]any{"response_format": map[string]any{"type": "json_object"}}},
		{name: "null structured output", surface: reqcommon.APITypeChatCompletions, body: map[string]any{"response_format": nil}},
		{name: "empty tools", surface: reqcommon.APITypeChatCompletions, body: map[string]any{"tools": []any{}}, wantPresent: true},
		{name: "null tools", surface: reqcommon.APITypeChatCompletions, body: map[string]any{"tools": nil}, wantPresent: true},
		{name: "tool choice only", surface: reqcommon.APITypeChatCompletions, body: map[string]any{"tool_choice": "required"}, wantPresent: true, wantChoice: true},
		{name: "parallel calls only", surface: reqcommon.APITypeChatCompletions, body: map[string]any{"parallel_tool_calls": false}, wantPresent: true},
		{name: "Messages null choice", surface: reqcommon.APITypeMessages, body: map[string]any{"tool_choice": nil}, wantPresent: true, wantChoice: true},
		{name: "Messages unsupported fields", surface: reqcommon.APITypeMessages, body: map[string]any{"response_format": nil, "parallel_tool_calls": true}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			snapshot, err := captureRequest(tt.surface, tt.body)
			require.NoError(t, err)
			summary := snapshot.Summary()
			require.Equal(t, tt.wantPresent, summary.ToolCallingPresent)
			require.Equal(t, tt.wantChoice, summary.ToolChoicePresent)
			require.Empty(t, summary.ToolCountBucket)
		})
	}
}

func TestSpanAttributesContainOnlyBoundedSummaryAndFieldStatuses(t *testing.T) {
	snapshot, err := captureRequest(reqcommon.APITypeChatCompletions, map[string]any{
		"tools":       []any{map[string]any{"function": map[string]any{"name": "sentinel_private_name"}}},
		"tool_choice": "required",
	})
	require.NoError(t, err)
	results := []FieldStatus{
		{Field: FieldTools, Status: FieldStatusPreserved, Observed: true},
		{Field: FieldToolChoice, Status: FieldStatusChanged, Observed: true},
		{Field: FieldParallelToolCalls, Status: FieldStatusDropped, Observed: true},
		{Field: FieldResponseFormat, Status: FieldStatusRejected, Observed: true},
		{Field: FieldTools, Status: FieldStatusRejected},
		{Field: Field("sentinel_private_name"), Status: FieldStatusChanged, Observed: true},
	}

	attributes := snapshot.SpanAttributes(results)
	require.Equal(t, []attribute.KeyValue{
		attribute.String("llm_d.tool_calling.api_surface", "chat_completions"),
		attribute.Bool("llm_d.tool_calling.present", true),
		attribute.String("llm_d.tool_calling.tool_choice", "required"),
		attribute.String("llm_d.tool_calling.tool_count", "1"),
		attribute.String("llm_d.tool_calling.field.tools.status", "preserved"),
		attribute.String("llm_d.tool_calling.field.tool_choice.status", "changed"),
		attribute.String("llm_d.tool_calling.field.parallel_tool_calls.status", "dropped"),
		attribute.String("llm_d.tool_calling.field.response_format.status", "rejected"),
	}, attributes)
	var encoded strings.Builder
	for _, attribute := range attributes {
		encoded.WriteString(string(attribute.Key) + "=")
		encoded.WriteString(attribute.Value.String())
		encoded.WriteByte('\n')
	}
	require.Contains(t, encoded.String(), "llm_d.tool_calling.field.tools.status=preserved")
	require.NotContains(t, encoded.String(), "sentinel_private_name")
}

func TestSpanAttributesOmittedForRequestsWithoutToolCallingFields(t *testing.T) {
	snapshot, err := captureRequest(reqcommon.APITypeChatCompletions, map[string]any{"model": "test-model"})
	require.NoError(t, err)

	statuses, err := CompareRequests(snapshot, snapshot)
	require.NoError(t, err)
	require.Empty(t, snapshot.SpanAttributes(statuses))
}

func TestSpanAttributesRetainObservedToolCallingMutation(t *testing.T) {
	snapshot, err := captureRequest(reqcommon.APITypeChatCompletions, map[string]any{"model": "test-model"})
	require.NoError(t, err)

	attrs := snapshot.SpanAttributes([]FieldStatus{{
		Field:    FieldTools,
		Status:   FieldStatusChanged,
		Observed: true,
	}})
	require.Len(t, attrs, 3)
	require.Equal(t, "llm_d.tool_calling.field.tools.status", string(attrs[2].Key))
	require.Equal(t, string(FieldStatusChanged), attrs[2].Value.AsString())
}

func TestRequestSummaryMessagesToolChoice(t *testing.T) {
	snapshot, err := captureRequest(reqcommon.APITypeMessages, map[string]any{
		"tool_choice": map[string]any{"type": "tool", "name": "sentinel_private_name"},
	})
	require.NoError(t, err)

	summary := snapshot.Summary()
	require.True(t, summary.ToolChoicePresent)
	require.Equal(t, ToolChoiceNamed, summary.ToolChoiceKind)
	require.True(t, summary.ToolCallingPresent)
}

func TestSupportedRequestFields(t *testing.T) {
	require.Equal(t, []Field{FieldTools, FieldToolChoice, FieldParallelToolCalls, FieldResponseFormat}, mustFieldsForSurface(t, reqcommon.APITypeChatCompletions))
	require.Equal(t, []Field{FieldTools, FieldToolChoice}, mustFieldsForSurface(t, reqcommon.APITypeMessages))
	require.Equal(t, []Field{FieldTools, FieldToolChoice, FieldParallelToolCalls}, mustFieldsForSurface(t, reqcommon.APITypeResponses))
}

func TestCaptureRequestRejectsUnsupportedAPIs(t *testing.T) {
	for _, apiType := range []reqcommon.APIType{reqcommon.APITypeCompletions, reqcommon.APITypeVLLMGenerate, reqcommon.APITypeSGLangGenerate, reqcommon.APIType(-1)} {
		t.Run(apiType.String(), func(t *testing.T) {
			_, err := captureRequest(apiType, map[string]any{"tools": []any{}})
			require.Error(t, err)
			_, err = CaptureRequestJSON(apiType, []byte(`{"tools":[]}`))
			require.Error(t, err)
		})
	}
}

func TestToolCountBucket(t *testing.T) {
	tests := []struct {
		count int
		want  ToolCountBucket
	}{
		{count: 0, want: ""},
		{count: 1, want: ToolCountBucketOne},
		{count: 2, want: ToolCountBucketTwoToFive},
		{count: 5, want: ToolCountBucketTwoToFive},
		{count: 6, want: ToolCountBucketSixToTen},
		{count: 10, want: ToolCountBucketSixToTen},
		{count: 11, want: ToolCountBucketElevenPlus},
	}
	for _, tt := range tests {
		require.Equal(t, tt.want, bucketToolCount(tt.count))
	}
}

// captureRequest provides a decoded-body reference for JSON capture tests.
func captureRequest(surface reqcommon.APIType, body map[string]any) (RequestSnapshot, error) {
	fields, err := fieldsForSurface(surface)
	if err != nil {
		return RequestSnapshot{}, err
	}
	if body == nil {
		body = map[string]any{}
	}

	snapshot := RequestSnapshot{
		surface: surface,
		fields:  make(map[Field]capturedField, len(fields)),
		summary: RequestSummary{ToolChoiceKind: ToolChoiceUnknown},
	}
	for _, field := range fields {
		value, present := body[string(field)]
		captured := capturedField{present: present}
		if present {
			captured.value, err = json.Marshal(value)
			if err != nil {
				return RequestSnapshot{}, fmt.Errorf("marshal %s field: %w", field, err)
			}
		}
		snapshot.fields[field] = captured
		if present && field != FieldResponseFormat {
			snapshot.summary.ToolCallingPresent = true
		}
	}

	if tools, ok := body[string(FieldTools)].([]any); ok {
		if len(tools) > 0 {
			snapshot.summary.ToolCallingPresent = true
			snapshot.summary.ToolCountBucket = bucketToolCount(len(tools))
		}
	}
	if choice, present := body[string(FieldToolChoice)]; present {
		snapshot.summary.ToolChoicePresent = true
		snapshot.summary.ToolChoiceKind = normalizeToolChoiceValue(surface, choice)
	}

	return snapshot, nil
}

func resultStatus(t *testing.T, results []FieldStatus, field Field) FieldStatusValue {
	t.Helper()
	for _, result := range results {
		if result.Field == field {
			return result.Status
		}
	}
	t.Fatalf("field %q missing from comparison results", field)
	return ""
}

func resultFor(t *testing.T, results []FieldStatus, field Field) FieldStatus {
	t.Helper()
	for _, result := range results {
		if result.Field == field {
			return result
		}
	}
	t.Fatalf("field %q missing from comparison results", field)
	return FieldStatus{}
}

func mustFieldsForSurface(t *testing.T, surface reqcommon.APIType) []Field {
	t.Helper()
	fields, err := fieldsForSurface(surface)
	require.NoError(t, err)
	return fields
}
