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
	"errors"
	"fmt"
	"math/big"
	"slices"
	"strings"

	"go.opentelemetry.io/otel/attribute"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// Field is a request field compared at a request-mutating boundary.
type Field string

const (
	FieldTools             Field = "tools"
	FieldToolChoice        Field = "tool_choice"
	FieldParallelToolCalls Field = "parallel_tool_calls"
	FieldResponseFormat    Field = "response_format"
)

// FieldStatusValue is the bounded outcome for a compared request field.
type FieldStatusValue string

const (
	FieldStatusPreserved FieldStatusValue = "preserved"
	FieldStatusChanged   FieldStatusValue = "changed"
	FieldStatusDropped   FieldStatusValue = "dropped"
	FieldStatusRejected  FieldStatusValue = "rejected"
)

// Request field outcome metric labels and component values.
const (
	MetricLabelComponent = "component"
	MetricLabelDirection = "direction"
	MetricLabelField     = "field"
	MetricLabelStatus    = "status"

	ComponentEPP = "epp"

	DirectionRequest = "request"
)

// ToolChoiceKind is a sanitized tool_choice category.
type ToolChoiceKind string

const (
	ToolChoiceAuto     ToolChoiceKind = "auto"
	ToolChoiceNone     ToolChoiceKind = "none"
	ToolChoiceRequired ToolChoiceKind = "required"
	ToolChoiceNamed    ToolChoiceKind = "named"
	ToolChoiceUnknown  ToolChoiceKind = "unknown"
)

// ToolCountBucket is a bounded bucket for the number of tool definitions.
type ToolCountBucket string

const (
	ToolCountBucketOne        ToolCountBucket = "1"
	ToolCountBucketTwoToFive  ToolCountBucket = "2-5"
	ToolCountBucketSixToTen   ToolCountBucket = "6-10"
	ToolCountBucketElevenPlus ToolCountBucket = "11+"
)

// FieldStatus reports the comparison result for one supported field.
type FieldStatus struct {
	Field    Field
	Status   FieldStatusValue
	Observed bool
}

// RequestSummary contains only bounded, non-content-derived telemetry values.
type RequestSummary struct {
	ToolCallingPresent bool
	ToolChoicePresent  bool
	ToolChoiceKind     ToolChoiceKind
	ToolCountBucket    ToolCountBucket
}

type capturedField struct {
	present bool
	value   []byte
}

// RequestSnapshot stores serialized JSON for supported fields so later body
// mutations cannot alter the comparison baseline. Field contents remain
// private and are never included in Summary or metric labels.
type RequestSnapshot struct {
	surface reqcommon.APIType
	fields  map[Field]capturedField
	summary RequestSummary
}

// CaptureRequestJSON decodes only supported fields in a JSON request object.
func CaptureRequestJSON(surface reqcommon.APIType, body []byte) (RequestSnapshot, error) {
	fields, err := fieldsForSurface(surface)
	if err != nil {
		return RequestSnapshot{}, err
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '{' || !json.Valid(body) {
		return RequestSnapshot{}, errors.New("decode request body: expected valid JSON object")
	}
	captured, err := captureJSONFields(fields, trimmed)
	if err != nil {
		return RequestSnapshot{}, err
	}
	snapshot := RequestSnapshot{
		surface: surface,
		fields:  captured,
		summary: RequestSummary{ToolChoiceKind: ToolChoiceUnknown},
	}
	for _, field := range fields {
		if value := snapshot.fields[field]; value.present {
			raw := value.value
			if field != FieldResponseFormat {
				snapshot.summary.ToolCallingPresent = true
			}
			switch field {
			case FieldTools:
				var tools []json.RawMessage
				if json.Unmarshal(raw, &tools) == nil && len(tools) > 0 {
					snapshot.summary.ToolCountBucket = bucketToolCount(len(tools))
				}
			case FieldToolChoice:
				decoder := json.NewDecoder(bytes.NewReader(raw))
				decoder.UseNumber()
				var value any
				if err := decoder.Decode(&value); err != nil {
					return RequestSnapshot{}, fmt.Errorf("decode %s field: %w", field, err)
				}
				snapshot.summary.ToolChoicePresent = true
				snapshot.summary.ToolChoiceKind = normalizeToolChoiceValue(surface, value)
			}
		}
	}
	return snapshot, nil
}

// captureJSONFields walks an object already validated by encoding/json. Only
// supported top-level values are copied; unknown values are skipped by boundary.
func captureJSONFields(fields []Field, body []byte) (map[Field]capturedField, error) {
	var captured map[Field]capturedField
	for rest := body[1:]; ; {
		rest = bytes.TrimLeft(rest, " \t\r\n")
		if rest[0] == '}' {
			return captured, nil
		}
		end := jsonStringEnd(rest)
		rawKey := rest[:end]
		key := rawKey[1 : end-1]
		rest = bytes.TrimLeft(rest[end:], " \t\r\n")[1:] // Validated colon.
		rest = bytes.TrimLeft(rest, " \t\r\n")
		end = jsonValueEnd(rest)
		if bytes.IndexByte(key, '\\') >= 0 {
			var name string
			if err := json.Unmarshal(rawKey, &name); err != nil {
				return nil, fmt.Errorf("decode request field name: %w", err)
			}
			key = []byte(name)
		}
		for _, field := range fields {
			if bytes.Equal(key, []byte(field)) {
				if captured == nil {
					captured = make(map[Field]capturedField, len(fields))
				}
				captured[field] = capturedField{present: true, value: bytes.Clone(rest[:end])}
				break
			}
		}
		rest = bytes.TrimLeft(rest[end:], " \t\r\n")
		if rest[0] == ',' {
			rest = rest[1:]
		}
	}
}

// jsonStringEnd and jsonValueEnd find boundaries only in validated JSON.
func jsonStringEnd(body []byte) int {
	for offset := 1; ; {
		end := offset + bytes.IndexByte(body[offset:], '"')
		backslashes := 0
		for i := end - 1; i >= 0 && body[i] == '\\'; i-- {
			backslashes++
		}
		if backslashes%2 == 0 {
			return end + 1
		}
		offset = end + 1
	}
}

func jsonValueEnd(body []byte) int {
	switch body[0] {
	case '"':
		return jsonStringEnd(body)
	case '{', '[':
		depth := 0
		for i := 0; i < len(body); i++ {
			switch body[i] {
			case '"':
				i += jsonStringEnd(body[i:]) - 1
			case '{', '[':
				depth++
			case '}', ']':
				depth--
				if depth == 0 {
					return i + 1
				}
			}
		}
	}
	return bytes.IndexAny(body, ",}] \t\r\n")
}

// CompareRequests compares every supported field for the common API surface.
// An absent field and an explicit null are different values. Fields absent on
// both sides are reported as preserved. Numbers are compared exactly, allowing
// equivalent decimal and exponent notation.
func CompareRequests(before, after RequestSnapshot) ([]FieldStatus, error) {
	if before.surface != after.surface {
		return nil, fmt.Errorf("cannot compare API surfaces %q and %q", before.surface, after.surface)
	}
	fields, err := fieldsForSurface(before.surface)
	if err != nil {
		return nil, err
	}

	results := make([]FieldStatus, 0, len(fields))
	for _, field := range fields {
		left := before.fields[field]
		right := after.fields[field]
		status := FieldStatusPreserved
		switch {
		case left.present && !right.present:
			status = FieldStatusDropped
		case !left.present && right.present:
			status = FieldStatusChanged
		case left.present && right.present && !equalJSONFieldValues(left.value, right.value):
			status = FieldStatusChanged
		}
		results = append(results, FieldStatus{
			Field:    field,
			Status:   status,
			Observed: left.present || right.present,
		})
	}
	return results, nil
}

// Only fields whose bytes differ need decoding for semantic comparison.
// UseNumber preserves precision; normalization equates forms such as 1 and 1.0.
func equalJSONFieldValues(left, right []byte) bool {
	if bytes.Equal(left, right) {
		return true
	}
	leftDecoder := json.NewDecoder(bytes.NewReader(left))
	leftDecoder.UseNumber()
	rightDecoder := json.NewDecoder(bytes.NewReader(right))
	rightDecoder.UseNumber()
	var leftValue, rightValue any
	if leftDecoder.Decode(&leftValue) != nil || rightDecoder.Decode(&rightValue) != nil {
		return false
	}
	return equalJSONValues(leftValue, rightValue)
}

func equalJSONValues(left, right any) bool {
	switch left := left.(type) {
	case map[string]any:
		right, ok := right.(map[string]any)
		if !ok || len(left) != len(right) {
			return false
		}
		for key, value := range left {
			other, present := right[key]
			if !present || !equalJSONValues(value, other) {
				return false
			}
		}
		return true
	case []any:
		right, ok := right.([]any)
		if !ok || len(left) != len(right) {
			return false
		}
		for i, value := range left {
			if !equalJSONValues(value, right[i]) {
				return false
			}
		}
		return true
	case json.Number:
		right, ok := right.(json.Number)
		return ok && (left == right || normalizeJSONNumber(left) == normalizeJSONNumber(right))
	default:
		return left == right
	}
}

// Normalize numbers to a signed coefficient and decimal exponent without rounding.
// Keep the exponent separate to avoid expanding numbers such as 1e1000000000.
func normalizeJSONNumber(number json.Number) string {
	value := string(number)
	sign := ""
	if value[0] == '-' {
		sign = "-"
		value = value[1:]
	}
	var exponent big.Int
	if i := strings.IndexAny(value, "eE"); i >= 0 {
		exponent.SetString(value[i+1:], 10)
		value = value[:i]
	}
	fractionalDigits := 0
	if i := strings.IndexByte(value, '.'); i >= 0 {
		fractionalDigits = len(value) - i - 1
		value = value[:i] + value[i+1:]
	}
	value = strings.TrimLeft(value, "0")
	if value == "" {
		return "0"
	}
	coefficient := strings.TrimRight(value, "0")
	exponent.Add(&exponent, big.NewInt(int64(len(value)-len(coefficient)-fractionalDigits)))
	return sign + coefficient + "e" + exponent.String()
}

// RejectedFieldStatuses reports explicitly rejected fields present in the snapshot.
func RejectedFieldStatuses(snapshot RequestSnapshot, rejectedFields ...Field) []FieldStatus {
	fields, err := fieldsForSurface(snapshot.surface)
	if err != nil {
		return nil
	}

	results := make([]FieldStatus, 0, len(fields))
	for _, field := range fields {
		if snapshot.fields[field].present && slices.Contains(rejectedFields, field) {
			results = append(results, FieldStatus{
				Field:    field,
				Status:   FieldStatusRejected,
				Observed: true,
			})
		}
	}
	return results
}

// Summary returns the snapshot's bounded telemetry values without field data.
func (snapshot RequestSnapshot) Summary() RequestSummary {
	return snapshot.summary
}

// SpanAttributes returns bounded summary values and observed field statuses.
// It returns nil when there is no tool-field presence or observed field outcome.
func (snapshot RequestSnapshot) SpanAttributes(statuses []FieldStatus) []attribute.KeyValue {
	if !snapshot.summary.ToolCallingPresent && !hasObservedFieldStatus(statuses) {
		return nil
	}

	attrs := make([]attribute.KeyValue, 0, 3)
	attrs = append(attrs,
		semconv.LLMDToolCallingAPISurfaceKey.String(snapshot.surface.String()),
		semconv.LLMDToolCallingPresentKey.Bool(snapshot.summary.ToolCallingPresent),
	)
	if snapshot.summary.ToolCallingPresent {
		attrs = append(attrs, semconv.LLMDToolCallingToolChoiceKey.String(string(snapshot.summary.ToolChoiceKind)))
	}
	if snapshot.summary.ToolCountBucket != "" {
		attrs = append(attrs, semconv.LLMDToolCallingToolCountKey.String(string(snapshot.summary.ToolCountBucket)))
	}
	for _, result := range statuses {
		if !result.Observed {
			continue
		}
		var key attribute.Key
		switch result.Field {
		case FieldTools:
			key = semconv.LLMDToolCallingFieldToolsStatusKey
		case FieldToolChoice:
			key = semconv.LLMDToolCallingFieldToolChoiceStatusKey
		case FieldParallelToolCalls:
			key = semconv.LLMDToolCallingFieldParallelToolCallsStatusKey
		case FieldResponseFormat:
			key = semconv.LLMDToolCallingFieldResponseFormatStatusKey
		default:
			continue
		}
		attrs = append(attrs, key.String(string(result.Status)))
	}
	return attrs
}

func hasObservedFieldStatus(statuses []FieldStatus) bool {
	for _, status := range statuses {
		if status.Observed {
			return true
		}
	}
	return false
}

func fieldsForSurface(surface reqcommon.APIType) ([]Field, error) {
	switch surface {
	case reqcommon.APITypeChatCompletions:
		return []Field{FieldTools, FieldToolChoice, FieldParallelToolCalls, FieldResponseFormat}, nil
	case reqcommon.APITypeResponses:
		return []Field{FieldTools, FieldToolChoice, FieldParallelToolCalls}, nil
	case reqcommon.APITypeMessages:
		return []Field{FieldTools, FieldToolChoice}, nil
	default:
		return nil, fmt.Errorf("unsupported tool-calling API surface %q", surface)
	}
}

func normalizeToolChoiceValue(surface reqcommon.APIType, value any) ToolChoiceKind {
	switch choice := value.(type) {
	case string:
		switch choice {
		case "auto":
			return ToolChoiceAuto
		case "none":
			return ToolChoiceNone
		case "required":
			return ToolChoiceRequired
		default:
			return ToolChoiceUnknown
		}
	case map[string]any:
		switch choiceType, _ := choice["type"].(string); choiceType {
		case "auto":
			return ToolChoiceAuto
		case "none":
			return ToolChoiceNone
		case "any":
			return ToolChoiceRequired
		case "tool":
			if name, ok := choice["name"].(string); ok && name != "" {
				return ToolChoiceNamed
			}
		case "function":
			function := choice
			if surface != reqcommon.APITypeResponses {
				function, _ = choice["function"].(map[string]any)
			}
			if name, ok := function["name"].(string); ok && name != "" {
				return ToolChoiceNamed
			}
		}
	}
	return ToolChoiceUnknown
}

func bucketToolCount(count int) ToolCountBucket {
	switch {
	case count <= 0:
		return ""
	case count == 1:
		return ToolCountBucketOne
	case count <= 5:
		return ToolCountBucketTwoToFive
	case count <= 10:
		return ToolCountBucketSixToTen
	default:
		return ToolCountBucketElevenPlus
	}
}
