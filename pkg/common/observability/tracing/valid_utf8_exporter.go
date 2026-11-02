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

package tracing

import (
	"context"
	"slices"
	"strings"
	"unicode/utf8"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// validUTF8Exporter replaces invalid UTF-8 in span strings before export: each
// run of invalid bytes becomes one U+FFFD. An OTLP string field must be valid
// UTF-8, and one invalid value fails the marshaling of the whole export
// request, so every span in the batch is lost. Request paths and header values
// reach span names, attributes and error messages as raw bytes.
//
// It covers the span name, string attribute values on spans and their events,
// and the status description. Attribute keys, event names, link attributes,
// collection-valued attributes, and scope and resource strings are unchecked:
// no request-input path to them was found. It wraps every exporter so the
// check lives in one place.
type validUTF8Exporter struct {
	sdktrace.SpanExporter
}

func (e validUTF8Exporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	cloned := false
	for i, span := range spans {
		valid, changed := withValidUTF8(span)
		if !changed {
			continue
		}
		if !cloned {
			spans = slices.Clone(spans)
			cloned = true
		}
		spans[i] = valid
	}
	return e.SpanExporter.ExportSpans(ctx, spans)
}

// validUTF8Span overrides the string fields of the span it embeds.
type validUTF8Span struct {
	sdktrace.ReadOnlySpan

	name       string
	attributes []attribute.KeyValue
	events     []sdktrace.Event
	status     sdktrace.Status
}

func (s validUTF8Span) Name() string                     { return s.name }
func (s validUTF8Span) Attributes() []attribute.KeyValue { return s.attributes }
func (s validUTF8Span) Events() []sdktrace.Event         { return s.events }
func (s validUTF8Span) Status() sdktrace.Status          { return s.status }

// withValidUTF8 returns span unchanged, and false, when all its strings are
// valid. The slices a span returns may be shared with the SDK, so they are
// cloned before a replacement is written.
func withValidUTF8(span sdktrace.ReadOnlySpan) (sdktrace.ReadOnlySpan, bool) {
	attributes, changed := validUTF8Attributes(span.Attributes())

	name := span.Name()
	if !utf8.ValidString(name) {
		name = strings.ToValidUTF8(name, "\uFFFD")
		changed = true
	}

	status := span.Status()
	if !utf8.ValidString(status.Description) {
		status.Description = strings.ToValidUTF8(status.Description, "\uFFFD")
		changed = true
	}

	events := span.Events()
	eventsCloned := false
	for i := range events {
		eventAttributes, eventChanged := validUTF8Attributes(events[i].Attributes)
		if !eventChanged {
			continue
		}
		if !eventsCloned {
			events = slices.Clone(events)
			eventsCloned = true
		}
		events[i].Attributes = eventAttributes
		changed = true
	}

	if !changed {
		return span, false
	}
	return validUTF8Span{ReadOnlySpan: span, name: name, attributes: attributes, events: events, status: status}, true
}

func validUTF8Attributes(attrs []attribute.KeyValue) ([]attribute.KeyValue, bool) {
	cloned := false
	for i, kv := range attrs {
		if kv.Value.Type() != attribute.STRING {
			continue
		}
		value := kv.Value.AsString()
		if utf8.ValidString(value) {
			continue
		}
		if !cloned {
			attrs = slices.Clone(attrs)
			cloned = true
		}
		attrs[i] = kv.Key.String(strings.ToValidUTF8(value, "\uFFFD"))
	}
	return attrs, cloned
}
