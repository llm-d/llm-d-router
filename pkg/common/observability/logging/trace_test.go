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

package logging

import (
	"context"
	"testing"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel/trace"
)

type capturedInfo struct {
	msg string
	kv  []any
}

// captureSink records Info calls, including values accumulated via WithValues.
type captureSink struct {
	keys  []any
	infos *[]capturedInfo
}

func (s *captureSink) Init(_ logr.RuntimeInfo) {}
func (s *captureSink) Enabled(_ int) bool      { return true }
func (s *captureSink) Info(_ int, msg string, kv ...any) {
	all := append(append([]any{}, s.keys...), kv...)
	*s.infos = append(*s.infos, capturedInfo{msg: msg, kv: all})
}
func (s *captureSink) Error(_ error, _ string, _ ...any) {}
func (s *captureSink) WithValues(kv ...any) logr.LogSink {
	n := *s
	n.keys = append(append([]any{}, s.keys...), kv...)
	return &n
}
func (s *captureSink) WithName(_ string) logr.LogSink { return s }

func valueOf(kv []any, key string) (any, bool) {
	for i := 0; i+1 < len(kv); i += 2 {
		if kv[i] == key {
			return kv[i+1], true
		}
	}
	return nil, false
}

func TestWithTrace_ValidSpan(t *testing.T) {
	t.Parallel()

	traceID, err := trace.TraceIDFromHex("0af7651916cd43dd8448eb211c80319c")
	if err != nil {
		t.Fatalf("TraceIDFromHex: %v", err)
	}
	spanID, err := trace.SpanIDFromHex("b7ad6b7169203331")
	if err != nil {
		t.Fatalf("SpanIDFromHex: %v", err)
	}
	sc := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    traceID,
		SpanID:     spanID,
		TraceFlags: trace.FlagsSampled,
	})
	ctx := trace.ContextWithSpanContext(context.Background(), sc)

	var infos []capturedInfo
	logger := logr.New(&captureSink{infos: &infos})
	WithTrace(ctx, logger).Info("correlated")

	if len(infos) != 1 {
		t.Fatalf("expected 1 info call, got %d", len(infos))
	}
	gotTrace, ok := valueOf(infos[0].kv, TraceIDKey)
	if !ok || gotTrace != traceID.String() {
		t.Errorf("trace_id = %v, want %s", gotTrace, traceID.String())
	}
	gotSpan, ok := valueOf(infos[0].kv, SpanIDKey)
	if !ok || gotSpan != spanID.String() {
		t.Errorf("span_id = %v, want %s", gotSpan, spanID.String())
	}
}

func TestWithTrace_NoSpanUnchanged(t *testing.T) {
	t.Parallel()

	var infos []capturedInfo
	logger := logr.New(&captureSink{infos: &infos})
	WithTrace(context.Background(), logger).Info("uncorrelated")

	if len(infos) != 1 {
		t.Fatalf("expected 1 info call, got %d", len(infos))
	}
	if _, ok := valueOf(infos[0].kv, TraceIDKey); ok {
		t.Error("trace_id should not be present without a span")
	}
	if _, ok := valueOf(infos[0].kv, SpanIDKey); ok {
		t.Error("span_id should not be present without a span")
	}
}

func TestWithTrace_InvalidSpanUnchanged(t *testing.T) {
	t.Parallel()

	// Zero IDs produce an invalid SpanContext.
	sc := trace.NewSpanContext(trace.SpanContextConfig{})
	ctx := trace.ContextWithSpanContext(context.Background(), sc)

	var infos []capturedInfo
	logger := logr.New(&captureSink{infos: &infos})
	WithTrace(ctx, logger).Info("invalid-span")

	if len(infos) != 1 {
		t.Fatalf("expected 1 info call, got %d", len(infos))
	}
	if _, ok := valueOf(infos[0].kv, TraceIDKey); ok {
		t.Error("trace_id should not be present for invalid span")
	}
	if _, ok := valueOf(infos[0].kv, SpanIDKey); ok {
		t.Error("span_id should not be present for invalid span")
	}
}
