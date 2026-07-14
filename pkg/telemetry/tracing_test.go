/*
Copyright 2025 The llm-d Authors.

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

package telemetry_test

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/instrumentation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/llm-d/llm-d-router/pkg/telemetry"
	"github.com/llm-d/llm-d-router/version"
)

// recordScope starts and ends a single span from the given tracer and returns
// the instrumentation scope recorded for it.
func recordScope(t *testing.T, scope ...string) instrumentation.Scope {
	t.Helper()

	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(tp)

	_, span := telemetry.Tracer(scope...).Start(context.Background(), "test")
	span.End()

	ended := recorder.Ended()
	if len(ended) != 1 {
		t.Fatalf("expected 1 recorded span, got %d", len(ended))
	}
	return ended[0].InstrumentationScope()
}

func TestTracerDefaultScope(t *testing.T) {
	got := recordScope(t)

	if got.Name != telemetry.InstrumentationName {
		t.Errorf("scope name = %q, want %q", got.Name, telemetry.InstrumentationName)
	}
	if got.Version != version.BuildRef {
		t.Errorf("scope version = %q, want %q", got.Version, version.BuildRef)
	}
}

func TestTracerCustomScope(t *testing.T) {
	const scope = "llm-d-kv-cache/pkg/kvcache"
	got := recordScope(t, scope)

	if got.Name != scope {
		t.Errorf("scope name = %q, want %q", got.Name, scope)
	}
}

func TestTracerEmptyScopeFallsBackToDefault(t *testing.T) {
	got := recordScope(t, "")

	if got.Name != telemetry.InstrumentationName {
		t.Errorf("scope name = %q, want %q", got.Name, telemetry.InstrumentationName)
	}
}

func TestTracerAttachesCommitSHA(t *testing.T) {
	got := recordScope(t)

	val, ok := got.Attributes.Value(attribute.Key("commit-sha"))
	if !ok {
		t.Fatal("scope is missing commit-sha attribute")
	}
	if val.AsString() != version.CommitSHA {
		t.Errorf("commit-sha = %q, want %q", val.AsString(), version.CommitSHA)
	}
}
