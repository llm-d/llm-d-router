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

package tracing

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/llm-d/llm-d-router/version"
)

func TestTracer(t *testing.T) {
	const (
		testBuildRef  = "test-build-ref"
		testCommitSHA = "test-commit-sha"
	)

	origBuildRef, origCommitSHA := version.BuildRef, version.CommitSHA
	version.BuildRef, version.CommitSHA = testBuildRef, testCommitSHA
	t.Cleanup(func() {
		version.BuildRef, version.CommitSHA = origBuildRef, origCommitSHA
	})

	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	origTP := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(origTP) })

	tests := []struct {
		name      string
		scope     []string
		wantScope string
	}{
		{name: "default scope", scope: nil, wantScope: instrumentationName},
		{name: "custom scope", scope: []string{"llm-d-router/pkg/epp/handlers"}, wantScope: "llm-d-router/pkg/epp/handlers"},
		{name: "empty scope falls back to default", scope: []string{""}, wantScope: instrumentationName},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			recorder.Reset()

			_, span := Tracer(tc.scope...).Start(context.Background(), "test-span")
			span.End()

			ended := recorder.Ended()
			if len(ended) != 1 {
				t.Fatalf("expected 1 recorded span, got %d", len(ended))
			}

			scope := ended[0].InstrumentationScope()
			if scope.Name != tc.wantScope {
				t.Errorf("scope name = %q, want %q", scope.Name, tc.wantScope)
			}
			if scope.Version != testBuildRef {
				t.Errorf("scope version = %q, want %q", scope.Version, testBuildRef)
			}

			commitSHA, ok := scope.Attributes.Value(attribute.Key("commit-sha"))
			if !ok {
				t.Fatal("commit-sha scope attribute not set")
			}
			if commitSHA.AsString() != testCommitSHA {
				t.Errorf("commit-sha = %q, want %q", commitSHA.AsString(), testCommitSHA)
			}
		})
	}
}
