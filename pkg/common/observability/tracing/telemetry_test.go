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
	"os"
	"testing"

	"github.com/go-logr/logr/testr"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"

	"github.com/llm-d/llm-d-router/version"
)

const testServiceName = "llm-d-test"

// clearEnv unsets keys for the duration of the test. t.Setenv records the original
// value so its cleanup restores it; the value it writes is discarded immediately.
func clearEnv(t *testing.T, keys ...string) {
	t.Helper()

	for _, key := range keys {
		if v, ok := os.LookupEnv(key); ok {
			t.Setenv(key, v)
			os.Unsetenv(key)
		}
	}
}

// attrValue returns the value of key in set, or "" when unset.
func attrValue(set *attribute.Set, key attribute.Key) string {
	v, ok := set.Value(key)
	if !ok {
		return ""
	}
	return v.AsString()
}

func TestNewResource(t *testing.T) {
	origBuildRef := version.BuildRef
	version.BuildRef = "test-build-ref"
	t.Cleanup(func() { version.BuildRef = origBuildRef })

	tests := []struct {
		name        string
		env         map[string]string
		wantService string
		wantExtra   map[attribute.Key]string
	}{
		{
			name:        "defaults when environment is empty",
			wantService: testServiceName,
		},
		{
			name:        "OTEL_SERVICE_NAME overrides the default",
			env:         map[string]string{"OTEL_SERVICE_NAME": "llm-d-epp"},
			wantService: "llm-d-epp",
		},
		{
			name:        "OTEL_RESOURCE_ATTRIBUTES service.name overrides the default",
			env:         map[string]string{"OTEL_RESOURCE_ATTRIBUTES": "service.name=from-attrs"},
			wantService: "from-attrs",
		},
		{
			name: "OTEL_SERVICE_NAME wins over OTEL_RESOURCE_ATTRIBUTES",
			env: map[string]string{
				"OTEL_SERVICE_NAME":        "from-service-name",
				"OTEL_RESOURCE_ATTRIBUTES": "service.name=from-attrs",
			},
			wantService: "from-service-name",
		},
		{
			name:        "OTEL_RESOURCE_ATTRIBUTES adds attributes alongside the defaults",
			env:         map[string]string{"OTEL_RESOURCE_ATTRIBUTES": "k8s.pod.name=epp-0,k8s.node.name=node-1"},
			wantService: testServiceName,
			wantExtra: map[attribute.Key]string{
				"k8s.pod.name":  "epp-0",
				"k8s.node.name": "node-1",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clearEnv(t, "OTEL_SERVICE_NAME", "OTEL_RESOURCE_ATTRIBUTES")
			for k, v := range tc.env {
				t.Setenv(k, v)
			}

			res, err := newResource(context.Background(), testServiceName)
			if err != nil {
				t.Fatalf("newResource() error = %v", err)
			}

			if res.SchemaURL() != semconv.SchemaURL {
				t.Errorf("schema URL = %q, want %q", res.SchemaURL(), semconv.SchemaURL)
			}

			set := res.Set()
			if got := attrValue(set, semconv.ServiceNameKey); got != tc.wantService {
				t.Errorf("service.name = %q, want %q", got, tc.wantService)
			}
			if got := attrValue(set, semconv.ServiceVersionKey); got != version.BuildRef {
				t.Errorf("service.version = %q, want %q", got, version.BuildRef)
			}
			for key, want := range tc.wantExtra {
				if got := attrValue(set, key); got != want {
					t.Errorf("%s = %q, want %q", key, got, want)
				}
			}
		})
	}
}

// resource.New reports a malformed OTEL_RESOURCE_ATTRIBUTES entry alongside a
// resource holding the entries it could parse. Tracing must keep that resource
// instead of failing, since InitTracing failing stops process startup.
func TestMalformedResourceAttributesDegradeRatherThanFail(t *testing.T) {
	clearEnv(t, "OTEL_SERVICE_NAME", "OTEL_RESOURCE_ATTRIBUTES", "OTEL_TRACES_EXPORTER")
	t.Setenv("OTEL_RESOURCE_ATTRIBUTES", "k8s.pod.name=epp-0,malformed,k8s.node.name=node-1")

	res, err := newResource(context.Background(), testServiceName)
	if err == nil {
		t.Fatal("newResource() error = nil, want the malformed entry reported")
	}

	set := res.Set()
	for key, want := range map[attribute.Key]string{
		"k8s.pod.name":  "epp-0",
		"k8s.node.name": "node-1",
		"service.name":  testServiceName,
	} {
		if got := attrValue(set, key); got != want {
			t.Errorf("%s = %q, want %q", key, got, want)
		}
	}

	origTP, origHandler, origProp := otel.GetTracerProvider(), otel.GetErrorHandler(), otel.GetTextMapPropagator()
	t.Cleanup(func() {
		otel.SetTracerProvider(origTP)
		otel.SetErrorHandler(origHandler)
		otel.SetTextMapPropagator(origProp)
	})

	shutdown, err := InitTracing(context.Background(), testr.New(t), testServiceName)
	if err != nil {
		t.Fatalf("InitTracing() error = %v, want startup to continue with the degraded resource", err)
	}
	t.Cleanup(func() {
		if err := shutdown(context.Background()); err != nil {
			t.Errorf("shutdown() error = %v", err)
		}
	})
}

// InitTracing must configure the SDK without writing to the process environment.
func TestInitTracingDoesNotMutateEnvironment(t *testing.T) {
	watched := []string{
		"OTEL_SERVICE_NAME",
		"OTEL_EXPORTER_OTLP_ENDPOINT",
		"OTEL_TRACES_EXPORTER",
		"OTEL_TRACES_SAMPLER",
		"OTEL_TRACES_SAMPLER_ARG",
	}
	clearEnv(t, watched...)

	origTP, origHandler, origProp := otel.GetTracerProvider(), otel.GetErrorHandler(), otel.GetTextMapPropagator()
	t.Cleanup(func() {
		otel.SetTracerProvider(origTP)
		otel.SetErrorHandler(origHandler)
		otel.SetTextMapPropagator(origProp)
	})

	shutdown, err := InitTracing(context.Background(), testr.New(t), testServiceName)
	if err != nil {
		t.Fatalf("InitTracing() error = %v", err)
	}
	t.Cleanup(func() {
		if err := shutdown(context.Background()); err != nil {
			t.Errorf("shutdown() error = %v", err)
		}
	})

	for _, key := range watched {
		if v, ok := os.LookupEnv(key); ok {
			t.Errorf("InitTracing set %s = %q, want unset", key, v)
		}
	}
}

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

// samplerEnv are the variables newSampler reads. Each subtest starts from them
// unset so an inherited value cannot decide the result.
var samplerEnv = []string{"OTEL_TRACES_SAMPLER", "OTEL_TRACES_SAMPLER_ARG"}

func TestNewSampler(t *testing.T) {
	tests := []struct {
		name    string
		env     map[string]string
		want    sdktrace.Sampler
		wantErr bool
	}{
		{
			name: "unset defaults to a tenth of traces",
			want: sdktrace.ParentBased(sdktrace.TraceIDRatioBased(0.1)),
		},
		{
			name: "always_on",
			env:  map[string]string{"OTEL_TRACES_SAMPLER": "always_on"},
			want: sdktrace.AlwaysSample(),
		},
		{
			name: "always_off",
			env:  map[string]string{"OTEL_TRACES_SAMPLER": "always_off"},
			want: sdktrace.NeverSample(),
		},
		{
			name: "parentbased_always_on",
			env:  map[string]string{"OTEL_TRACES_SAMPLER": "parentbased_always_on"},
			want: sdktrace.ParentBased(sdktrace.AlwaysSample()),
		},
		{
			name: "parentbased_always_off",
			env:  map[string]string{"OTEL_TRACES_SAMPLER": "parentbased_always_off"},
			want: sdktrace.ParentBased(sdktrace.NeverSample()),
		},
		{
			name: "traceidratio is not wrapped in a parent-based decorator",
			env:  map[string]string{"OTEL_TRACES_SAMPLER": "traceidratio", "OTEL_TRACES_SAMPLER_ARG": "0.25"},
			want: sdktrace.TraceIDRatioBased(0.25),
		},
		{
			name: "parentbased_traceidratio",
			env:  map[string]string{"OTEL_TRACES_SAMPLER": "parentbased_traceidratio", "OTEL_TRACES_SAMPLER_ARG": "0.5"},
			want: sdktrace.ParentBased(sdktrace.TraceIDRatioBased(0.5)),
		},
		{
			name: "ratio sampler without an argument keeps the default ratio",
			env:  map[string]string{"OTEL_TRACES_SAMPLER": "traceidratio"},
			want: sdktrace.TraceIDRatioBased(0.1),
		},
		{
			name: "the argument is ignored by samplers that take none",
			env:  map[string]string{"OTEL_TRACES_SAMPLER": "always_off", "OTEL_TRACES_SAMPLER_ARG": "nonsense"},
			want: sdktrace.NeverSample(),
		},
		{
			name:    "an unsupported sampler is reported",
			env:     map[string]string{"OTEL_TRACES_SAMPLER": "jaeger_remote"},
			want:    sdktrace.ParentBased(sdktrace.TraceIDRatioBased(0.1)),
			wantErr: true,
		},
		{
			name:    "an empty sampler is reported rather than treated as unset",
			env:     map[string]string{"OTEL_TRACES_SAMPLER": ""},
			want:    sdktrace.ParentBased(sdktrace.TraceIDRatioBased(0.1)),
			wantErr: true,
		},
		{
			name:    "an unparsable argument is reported and does not change the sampler shape",
			env:     map[string]string{"OTEL_TRACES_SAMPLER": "traceidratio", "OTEL_TRACES_SAMPLER_ARG": "0.1percent"},
			want:    sdktrace.TraceIDRatioBased(0.1),
			wantErr: true,
		},
		{
			name:    "an argument above one is reported instead of clamping to always-on",
			env:     map[string]string{"OTEL_TRACES_SAMPLER": "parentbased_traceidratio", "OTEL_TRACES_SAMPLER_ARG": "10"},
			want:    sdktrace.ParentBased(sdktrace.TraceIDRatioBased(0.1)),
			wantErr: true,
		},
		{
			name:    "a negative argument is reported instead of clamping to always-off",
			env:     map[string]string{"OTEL_TRACES_SAMPLER": "parentbased_traceidratio", "OTEL_TRACES_SAMPLER_ARG": "-1"},
			want:    sdktrace.ParentBased(sdktrace.TraceIDRatioBased(0.1)),
			wantErr: true,
		},
		{
			name: "the range bounds are accepted",
			env:  map[string]string{"OTEL_TRACES_SAMPLER": "parentbased_traceidratio", "OTEL_TRACES_SAMPLER_ARG": "1"},
			want: sdktrace.ParentBased(sdktrace.TraceIDRatioBased(1)),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clearEnv(t, samplerEnv...)
			for key, value := range tc.env {
				t.Setenv(key, value)
			}

			got, err := newSampler()
			if (err != nil) != tc.wantErr {
				t.Fatalf("newSampler() error = %v, wantErr %v", err, tc.wantErr)
			}
			if got.Description() != tc.want.Description() {
				t.Errorf("newSampler() = %s, want %s", got.Description(), tc.want.Description())
			}
		})
	}
}

// A rejected sampler configuration must not stop process startup, since sampling
// is a runtime knob rather than a precondition for serving traffic.
func TestInitTracingSurvivesRejectedSampler(t *testing.T) {
	clearEnv(t, append(samplerEnv, "OTEL_TRACES_EXPORTER")...)
	t.Setenv("OTEL_TRACES_SAMPLER", "parentbased_traceidratio")
	t.Setenv("OTEL_TRACES_SAMPLER_ARG", "not-a-ratio")

	origTP, origHandler, origProp := otel.GetTracerProvider(), otel.GetErrorHandler(), otel.GetTextMapPropagator()
	t.Cleanup(func() {
		otel.SetTracerProvider(origTP)
		otel.SetErrorHandler(origHandler)
		otel.SetTextMapPropagator(origProp)
	})

	shutdown, err := InitTracing(context.Background(), testr.New(t), testServiceName)
	if err != nil {
		t.Fatalf("InitTracing() error = %v, want startup to continue at the default ratio", err)
	}
	t.Cleanup(func() {
		if err := shutdown(context.Background()); err != nil {
			t.Errorf("shutdown() error = %v", err)
		}
	})
}
