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

package tracing

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	tracenoop "go.opentelemetry.io/otel/trace/noop"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
)

const testFairnessID = "team-a"

// installAttributionRecorder makes a recording SDK provider global, with the
// same attribution span processor InitTracing registers, and returns the
// recorder.
func installAttributionRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()

	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(NewRequestAttributionProcessor()),
		sdktrace.WithSpanProcessor(recorder),
	)
	original := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() {
		otel.SetTracerProvider(original)
		_ = tp.Shutdown(context.Background())
	})

	return recorder
}

// endedAttribution returns the paired attribution values recorded on the
// ended span named name.
func endedAttribution(t *testing.T, recorder *tracetest.SpanRecorder, name string) (id, source string, present bool) {
	t.Helper()

	for _, span := range recorder.Ended() {
		if span.Name() != name {
			continue
		}
		var hasID, hasSource bool
		for _, attr := range span.Attributes() {
			switch attr.Key {
			case semconv.LLMDEPPFairnessIDKey:
				id, hasID = attr.Value.AsString(), true
			case semconv.LLMDEPPFairnessSourceKey:
				source, hasSource = attr.Value.AsString(), true
			}
		}
		return id, source, hasID && hasSource
	}

	t.Fatalf("no ended span named %q; recorded %d spans", name, len(recorder.Ended()))
	return "", "", false
}

func TestRequestAttributionResolution(t *testing.T) {
	tests := []struct {
		name       string
		headerID   string
		wantID     string
		wantSource string
	}{
		{
			name:       "empty header defaults",
			wantID:     DefaultAttributionID,
			wantSource: AttributionSourceDefault,
		},
		{
			name:       "header becomes fairness identity",
			headerID:   testFairnessID,
			wantID:     testFairnessID,
			wantSource: AttributionSourceHeader,
		},
		{
			name:       "header identity preserved verbatim",
			headerID:   "a-very-long-identity-with-/slashes-and-uppercase-KEPT-verbatim",
			wantID:     "a-very-long-identity-with-/slashes-and-uppercase-KEPT-verbatim",
			wantSource: AttributionSourceHeader,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := BeginRequestAttribution(context.Background(), tc.headerID)

			id, source, ok := RequestAttribution(ctx)
			if !ok {
				t.Fatal("RequestAttribution() ok = false, want true after BeginRequestAttribution")
			}
			if id != tc.wantID || source != tc.wantSource {
				t.Errorf("RequestAttribution() = (%q, %q), want (%q, %q)", id, source, tc.wantID, tc.wantSource)
			}
		})
	}
}

func TestSetRequestAttributionNormalizes(t *testing.T) {
	tests := []struct {
		name       string
		id         string
		source     string
		wantID     string
		wantSource string
	}{
		{
			name:       "agent identity preserved",
			id:         "agent-7",
			source:     AttributionSourceAgentIdentity,
			wantID:     "agent-7",
			wantSource: AttributionSourceAgentIdentity,
		},
		{
			name:       "empty identity defaults",
			source:     AttributionSourceAgentIdentity,
			wantID:     DefaultAttributionID,
			wantSource: AttributionSourceDefault,
		},
		{
			name:       "unknown source reported as default",
			id:         testFairnessID,
			source:     "unknown",
			wantID:     testFairnessID,
			wantSource: AttributionSourceDefault,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := BeginRequestAttribution(context.Background(), "")
			SetRequestAttribution(ctx, tc.id, tc.source)

			if id, source, _ := RequestAttribution(ctx); id != tc.wantID || source != tc.wantSource {
				t.Errorf("RequestAttribution() = (%q, %q), want (%q, %q)", id, source, tc.wantID, tc.wantSource)
			}
		})
	}
}

// An unattributed context has to be distinguishable from one that resolved to
// default-flow: background spans must not look like unattributed requests.
func TestRequestAttributionAbsent(t *testing.T) {
	if id, source, ok := RequestAttribution(context.Background()); ok {
		t.Errorf("RequestAttribution() = (%q, %q, true), want ok = false on a plain context", id, source)
	}
}

func TestSetRequestAttributionVisibleAcrossDerivedContexts(t *testing.T) {
	root := BeginRequestAttribution(context.Background(), "")
	derived, cancel := context.WithCancel(root)
	defer cancel()

	SetRequestAttribution(derived, testFairnessID, AttributionSourceAgentIdentity)

	for name, ctx := range map[string]context.Context{"entry point": root, "derived": derived} {
		id, source, ok := RequestAttribution(ctx)
		if !ok || id != testFairnessID || source != AttributionSourceAgentIdentity {
			t.Errorf("%s: RequestAttribution() = (%q, %q, %v), want (%s, agent_identity, true)", name, id, source, ok, testFairnessID)
		}
	}
}

func TestSpanProcessorAttributesRequestSpanTree(t *testing.T) {
	recorder := installAttributionRecorder(t)
	tracer := Tracer("attribution-test")

	ctx := BeginRequestAttribution(context.Background(), testFairnessID)
	ctx, root := tracer.Start(ctx, "request")
	_, child := tracer.Start(ctx, "child")
	child.End()
	root.End()

	for _, name := range []string{"request", "child"} {
		id, source, present := endedAttribution(t, recorder, name)
		if !present {
			t.Fatalf("span %q carries no paired attribution", name)
		}
		if id != testFairnessID || source != AttributionSourceHeader {
			t.Errorf("span %q attribution = (%q, %q), want (team-a, header)", name, id, source)
		}
	}
}

func TestAttributeRequestRefreshesOpenSpan(t *testing.T) {
	recorder := installAttributionRecorder(t)
	tracer := Tracer("attribution-test")

	ctx := BeginRequestAttribution(context.Background(), "")
	ctx, span := tracer.Start(ctx, "request")

	SetRequestAttribution(ctx, "agent-7", AttributionSourceAgentIdentity)
	AttributeRequest(ctx, span)
	span.End()

	id, source, present := endedAttribution(t, recorder, "request")
	if !present || id != "agent-7" || source != AttributionSourceAgentIdentity {
		t.Errorf("request attribution = (%q, %q, %v), want (agent-7, agent_identity, true)", id, source, present)
	}
}

// Spans of one concurrent request must never report another request's fairness
// identity. Run under -race.
func TestSpanAttributionIsolatedAcrossConcurrentRequests(t *testing.T) {
	recorder := installAttributionRecorder(t)
	tracer := Tracer("attribution-test")

	const requests = 16

	var wg sync.WaitGroup
	for i := range requests {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()

			team := fmt.Sprintf("team-%02d", i)
			ctx := BeginRequestAttribution(context.Background(), team)
			ctx, root := tracer.Start(ctx, "request/"+team)
			_, child := tracer.Start(ctx, "child/"+team)
			child.End()
			root.End()
		}(i)
	}
	wg.Wait()

	if got := len(recorder.Ended()); got != 2*requests {
		t.Fatalf("ended span count = %d, want %d", got, 2*requests)
	}

	for _, span := range recorder.Ended() {
		_, team, found := strings.Cut(span.Name(), "/")
		if !found {
			t.Fatalf("unexpected span %q", span.Name())
		}

		id, source, present := endedAttribution(t, recorder, span.Name())
		if !present {
			t.Errorf("span %q carries no paired attribution", span.Name())
			continue
		}
		if id != team || source != AttributionSourceHeader {
			t.Errorf("span %q attribution = (%q, %q), want (%q, header)", span.Name(), id, source, team)
		}
	}
}

// The Director resolves the fairness identity while other goroutines of the same
// request start spans. Every span must carry a whole pair from before or after
// resolution, never a torn mix. Run under -race.
func TestSetRequestAttributionRacesSpanStart(t *testing.T) {
	recorder := installAttributionRecorder(t)
	tracer := Tracer("attribution-test")

	const children = 16
	ctx := BeginRequestAttribution(context.Background(), "")

	var wg sync.WaitGroup
	for i := range children {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, child := tracer.Start(ctx, fmt.Sprintf("child-%02d", i))
			child.End()
		}(i)
	}
	SetRequestAttribution(ctx, "agent-7", AttributionSourceAgentIdentity)
	wg.Wait()

	if got := len(recorder.Ended()); got != children {
		t.Fatalf("ended span count = %d, want %d", got, children)
	}

	for _, span := range recorder.Ended() {
		id, source, present := endedAttribution(t, recorder, span.Name())
		before := id == DefaultAttributionID && source == AttributionSourceDefault
		after := id == "agent-7" && source == AttributionSourceAgentIdentity
		if !present || !(before || after) {
			t.Errorf("span %q attribution = (%q, %q, %v), want a whole pair from before or after resolution", span.Name(), id, source, present)
		}
	}
}

// Background work is not a request and must not be attributed: a default
// identity on a cache-refresh or metrics span would read as an unattributed
// request.
func TestBackgroundSpansCarryNoAttribution(t *testing.T) {
	recorder := installAttributionRecorder(t)
	tracer := Tracer("attribution-test")

	_, span := tracer.Start(context.Background(), "background")
	span.End()

	if got := len(recorder.Ended()); got != 1 {
		t.Fatalf("ended span count = %d, want 1", got)
	}

	for _, ended := range recorder.Ended() {
		for _, attr := range ended.Attributes() {
			if attr.Key == semconv.LLMDEPPFairnessIDKey || attr.Key == semconv.LLMDEPPFairnessSourceKey {
				t.Errorf("background span carries %s = %q", attr.Key, attr.Value.AsString())
			}
		}
	}
}

// Instrumentation that holds its own tracer from the global provider, such as
// otelhttp's middleware, is attributed by the span processor rather than by
// Tracer.
func TestSpanProcessorAttributesRawProviderSpans(t *testing.T) {
	recorder := installAttributionRecorder(t)

	ctx := BeginRequestAttribution(context.Background(), testFairnessID)
	_, span := otel.Tracer("raw-instrumentation").Start(ctx, "http-server")
	span.End()

	id, source, present := endedAttribution(t, recorder, "http-server")
	if !present {
		t.Fatal("span started outside Tracer carries no paired attribution")
	}
	if id != testFairnessID || source != AttributionSourceHeader {
		t.Errorf("attribution = (%q, %q), want (team-a, header)", id, source)
	}
}

// With tracing off every span is non-recording, and attribution must neither
// panic nor force the SDK's recording path.
func TestAttributionWithTracingDisabled(t *testing.T) {
	original := otel.GetTracerProvider()
	otel.SetTracerProvider(tracenoop.NewTracerProvider())
	t.Cleanup(func() { otel.SetTracerProvider(original) })

	ctx := BeginRequestAttribution(context.Background(), testFairnessID)
	ctx, span := Tracer("attribution-test").Start(ctx, "request")
	if span.IsRecording() {
		t.Fatal("noop provider returned a recording span")
	}
	span.End()

	if id, source, ok := RequestAttribution(ctx); !ok || id != testFairnessID || source != AttributionSourceHeader {
		t.Errorf("RequestAttribution() = (%q, %q, %v), want (team-a, header, true)", id, source, ok)
	}
}
