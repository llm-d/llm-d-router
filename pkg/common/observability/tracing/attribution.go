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

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
)

const (
	// DefaultAttributionID marks requests for which no tenant header was provided.
	DefaultAttributionID = "default-flow"

	// AttributionSourceHeader is the fairness-header resolution branch, not
	// producer authentication.
	AttributionSourceHeader = "header"
	// AttributionSourceDefault is used when no tenant header was provided.
	AttributionSourceDefault = "default"
)

// requestAttribution is immutable per-request state shared by every span of
// that request.
type requestAttribution struct {
	id     string
	source string
}

// attributionKey avoids colliding with other packages' context keys.
type attributionKey struct{}

// BeginRequestAttribution attaches fresh per-request tenant attribution. Only
// the fairness header is a tenant identity; an empty value is unattributed.
// Call once at the request entry point, before starting spans.
func BeginRequestAttribution(ctx context.Context, id string) context.Context {
	attribution := requestAttribution{
		id:     id,
		source: AttributionSourceHeader,
	}
	if id == "" {
		attribution.id = DefaultAttributionID
		attribution.source = AttributionSourceDefault
	}

	return context.WithValue(ctx, attributionKey{}, attribution)
}

// RequestAttribution reports ok=false when ctx was never begun, which is
// distinct from a request that resolved to default-flow.
func RequestAttribution(ctx context.Context) (id, source string, ok bool) {
	if ctx == nil {
		return "", "", false
	}

	attribution, ok := ctx.Value(attributionKey{}).(requestAttribution)
	if !ok {
		return "", "", false
	}

	return attribution.id, attribution.source, true
}

func attributionAttributes(id, source string) []attribute.KeyValue {
	return []attribute.KeyValue{
		semconv.LLMDRequestAttributionID(id),
		semconv.LLMDRequestAttributionSource(source),
	}
}

// attributionSpanProcessor attributes every span started within an EPP
// request, including spans started outside Tracer.
type attributionSpanProcessor struct{}

var _ sdktrace.SpanProcessor = attributionSpanProcessor{}

// NewRequestAttributionProcessor is installed by InitTracing and by tests that
// build their own provider.
func NewRequestAttributionProcessor() sdktrace.SpanProcessor { return attributionSpanProcessor{} }

func (attributionSpanProcessor) OnStart(parent context.Context, s sdktrace.ReadWriteSpan) {
	id, source, ok := RequestAttribution(parent)
	if !ok || !s.IsRecording() {
		return
	}

	s.SetAttributes(attributionAttributes(id, source)...)
}

func (attributionSpanProcessor) OnEnd(sdktrace.ReadOnlySpan) {}

func (attributionSpanProcessor) Shutdown(context.Context) error { return nil }

func (attributionSpanProcessor) ForceFlush(context.Context) error { return nil }
