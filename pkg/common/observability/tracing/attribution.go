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
	"sync"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
)

const (
	// DefaultAttributionID matches the default fairness identity used by scheduling.
	DefaultAttributionID = "default-flow"

	// AttributionSourceHeader means the fairness identity came from the request header.
	// It does not assert that the header producer was authenticated.
	AttributionSourceHeader = "header"
	// AttributionSourceAgentIdentity means the fairness identity came from agent identity.
	AttributionSourceAgentIdentity = "agent_identity"
	// AttributionSourceDefault means no fairness identity resolved.
	AttributionSourceDefault = "default"
)

// requestAttribution is request-local state shared by every span of one request.
type requestAttribution struct {
	mu     sync.RWMutex
	id     string
	source string
}

func (a *requestAttribution) load() (id, source string) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	return a.id, a.source
}

func (a *requestAttribution) store(id, source string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.id, a.source = id, source
}

// attributionKey avoids colliding with other packages' context keys.
type attributionKey struct{}

// normalizeAttribution maps an empty identity to the default sentinel and reports
// an unrecognised source as default, keeping the source attribute a closed set.
func normalizeAttribution(id, source string) (string, string) {
	if id == "" {
		return DefaultAttributionID, AttributionSourceDefault
	}

	switch source {
	case AttributionSourceHeader, AttributionSourceAgentIdentity, AttributionSourceDefault:
		return id, source
	default:
		return id, AttributionSourceDefault
	}
}

// BeginRequestAttribution attaches fresh per-request fairness attribution from
// the fairness header value. An empty value resolves to the default sentinel.
// Call once at the request entry point, before starting spans.
func BeginRequestAttribution(ctx context.Context, id string) context.Context {
	id, source := normalizeAttribution(id, AttributionSourceHeader)
	return context.WithValue(ctx, attributionKey{}, &requestAttribution{id: id, source: source})
}

// SetRequestAttribution replaces the attribution attached to ctx once the
// fairness identity resolves. Only the component that resolves it should call
// this. It is a no-op when ctx was never begun. Spans started afterwards pick
// up the new value; spans already open keep theirs until AttributeRequest.
func SetRequestAttribution(ctx context.Context, id, source string) {
	state := attributionFromContext(ctx)
	if state == nil {
		return
	}

	state.store(normalizeAttribution(id, source))
}

// RequestAttribution reports ok=false when ctx was never begun.
func RequestAttribution(ctx context.Context) (id, source string, ok bool) {
	state := attributionFromContext(ctx)
	if state == nil {
		return "", "", false
	}

	id, source = state.load()
	return id, source, true
}

func attributionFromContext(ctx context.Context) *requestAttribution {
	if ctx == nil {
		return nil
	}
	state, _ := ctx.Value(attributionKey{}).(*requestAttribution)
	return state
}

func attributionAttributes(id, source string) []attribute.KeyValue {
	return []attribute.KeyValue{
		semconv.LLMDEPPFairnessID(id),
		semconv.LLMDEPPFairnessSource(source),
	}
}

// AttributeRequest refreshes a span that was opened before fairness resolved.
func AttributeRequest(ctx context.Context, span trace.Span) {
	state := attributionFromContext(ctx)
	if state == nil || span == nil || !span.IsRecording() {
		return
	}

	id, source := state.load()
	span.SetAttributes(attributionAttributes(id, source)...)
}

// attributionSpanProcessor attributes every span started within an EPP request.
type attributionSpanProcessor struct{}

var _ sdktrace.SpanProcessor = attributionSpanProcessor{}

// NewRequestAttributionProcessor is installed by InitTracing and tests.
func NewRequestAttributionProcessor() sdktrace.SpanProcessor { return attributionSpanProcessor{} }

func (attributionSpanProcessor) OnStart(parent context.Context, span sdktrace.ReadWriteSpan) {
	state := attributionFromContext(parent)
	if state == nil || !span.IsRecording() {
		return
	}

	id, source := state.load()
	span.SetAttributes(attributionAttributes(id, source)...)
}

func (attributionSpanProcessor) OnEnd(sdktrace.ReadOnlySpan) {}

func (attributionSpanProcessor) Shutdown(context.Context) error { return nil }

func (attributionSpanProcessor) ForceFlush(context.Context) error { return nil }
