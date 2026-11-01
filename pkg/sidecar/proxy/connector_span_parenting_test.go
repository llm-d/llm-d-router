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

package proxy

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"github.com/llm-d/llm-d-router/test/sidecar/mock"
)

// recordSpans installs a recording tracer provider and the W3C propagator for
// the duration of the test.
func recordSpans(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()

	sr := tracetest.NewSpanRecorder()
	useTracerProviderForTest(t, sdktrace.NewTracerProvider(
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
		sdktrace.WithSpanProcessor(sr),
	))
	prev := otel.GetTextMapPropagator()
	otel.SetTextMapPropagator(propagation.TraceContext{})
	t.Cleanup(func() { otel.SetTextMapPropagator(prev) })
	return sr
}

// spanTree indexes ended spans by name and by span ID. Names may repeat (the
// otelhttp client spans do); span() is only used for names that must not.
type spanTree struct {
	byName map[string][]sdktrace.ReadOnlySpan
	byID   map[trace.SpanID]sdktrace.ReadOnlySpan
}

func newSpanTree(t *testing.T, spans []sdktrace.ReadOnlySpan) spanTree {
	t.Helper()

	tree := spanTree{byName: map[string][]sdktrace.ReadOnlySpan{}, byID: map[trace.SpanID]sdktrace.ReadOnlySpan{}}
	for _, s := range spans {
		tree.byName[s.Name()] = append(tree.byName[s.Name()], s)
		tree.byID[s.SpanContext().SpanID()] = s
	}
	return tree
}

func (tree spanTree) span(t *testing.T, name string) sdktrace.ReadOnlySpan {
	t.Helper()

	spans := tree.byName[name]
	require.Len(t, spans, 1, "want exactly one %q span", name)
	return spans[0]
}

// hasAncestor reports whether the span named ancestor is on the parent chain of
// the span named name.
func (tree spanTree) hasAncestor(t *testing.T, name, ancestor string) bool {
	t.Helper()

	want := tree.span(t, ancestor).SpanContext().SpanID()
	for s := tree.span(t, name); s.Parent().IsValid(); {
		if s.Parent().SpanID() == want {
			return true
		}
		parent, ok := tree.byID[s.Parent().SpanID()]
		if !ok {
			return false
		}
		s = parent
	}
	return false
}

// assertSiblingStages checks that prefill and decode are both children of
// forward_request, and that each backend request hangs under its own stage.
func assertSiblingStages(t *testing.T, tree spanTree) {
	t.Helper()

	forward := tree.span(t, "forward_request").SpanContext().SpanID()
	require.Equal(t, forward, tree.span(t, "prefill").Parent().SpanID(), "prefill must be a child of forward_request")
	require.Equal(t, forward, tree.span(t, "decode").Parent().SpanID(), "decode must be a child of forward_request, not of prefill")

	require.True(t, tree.hasAncestor(t, "prefill-backend", "prefill"), "prefill backend request must hang under prefill")
	require.False(t, tree.hasAncestor(t, "prefill-backend", "decode"), "prefill backend request must not hang under decode")
	require.True(t, tree.hasAncestor(t, "decode-backend", "decode"), "decode backend request must hang under decode")
	require.False(t, tree.hasAncestor(t, "decode-backend", "prefill"), "decode backend request must not hang under prefill")
}

func hasAttribute(s sdktrace.ReadOnlySpan, key string) bool {
	for _, kv := range s.Attributes() {
		if string(kv.Key) == key {
			return true
		}
	}
	return false
}

// tracedBackend serves h behind an otelhttp server span named name, standing in
// for a vLLM engine that continues the incoming trace.
func tracedBackend(t *testing.T, name string, h http.Handler) *url.URL {
	t.Helper()

	srv := httptest.NewServer(otelhttp.NewHandler(h, name,
		otelhttp.WithSpanNameFormatter(func(operation string, _ *http.Request) string { return operation })))
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	require.NoError(t, err)
	return u
}

// requestWithForwardSpan returns a request whose context carries an open
// forward_request span, as disaggregatedPrefillHandler would hand it on.
func requestWithForwardSpan(t *testing.T, path string, body []byte) (*http.Request, trace.Span) {
	t.Helper()

	ctx, span := otel.Tracer("test").Start(t.Context(), "forward_request")
	return httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body)).WithContext(ctx), span
}

func TestNIXLV2StagesAreSiblingSpans(t *testing.T) {
	sr := recordSpans(t)

	prefillURL := tracedBackend(t, "prefill-backend",
		&mock.ChatCompletionHandler{Connector: constants.KVConnectorNIXLV2, Role: mock.RolePrefill})
	decodeURL := tracedBackend(t, "decode-backend",
		&mock.ChatCompletionHandler{Connector: constants.KVConnectorNIXLV2, Role: mock.RoleDecode})

	s := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: constants.KVConnectorNIXLV2})
	s.decoderProxy = s.createDecoderProxyHandler(decodeURL, false)

	req, forward := requestWithForwardSpan(t, reqcommon.PathChatCompletions, []byte(chatCompletionsRequestBody))
	rec := httptest.NewRecorder()
	s.handleNIXLV2(rec, req, prefillURL.Host, "", reqcommon.APITypeChatCompletions)
	forward.End()
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())

	tree := newSpanTree(t, sr.Ended())
	assertSiblingStages(t, tree)

	// The end-to-end summary stays on the decode span.
	decode := tree.span(t, "decode")
	for _, key := range []string{
		"llm_d.pd_proxy.true_ttft_ms",
		"llm_d.pd_proxy.total_duration_ms",
		"llm_d.pd_proxy.coordinator_overhead_ms",
	} {
		require.True(t, hasAttribute(decode, key), "decode span missing %s", key)
	}
}

func TestConcurrentPDStagesAreSiblingSpans(t *testing.T) {
	sr := recordSpans(t)

	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusOK)
	})
	prefillURL := tracedBackend(t, "prefill-backend", ok)
	decodeURL := tracedBackend(t, "decode-backend", ok)

	s := NewProxy(Config{Port: "0", DecoderURL: decodeURL})
	s.decoderProxy = s.createDecoderProxyHandler(decodeURL, false)

	body := []byte(`{"input_ids":[1]}`)
	req, forward := requestWithForwardSpan(t, reqcommon.PathSGLangGenerate, body)
	s.runConcurrentPD(httptest.NewRecorder(), req, body, body, prefillURL.Host, constants.KVConnectorSGLang, nil, nil)
	forward.End()

	// Prefill runs in a goroutine; wait for its span and its backend span.
	require.Eventually(t, func() bool {
		var prefillDone, backendDone bool
		for _, s := range sr.Ended() {
			prefillDone = prefillDone || s.Name() == "prefill"
			backendDone = backendDone || s.Name() == "prefill-backend"
		}
		return prefillDone && backendDone
	}, 2*time.Second, 10*time.Millisecond)

	tree := newSpanTree(t, sr.Ended())
	assertSiblingStages(t, tree)
	require.True(t, hasAttribute(tree.span(t, "decode"), "llm_d.pd_proxy.true_ttft_ms"), "decode span missing true_ttft_ms")
}
