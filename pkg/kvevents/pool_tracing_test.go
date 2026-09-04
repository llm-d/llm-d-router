package kvevents //nolint:testpackage // tests drive unexported processRawMessage

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// setupEventSpanRecorder installs an in-memory span recorder as the global
// tracer provider and returns it, restoring the previous provider on cleanup.
func setupEventSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	origTP := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(origTP) })
	return recorder
}

func findEventSpan(t *testing.T, recorder *tracetest.SpanRecorder, name string) sdktrace.ReadOnlySpan {
	t.Helper()
	for _, s := range recorder.Ended() {
		if s.Name() == name {
			return s
		}
	}
	t.Fatalf("no %s span recorded", name)
	return nil
}

// newTracingPool builds a pool with event tracing turned on, which is what
// every test in this file exercises. Config.Tracing is off by default.
func newTracingPool(t *testing.T) *Pool {
	t.Helper()

	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)
	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSizeTokens: 16,
		HashSeed:        "test",
	})
	require.NoError(t, err)

	cfg := DefaultConfig()
	cfg.Tracing = true
	return NewPool(cfg, idx, tp, nil)
}

func eventSpanAttrs(span sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	attrs := make(map[attribute.Key]attribute.Value)
	for _, kv := range span.Attributes() {
		attrs[kv.Key] = kv.Value
	}
	return attrs
}

type failingAdapter struct{}

func (a *failingAdapter) ParseMessage(*RawMessage) (string, string, EventBatch, error) {
	return "", "", EventBatch{}, errors.New("decode boom")
}

func (a *failingAdapter) ShardingKey(*RawMessage) string { return "" }

func TestProcessRawMessage_EmitsProcessAndDecodeSpans(t *testing.T) {
	recorder := setupEventSpanRecorder(t)
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool := newTracingPool(t)
	pool.adapter = &sourceEndpointAdapter{}

	pool.processRawMessage(ctx, &RawMessage{
		Topic:          "kv@10.0.0.1:8000@test-model",
		Payload:        []byte{1},
		SourceEndpoint: "10.0.0.1:8003",
	})

	process := findEventSpan(t, recorder, "events_process")
	pAttrs := eventSpanAttrs(process)
	assert.Equal(t, "kv@10.0.0.1:8000@test-model", pAttrs["llm_d.kv_cache.events.topic"].AsString())
	assert.Equal(t, int64(1), pAttrs["llm_d.kv_cache.events.payload_size_bytes"].AsInt64())
	assert.Equal(t, int64(1), pAttrs["llm_d.kv_cache.events.event_count"].AsInt64())
	// The subscriber's endpoint wins over the adapter-parsed one.
	assert.Equal(t, "10.0.0.1:8003", pAttrs["llm_d.kv_cache.events.pod_id"].AsString())

	decode := findEventSpan(t, recorder, "events_decode")
	dAttrs := eventSpanAttrs(decode)
	assert.Equal(t, "test-model", dAttrs["gen_ai.request.model"].AsString())
	// pod_id carries the effective pod and belongs on events_process alone, so
	// a query on it cannot return the pre-override value.
	assert.NotContains(t, dAttrs, attribute.Key("llm_d.kv_cache.events.pod_id"))
	assert.NotContains(t, dAttrs, attribute.Key("llm_d.kv_cache.events.event_count"))

	// decode nests under process
	assert.Equal(t, process.SpanContext().SpanID(), decode.Parent().SpanID())
}

// The message carries the receive span's identity across the worker queue, so
// processing must join that trace rather than start a detached one.
func TestProcessRawMessage_ParentsToReceiveSpan(t *testing.T) {
	recorder := setupEventSpanRecorder(t)
	pool := newTracingPool(t)
	pool.adapter = &sourceEndpointAdapter{}

	recvCtx, recvSpan := otel.Tracer("test").Start(context.Background(), "events_receive")
	carried := trace.SpanContextFromContext(recvCtx)
	recvSpan.End()

	// A fresh worker context, as the real worker uses.
	pool.processRawMessage(logging.NewTestLoggerIntoContext(context.Background()), &RawMessage{
		Topic:       "kv@10.0.0.1:8000@test-model",
		Payload:     []byte{1},
		SpanContext: &carried,
	})

	process := findEventSpan(t, recorder, "events_process")
	assert.Equal(t, carried.TraceID(), process.SpanContext().TraceID())
	assert.Equal(t, carried.SpanID(), process.Parent().SpanID())
	// The handoff crosses a worker queue, not a process boundary. A remote
	// parent selects a different ParentBased sampler branch.
	assert.False(t, process.Parent().IsRemote(), "queue handoff parent must be local")
}

// Without a carried span context the pipeline still traces, just as its own root.
func TestProcessRawMessage_RootsWhenNoReceiveSpan(t *testing.T) {
	recorder := setupEventSpanRecorder(t)
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool := newTracingPool(t)
	pool.adapter = &sourceEndpointAdapter{}

	pool.processRawMessage(ctx, &RawMessage{
		Topic:   "kv@10.0.0.1:8000@test-model",
		Payload: []byte{1},
	})

	process := findEventSpan(t, recorder, "events_process")
	assert.False(t, process.Parent().IsValid())
}

func TestProcessRawMessage_DecodeErrorMarksBothSpans(t *testing.T) {
	recorder := setupEventSpanRecorder(t)
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool := newTracingPool(t)
	pool.adapter = &failingAdapter{}

	pool.processRawMessage(ctx, &RawMessage{Topic: "kv@bad", Payload: []byte{1}})

	decode := findEventSpan(t, recorder, "events_decode")
	assert.Equal(t, codes.Error, decode.Status().Code)
	assert.Contains(t, decode.Status().Description, "decode boom")

	process := findEventSpan(t, recorder, "events_process")
	assert.Equal(t, codes.Error, process.Status().Code)
}

// A pool built the production way must emit the decode span, otherwise the
// deployed path decodes untraced however well the span works in isolation.
func TestNewPool_EmitsDecodeSpan(t *testing.T) {
	recorder := setupEventSpanRecorder(t)
	ctx := logging.NewTestLoggerIntoContext(context.Background())

	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSizeTokens: 16,
		HashSeed:        "test",
	})
	require.NoError(t, err)

	cfg := DefaultConfig()
	cfg.Tracing = true
	pool := NewPool(cfg, idx, tokenProcessor, &sourceEndpointAdapter{})
	pool.processRawMessage(ctx, &RawMessage{
		Topic:   "kv@10.0.0.1:8000@test-model",
		Payload: []byte{1},
	})

	// Fails the test if the decode span was not emitted.
	findEventSpan(t, recorder, "events_decode")
}

// The index write is the payoff: it must land inside the message's trace
// instead of appearing as an unattributable root span.
func TestProcessRawMessage_IndexSpansNestUnderProcess(t *testing.T) {
	recorder := setupEventSpanRecorder(t)
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool := newTracingPool(t)
	pool.adapter = &sourceEndpointAdapter{}
	pool.index = kvblock.NewTracedIndex(pool.index)

	pool.processRawMessage(ctx, &RawMessage{
		Topic:   "kv@10.0.0.1:8000@test-model",
		Payload: []byte{1},
	})

	process := findEventSpan(t, recorder, "events_process")
	add := findEventSpan(t, recorder, "index_add")
	require.Equal(t, process.SpanContext().TraceID(), add.SpanContext().TraceID())
	assert.Equal(t, process.SpanContext().SpanID(), add.Parent().SpanID())
}

// Event tracing is opt-in; see Config.Tracing for why.
func TestPool_NoEventSpansUnlessConfigured(t *testing.T) {
	recorder := setupEventSpanRecorder(t)
	ctx := logging.NewTestLoggerIntoContext(context.Background())

	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)
	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSizeTokens: 16, HashSeed: "test",
	})
	require.NoError(t, err)

	require.False(t, DefaultConfig().Tracing, "event tracing must default off")

	pool := NewPool(DefaultConfig(), idx, tp, &sourceEndpointAdapter{})
	z := newZMQSubscriber(pool, "pod-1", "", "tcp://x", "", "kv@", false)

	z.addTask(context.Background(), "kv@10.0.0.1:8000@test-model", 1, []byte{1})
	pool.processRawMessage(ctx, drainOne(t, pool))

	assert.Empty(t, recorder.Ended(), "no event spans when Config.Tracing is false")
}

// The default path must not pay for spans it never records.
func TestStartSpan_DisabledPathAllocatesNothing(t *testing.T) {
	pool := NewPool(DefaultConfig(), nil, nil, nil)
	require.Nil(t, pool.tracer, "event tracing must default off")

	ctx := context.Background()
	ctxPreserved := true
	allocs := testing.AllocsPerRun(1000, func() {
		spanCtx, span := pool.startSpan(ctx, "events_process", consumerSpanOptions)
		span.End()
		if spanCtx != ctx {
			ctxPreserved = false
		}
	})

	assert.True(t, ctxPreserved, "disabled path must return ctx untouched")
	assert.Zero(t, allocs, "disabled path must not allocate")
}

// BenchmarkPipelineSpans_Disabled measures the per-message span cost carried by
// the default configuration: one receive, one process, one decode.
func BenchmarkPipelineSpans_Disabled(b *testing.B) {
	pool := NewPool(DefaultConfig(), nil, nil, nil)
	ctx := context.Background()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, receive := pool.startSpan(ctx, "events_receive", consumerSpanOptions)
		_ = receive.SpanContext()
		receive.End()

		processCtx, process := pool.startSpan(ctx, "events_process", consumerSpanOptions)
		_, decode := pool.startSpan(processCtx, "events_decode", internalSpanOptions)
		decode.End()
		process.End()
	}
}
