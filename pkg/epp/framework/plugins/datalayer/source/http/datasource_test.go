package http

import (
	"context"
	"errors"
	"io"
	"net/url"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

type fakeClient struct {
	value any
	err   error
}

func (c *fakeClient) Get(_ context.Context, _ *url.URL, _ Addressable,
	parser func(io.Reader) (any, error)) (any, error) {
	if c.err != nil {
		return nil, c.err
	}
	if c.value != nil {
		return c.value, nil
	}
	return parser(nil)
}

type stubExtractor struct {
	extType string
	err     error
	block   time.Duration

	mu        sync.Mutex
	callCount int
	lastCtx   context.Context
}

func newStubExtractor(extType string) *stubExtractor {
	return &stubExtractor{extType: extType}
}

func (e *stubExtractor) withError(err error) *stubExtractor { e.err = err; return e }
func (e *stubExtractor) withBlock(d time.Duration) *stubExtractor {
	e.block = d
	return e
}

func (e *stubExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: e.extType, Name: e.extType}
}

func (e *stubExtractor) Extract(ctx context.Context, _ fwkdl.PollInput[int]) error {
	e.mu.Lock()
	e.callCount++
	e.lastCtx = ctx
	e.mu.Unlock()
	if e.block > 0 {
		select {
		case <-time.After(e.block):
		case <-ctx.Done():
		}
	}
	return e.err
}

func (e *stubExtractor) calls() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.callCount
}

// wrongTypedExtractor satisfies plugin.Plugin but not Extractor[PollInput[int]].
type wrongTypedExtractor struct{}

func (wrongTypedExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "wrong", Name: "wrong"}
}

func newDS(t *testing.T, srcType string, fc *fakeClient) *HTTPDataSource[int] {
	t.Helper()
	return &HTTPDataSource[int]{
		typedName: fwkplugin.TypedName{Type: srcType, Name: srcType},
		scheme:    "http",
		path:      "/test",
		client:    fc,
		parser:    func(_ io.Reader) (int, error) { return 0, nil },
	}
}

func newTestEndpoint() fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{MetricsHost: "10.0.0.1:8000"}, fwkdl.NewMetrics())
}

func extDelta(t *testing.T, srcType, extType string, before float64) float64 {
	t.Helper()
	return testutil.ToFloat64(metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(srcType, extType)) - before
}

func extBefore(t *testing.T, srcType, extType string) float64 {
	t.Helper()
	return testutil.ToFloat64(metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(srcType, extType))
}

func TestDispatch_PollFailure_ReturnsError(t *testing.T) {
	wantErr := errors.New("upstream offline")
	s := newDS(t, "src-poll-fail", &fakeClient{err: wantErr})
	require.NoError(t, s.AppendExtractor(newStubExtractor("ext")))

	err := s.Dispatch(context.Background(), newTestEndpoint())
	require.ErrorIs(t, err, wantErr, "poll-level failure must surface as the returned error")
}

func TestDispatch_ExtractorFailure_ReturnsNil(t *testing.T) {
	const src, ext = "src-ext-fail", "ext-fail"
	s := newDS(t, src, &fakeClient{value: 42})
	require.NoError(t, s.AppendExtractor(newStubExtractor(ext).withError(errors.New("bad data"))))

	before := extBefore(t, src, ext)
	require.NoError(t, s.Dispatch(context.Background(), newTestEndpoint()),
		"extractor failure must not surface as a returned error")
	assert.Equal(t, float64(1), extDelta(t, src, ext, before),
		"extractor failure must record one DataLayerExtractErrorsTotal increment")
}

func TestDispatch_ExtractorFailure_LabelsBothSrcAndExt(t *testing.T) {
	const src, ext = "src-label", "ext-label"
	s := newDS(t, src, &fakeClient{value: 1})
	require.NoError(t, s.AppendExtractor(newStubExtractor(ext).withError(errors.New("x"))))

	before := extBefore(t, src, ext)
	require.NoError(t, s.Dispatch(context.Background(), newTestEndpoint()))
	assert.Equal(t, float64(1), extDelta(t, src, ext, before),
		"counter must be incremented under (src=%q, ext=%q); any other label tuple is a regression",
		src, ext)
	assert.Zero(t, testutil.ToFloat64(metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(src, src)),
		"counter must not appear under (src, src)")
	assert.Zero(t, testutil.ToFloat64(metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(ext, ext)),
		"counter must not appear under (ext, ext)")
}

func TestDispatch_MultipleExtractorFailures_BothRecorded(t *testing.T) {
	const src, extA, extB = "src-multi", "ext-a", "ext-b"
	s := newDS(t, src, &fakeClient{value: 1})
	require.NoError(t, s.AppendExtractor(newStubExtractor(extA).withError(errors.New("a"))))
	require.NoError(t, s.AppendExtractor(newStubExtractor(extB).withError(errors.New("b"))))

	beforeA := extBefore(t, src, extA)
	beforeB := extBefore(t, src, extB)
	require.NoError(t, s.Dispatch(context.Background(), newTestEndpoint()))

	assert.Equal(t, float64(1), extDelta(t, src, extA, beforeA),
		"each failing extractor must record its own increment")
	assert.Equal(t, float64(1), extDelta(t, src, extB, beforeB))
}

func TestDispatch_SlowExtractor_DoesNotStarveNext(t *testing.T) {
	slow := newStubExtractor("ext-slow").
		withBlock(2 * defaultStepTimeout).
		withError(context.DeadlineExceeded)
	fast := newStubExtractor("ext-fast")

	s := newDS(t, "src-isolation", &fakeClient{value: 1})
	require.NoError(t, s.AppendExtractor(slow))
	require.NoError(t, s.AppendExtractor(fast))

	start := time.Now()
	require.NoError(t, s.Dispatch(context.Background(), newTestEndpoint()))
	elapsed := time.Since(start)

	assert.Equal(t, 1, fast.calls(), "fast extractor must still run after slow extractor timed out")
	assert.Less(t, elapsed, 2*defaultStepTimeout+500*time.Millisecond,
		"total Dispatch time must be ~one step-timeout (slow) + epsilon (fast); got %s", elapsed)
}

func TestDispatch_ParentCtxCancelled_StopsRemainingExtractors(t *testing.T) {
	first := newStubExtractor("ext-first")
	second := newStubExtractor("ext-second")
	third := newStubExtractor("ext-third")

	ctx, cancel := context.WithCancel(context.Background())
	firstWithSideEffect := &cancelOnExtract{stub: first, cancel: cancel}

	s := newDS(t, "src-cancel", &fakeClient{value: 1})
	require.NoError(t, s.AppendExtractor(firstWithSideEffect))
	require.NoError(t, s.AppendExtractor(second))
	require.NoError(t, s.AppendExtractor(third))

	require.NoError(t, s.Dispatch(ctx, newTestEndpoint()))

	assert.Equal(t, 1, firstWithSideEffect.stub.calls(), "first extractor runs before cancel observed")
	assert.Equal(t, 0, second.calls(), "second extractor must be skipped once parent ctx is cancelled")
	assert.Equal(t, 0, third.calls(), "third extractor must be skipped once parent ctx is cancelled")
}

type cancelOnExtract struct {
	stub   *stubExtractor
	cancel context.CancelFunc
}

func (c *cancelOnExtract) TypedName() fwkplugin.TypedName { return c.stub.TypedName() }
func (c *cancelOnExtract) Extract(ctx context.Context, in fwkdl.PollInput[int]) error {
	err := c.stub.Extract(ctx, in)
	c.cancel()
	return err
}

func TestAppendExtractor(t *testing.T) {
	cases := []struct {
		name      string
		preBound  []fwkplugin.Plugin
		appending fwkplugin.Plugin
		wantErr   error
		wantCount int
	}{
		{
			name:      "valid first append",
			appending: newStubExtractor("ext-a"),
			wantCount: 1,
		},
		{
			name:      "different type appends successfully",
			preBound:  []fwkplugin.Plugin{newStubExtractor("ext-a")},
			appending: newStubExtractor("ext-b"),
			wantCount: 2,
		},
		{
			name:      "wrong type returns error",
			appending: wrongTypedExtractor{},
			wantErr:   ErrExtractorTypeMismatch,
			wantCount: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := newDS(t, "src", &fakeClient{value: 1})
			for _, ext := range tc.preBound {
				require.NoError(t, s.AppendExtractor(ext), "pre-bind setup must succeed")
			}
			err := s.AppendExtractor(tc.appending)
			if tc.wantErr != nil {
				require.ErrorIs(t, err, tc.wantErr)
			} else {
				require.NoError(t, err)
			}
			s.mu.RLock()
			defer s.mu.RUnlock()
			assert.Len(t, s.exts, tc.wantCount)
		})
	}
}

func TestAppendExtractor_PreservesInsertionOrder(t *testing.T) {
	s := newDS(t, "src-order", &fakeClient{value: 1})
	calls := make([]string, 0, 3)
	var mu sync.Mutex
	for _, name := range []string{"a", "b", "c"} {
		wrapped := &orderedStub{stub: newStubExtractor(name), name: name, log: &calls, mu: &mu}
		require.NoError(t, s.AppendExtractor(wrapped))
	}
	require.NoError(t, s.Dispatch(context.Background(), newTestEndpoint()))

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, []string{"a", "b", "c"}, calls,
		"contract: extractors run in AppendExtractor-insertion order")
}

type orderedStub struct {
	stub *stubExtractor
	name string
	log  *[]string
	mu   *sync.Mutex
}

func (o *orderedStub) TypedName() fwkplugin.TypedName { return o.stub.TypedName() }
func (o *orderedStub) Extract(ctx context.Context, in fwkdl.PollInput[int]) error {
	o.mu.Lock()
	*o.log = append(*o.log, o.name)
	o.mu.Unlock()
	return o.stub.Extract(ctx, in)
}
