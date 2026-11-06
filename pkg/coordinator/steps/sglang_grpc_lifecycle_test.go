/*
Copyright 2026 The llm-d Authors.

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

package steps

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// sglangPDFixture runs the prefill and decode steps against one h2c gateway
// whose prefill and decode handlers the test supplies. It drives the steps
// directly, so the request context outlives the steps: the test, not
// net/http's handler teardown, decides when it ends.
type sglangPDFixture struct {
	prefill         *PrefillStep
	decode          *DecodeStep
	decodeCommitted chan struct{}
}

type commitNotifyRecorder struct {
	*httptest.ResponseRecorder
	committed chan struct{}
	once      sync.Once
}

func (r *commitNotifyRecorder) Write(b []byte) (int, error) {
	n, err := r.ResponseRecorder.Write(b)
	if r.committed != nil {
		r.once.Do(func() { close(r.committed) })
	}
	return n, err
}

// releaseOnCleanup closes stop when the test ends. Register it after the
// fixture: cleanups run last in, first out, so stop closes before the test
// server closes, and a handler that lingers on purpose cannot hang srv.Close
// when an assertion fails.
func releaseOnCleanup(t *testing.T, stop chan struct{}) {
	t.Helper()
	t.Cleanup(func() { close(stop) })
}

func newSGLangPDFixture(t *testing.T, prefill, decode http.HandlerFunc) *sglangPDFixture {
	t.Helper()
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
		case gateway.PhasePrefill:
			prefill(w, r)
		case gateway.PhaseDecode:
			decode(w, r)
		default:
			w.WriteHeader(http.StatusBadRequest)
		}
	}))
	protocols := new(http.Protocols)
	protocols.SetHTTP1(true)
	protocols.SetUnencryptedHTTP2(true)
	srv.Config.Protocols = protocols
	srv.Start()
	t.Cleanup(srv.Close)

	gw := gateway.New(config.GatewayConfig{Address: srv.URL})
	p, err := NewPrefillStep(gw, map[string]any{ParamKVConnector: kv.SGLang})
	require.NoError(t, err)
	d, err := NewDecodeStep(gw, map[string]any{ParamKVConnector: kv.SGLang})
	require.NoError(t, err)
	return &sglangPDFixture{prefill: p.(*PrefillStep), decode: d.(*DecodeStep)}
}

func (f *sglangPDFixture) run(ctx context.Context, t *testing.T) (*httptest.ResponseRecorder, error) {
	t.Helper()
	payload, err := grpcframe.MarshalProto(&sglangapiv1.GenerateRequest{
		Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hi"}},
	})
	require.NoError(t, err)
	rec := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:           "req-pd",
		OriginalHeaders:     http.Header{},
		OriginalBody:        payload,
		GRPCGeneratePayload: payload[grpcframe.HeaderLen:],
		Stream:              true,
		KVTransferParams:    map[string]any{},
		ResponseWriter:      &commitNotifyRecorder{ResponseRecorder: rec, committed: f.decodeCommitted},
	}
	if err := f.prefill.Execute(ctx, reqCtx); err != nil {
		return rec, err
	}
	return rec, f.decode.Execute(ctx, reqCtx)
}

func writePrefillHeaders(w http.ResponseWriter) {
	w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
	w.Header().Set(routing.PrefillEndpointHeader, "10.2.3.4:30000")
	w.WriteHeader(http.StatusOK)
	_ = http.NewResponseController(w).Flush()
}

func writeDecodeOK(t *testing.T, w http.ResponseWriter) {
	t.Helper()
	w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
	w.WriteHeader(http.StatusOK)
	frame, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
		Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{Text: "ok"}},
	})
	require.NoError(t, err)
	_, _ = w.Write(frame)
	_ = http.NewResponseController(w).Flush()
	w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
}

// After decode succeeds, a prefill stream that keeps running is cancelled by
// the grace timer, not by the request context, which is detached by then.
func TestSGLangGRPC_PrefillGraceTimerCancelsLingeringPrefill(t *testing.T) {
	prev := sglangPrefillGrace
	sglangPrefillGrace = 200 * time.Millisecond
	t.Cleanup(func() { sglangPrefillGrace = prev })
	stop := make(chan struct{})

	prefillDone := make(chan struct{})
	var prefillCancelledAt atomic.Int64
	f := newSGLangPDFixture(t,
		func(w http.ResponseWriter, r *http.Request) {
			writePrefillHeaders(w)
			select { // linger until the coordinator cancels
			case <-r.Context().Done():
				prefillCancelledAt.Store(time.Now().UnixNano())
				close(prefillDone)
			case <-stop:
			}
		},
		func(w http.ResponseWriter, _ *http.Request) { writeDecodeOK(t, w) },
	)
	releaseOnCleanup(t, stop)

	// The request context stays alive for the whole test, so only the grace
	// timer can end the prefill stream.
	start := time.Now()
	_, err := f.run(context.Background(), t)
	require.NoError(t, err)

	select {
	case <-prefillDone:
	case <-time.After(5 * time.Second):
		t.Fatal("prefill stream was not cancelled after decode succeeded")
	}
	elapsed := time.Duration(prefillCancelledAt.Load() - start.UnixNano())
	require.GreaterOrEqual(t, elapsed, sglangPrefillGrace, "prefill must get the grace period")
}

// After decode fails, the decode step itself cancels prefill at once. The
// request context stays alive, so the AfterFunc backup cannot be what fires.
func TestSGLangGRPC_DecodeFailureCancelsPrefillWithoutRequestContext(t *testing.T) {
	prev := sglangPrefillGrace
	sglangPrefillGrace = time.Hour // the grace timer must not be what fires either
	t.Cleanup(func() { sglangPrefillGrace = prev })
	stop := make(chan struct{})

	prefillDone := make(chan struct{})
	f := newSGLangPDFixture(t,
		func(w http.ResponseWriter, r *http.Request) {
			writePrefillHeaders(w)
			select {
			case <-r.Context().Done():
				close(prefillDone)
			case <-stop:
			}
		},
		func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) },
	)
	releaseOnCleanup(t, stop)

	_, err := f.run(context.Background(), t)
	require.Error(t, err)

	select {
	case <-prefillDone:
	case <-time.After(5 * time.Second):
		t.Fatal("prefill stream was not cancelled after decode failed")
	}
}

// A prefill failure that arrives as decode reaches end of stream is reported
// to the client in place of decode's OK trailer.
func TestSGLangGRPC_PrefillFailureAfterDecodeEOFIsReported(t *testing.T) {
	prev := sglangPrefillWaitOnTerminal
	sglangPrefillWaitOnTerminal = 5 * time.Second
	t.Cleanup(func() { sglangPrefillWaitOnTerminal = prev })

	decodeCommitted := make(chan struct{})
	f := newSGLangPDFixture(t,
		func(w http.ResponseWriter, r *http.Request) {
			writePrefillHeaders(w)
			select {
			case <-decodeCommitted:
			case <-r.Context().Done():
				return
			}
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, "14")
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderMessage, "prefill unavailable")
		},
		func(w http.ResponseWriter, _ *http.Request) {
			writeDecodeOK(t, w)
		},
	)
	f.decodeCommitted = decodeCommitted

	rec, err := f.run(context.Background(), t)
	require.Error(t, err)
	var streamed *pipeline.UpstreamStreamedError
	require.ErrorAs(t, err, &streamed)
	require.Equal(t, PrefillStepName, streamed.Step)
	// Decode committed its headers and frame, so the failure travels as a trailer.
	require.Equal(t, "14", rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderStatus))
	require.Equal(t, "prefill unavailable", rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderMessage))
}

// A prefill stream that ends inside a frame is a failure, not a clean end,
// even when the upstream closes it with an OK trailer.
func TestSGLangGRPC_PrefillStreamTruncatedInsideFrameFails(t *testing.T) {
	prev := sglangPrefillWaitOnTerminal
	sglangPrefillWaitOnTerminal = 5 * time.Second
	t.Cleanup(func() { sglangPrefillWaitOnTerminal = prev })

	decodeCommitted := make(chan struct{})
	f := newSGLangPDFixture(t,
		func(w http.ResponseWriter, r *http.Request) {
			writePrefillHeaders(w)
			select {
			case <-decodeCommitted:
			case <-r.Context().Done():
				return
			}
			// A frame header that declares 10 payload bytes, then nothing.
			_, _ = w.Write([]byte{0, 0, 0, 0, 10})
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
		},
		func(w http.ResponseWriter, _ *http.Request) {
			writeDecodeOK(t, w)
		},
	)
	f.decodeCommitted = decodeCommitted

	rec, err := f.run(context.Background(), t)
	require.Error(t, err)
	var streamed *pipeline.UpstreamStreamedError
	require.ErrorAs(t, err, &streamed)
	require.Equal(t, PrefillStepName, streamed.Step)
	require.Equal(t, "14", rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderStatus))
	require.Contains(t, rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderMessage), "prefill stream interrupted")
}
