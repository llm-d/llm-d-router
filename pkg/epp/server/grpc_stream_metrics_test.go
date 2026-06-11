package server

import (
	"context"
	"errors"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

type fakeServerStream struct{ ctx context.Context }

func (f *fakeServerStream) SetHeader(metadata.MD) error  { return nil }
func (f *fakeServerStream) SendHeader(metadata.MD) error { return nil }
func (f *fakeServerStream) SetTrailer(metadata.MD)       {}
func (f *fakeServerStream) Context() context.Context     { return f.ctx }
func (f *fakeServerStream) SendMsg(any) error            { return nil }
func (f *fakeServerStream) RecvMsg(any) error            { return nil }

func invoke(t *testing.T, method string, handlerErr error) error {
	t.Helper()
	called := false
	handler := func(_ any, _ grpc.ServerStream) error {
		called = true
		return handlerErr
	}
	err := streamMetricsInterceptor(nil, &fakeServerStream{ctx: context.Background()},
		&grpc.StreamServerInfo{FullMethod: method}, handler)
	require.True(t, called, "handler must be invoked")
	return err
}

// Handler error propagates verbatim and the code maps.
func TestStreamMetricsInterceptor_Propagation(t *testing.T) {
	tests := []struct {
		name string
		in   error
		code codes.Code
	}{
		{"ok", nil, codes.OK},
		{"status error", status.Error(codes.Internal, "boom"), codes.Internal},
		{"plain error", errors.New("x"), codes.Unknown},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := invoke(t, extProcPb.ExternalProcessor_Process_FullMethodName, tc.in)
			require.Equal(t, tc.in, err, "handler error must propagate verbatim")
			require.Equal(t, tc.code, status.Code(err))
		})
	}
}

// Only the ext_proc Process stream is recorded; health Watch passes through.
func TestStreamMetricsInterceptor_MethodScope(t *testing.T) {
	metrics.RegisterGRPCStreamMetrics()
	const name = "llm_d_router_epp_extproc_streams_total"

	before, err := promtestutil.GatherAndCount(ctrlmetrics.Registry, name)
	require.NoError(t, err)

	// Non-ext_proc method: served, not recorded.
	require.NoError(t, invoke(t, "/grpc.health.v1.Health/Watch", nil))
	mid, err := promtestutil.GatherAndCount(ctrlmetrics.Registry, name)
	require.NoError(t, err)
	require.Equal(t, before, mid, "health Watch must not be counted as an ext_proc stream")

	// ext_proc stream with a code unused elsewhere → adds one series.
	_ = invoke(t, extProcPb.ExternalProcessor_Process_FullMethodName, status.Error(codes.ResourceExhausted, "x"))
	after, err := promtestutil.GatherAndCount(ctrlmetrics.Registry, name)
	require.NoError(t, err)
	require.Equal(t, mid+1, after, "ext_proc Process stream must add a new completion series")
}
