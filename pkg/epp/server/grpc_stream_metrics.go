package server

import (
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"

	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// streamMetricsInterceptor records ext_proc stream count and hold duration.
func streamMetricsInterceptor(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	// Skip co-registered streams (health Watch, reflection).
	if info.FullMethod != extProcPb.ExternalProcessor_Process_FullMethodName {
		return handler(srv, ss)
	}
	metrics.ExtProcStreamStarted()
	start := time.Now()
	var err error
	// defer: grpc-go does not recover handler panics; keep the gauge balanced.
	defer func() {
		metrics.ExtProcStreamFinished(status.Code(err).String(), time.Since(start).Seconds())
	}()
	err = handler(srv, ss)
	return err
}
