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

package server

import (
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
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
		// Classify bare ctx.Err() so cancel/deadline don't collapse to Unknown.
		code := status.Code(err)
		if code == codes.Unknown {
			code = status.FromContextError(err).Code()
		}
		metrics.ExtProcStreamFinished(code.String(), time.Since(start).Seconds())
	}()
	err = handler(srv, ss)
	return err
}
