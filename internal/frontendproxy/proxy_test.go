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

package frontendproxy

import (
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
)

// A trailers-only gRPC error must reach the client even when no response body is sent.
func TestGRPCTrailersOnly(t *testing.T) {
	upstream := grpctest.StartGRPCServer(t, func(s *grpc.Server) {
		healthpb.RegisterHealthServer(s, health.NewServer())
	})
	proxy := grpctest.NewH2CServer(t, NewGRPC(upstream, 1024))
	client := healthpb.NewHealthClient(grpctest.Dial(t, proxy.Listener.Addr().String()))
	_, err := client.Check(t.Context(), &healthpb.HealthCheckRequest{Service: "unknown"})
	require.Equal(t, codes.NotFound, status.Code(err))
}
