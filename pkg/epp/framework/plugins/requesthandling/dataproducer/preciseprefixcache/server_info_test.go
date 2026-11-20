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

package preciseprefixcache

import (
	"context"
	"net"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
)

type fakeServerInfo struct {
	apiv1.UnimplementedSglangServiceServer
	resp *apiv1.GetServerInfoResponse
}

func (f *fakeServerInfo) GetServerInfo(context.Context, *apiv1.GetServerInfoRequest) (*apiv1.GetServerInfoResponse, error) {
	return f.resp, nil
}

func newServerInfoClient(t *testing.T, srv apiv1.SglangServiceServer) apiv1.SglangServiceClient {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	gs := grpc.NewServer()
	apiv1.RegisterSglangServiceServer(gs, srv)
	go func() { _ = gs.Serve(ln) }()
	t.Cleanup(gs.Stop)
	conn, err := grpc.NewClient(ln.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	return apiv1.NewSglangServiceClient(conn)
}

func serverInfoResponse(dpRank uint32, sources ...*apiv1.KvEventSource) *apiv1.GetServerInfoResponse {
	return &apiv1.GetServerInfoResponse{
		DeploymentInfo: &apiv1.DeploymentInfo{
			DpRank:         &dpRank,
			KvEventSources: sources,
		},
	}
}

// startFakeServerInfoPort starts a fake GetServerInfo server on a free local
// port and returns that port, for tests that wire GrpcBasePort.
func startFakeServerInfoPort(t *testing.T, resp *apiv1.GetServerInfoResponse) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	gs := grpc.NewServer()
	apiv1.RegisterSglangServiceServer(gs, &fakeServerInfo{resp: resp})
	go func() { _ = gs.Serve(ln) }()
	t.Cleanup(gs.Stop)
	return addrPort(t, ln.Addr())
}

// closedPort returns a local port with nothing listening, for the
// GetServerInfo failure path.
func closedPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer func() { _ = ln.Close() }()
	return addrPort(t, ln.Addr())
}

func addrPort(t *testing.T, addr net.Addr) int {
	t.Helper()
	_, portStr, err := net.SplitHostPort(addr.String())
	require.NoError(t, err)
	port, err := strconv.Atoi(portStr)
	require.NoError(t, err)
	return port
}

func TestReadServerInfo(t *testing.T) {
	for _, tc := range []struct {
		name          string
		resp          *apiv1.GetServerInfoResponse
		wantDpRank    uint32
		wantPublishes bool
	}{
		{
			name:          "own source advertised",
			resp:          serverInfoResponse(3, &apiv1.KvEventSource{DpRank: 3, Endpoint: "tcp://127.0.0.1:5559"}),
			wantDpRank:    3,
			wantPublishes: true,
		},
		{
			name:          "own source picked among other ranks",
			resp:          serverInfoResponse(3, &apiv1.KvEventSource{DpRank: 2, Endpoint: "tcp://127.0.0.1:5558"}, &apiv1.KvEventSource{DpRank: 3, Endpoint: "tcp://127.0.0.1:5559"}),
			wantDpRank:    3,
			wantPublishes: true,
		},
		{
			name:          "sources hold another rank only",
			resp:          serverInfoResponse(3, &apiv1.KvEventSource{DpRank: 2, Endpoint: "tcp://127.0.0.1:5558"}),
			wantDpRank:    3,
			wantPublishes: false,
		},
		{
			name:          "no sources",
			resp:          serverInfoResponse(3),
			wantDpRank:    3,
			wantPublishes: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := newServerInfoClient(t, &fakeServerInfo{resp: tc.resp})
			dpRank, publishes, err := readServerInfo(context.Background(), client)
			require.NoError(t, err)
			assert.Equal(t, tc.wantDpRank, dpRank)
			assert.Equal(t, tc.wantPublishes, publishes)
		})
	}
}

func TestReadServerInfoMissingRank(t *testing.T) {
	for _, tc := range []struct {
		name string
		resp *apiv1.GetServerInfoResponse
	}{
		{
			name: "no deployment info",
			resp: &apiv1.GetServerInfoResponse{},
		},
		{
			name: "deployment info without dp rank",
			resp: &apiv1.GetServerInfoResponse{
				DeploymentInfo: &apiv1.DeploymentInfo{},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := newServerInfoClient(t, &fakeServerInfo{resp: tc.resp})
			_, _, err := readServerInfo(context.Background(), client)
			require.Error(t, err)
		})
	}
}

func TestReadServerInfoUnreachable(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := ln.Addr().String()
	require.NoError(t, ln.Close())

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })

	_, _, err = readServerInfo(context.Background(), apiv1.NewSglangServiceClient(conn))
	require.Error(t, err)
}
