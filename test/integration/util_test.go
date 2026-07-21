/*
Copyright 2025 The Kubernetes Authors.

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

package integration

import (
	"context"
	"fmt"
	"net"
	"os"
	"syscall"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// stubExtProc answers a single Process message, which is enough to prove the server is
// serving rather than merely bound.
type stubExtProc struct {
	extProcPb.UnimplementedExternalProcessorServer
}

func (stubExtProc) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	if _, err := srv.Recv(); err != nil {
		return err
	}
	return srv.Send(&extProcPb.ProcessingResponse{})
}

// serveStub starts a stub ext-proc server on lis and stops it at test end.
func serveStub(t *testing.T, lis net.Listener) {
	t.Helper()
	srv := grpc.NewServer()
	extProcPb.RegisterExternalProcessorServer(srv, stubExtProc{})
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
}

func dialLocal(t *testing.T, port int) *grpc.ClientConn {
	t.Helper()
	conn, err := grpc.NewClient(fmt.Sprintf("127.0.0.1:%d", port),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

// ephemeralDraws is the number of ephemeral ports drawn while a reservation is held.
// Large enough to sample a meaningful slice of the ephemeral range, small enough to
// stay well under a second.
const ephemeralDraws = 2000

// TestReserveListenerDefendsPort asserts that a reserved listener's port cannot be
// reassigned while it is held: an explicit bind of it fails with EADDRINUSE, and the
// kernel never hands it back as an ephemeral allocation. This is the property the
// harness relies on to start a server on a known port without a bind race.
func TestReserveListenerDefendsPort(t *testing.T) {
	lis, err := testutils.ReserveListener()
	require.NoError(t, err)
	t.Cleanup(func() { _ = lis.Close() })
	reserved := lis.Addr().(*net.TCPAddr).Port

	_, err = net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", reserved))
	require.Error(t, err)
	assert.ErrorIs(t, err, syscall.EADDRINUSE)

	for i := range ephemeralDraws {
		drawn, err := net.Listen("tcp", "127.0.0.1:0")
		require.NoError(t, err)
		port := drawn.Addr().(*net.TCPAddr).Port
		require.NoError(t, drawn.Close())
		if port == reserved {
			t.Fatalf("ephemeral draw %d returned reserved port %d", i, reserved)
		}
	}
}

// TestPreBoundListenerQueuesDials asserts that a TCP dial to a bound but unserved
// listener succeeds, because the kernel accepts into the listen backlog. Only Ready
// proves the server is serving.
func TestPreBoundListenerQueuesDials(t *testing.T) {
	lis, err := testutils.ReserveListener()
	require.NoError(t, err)
	port := lis.Addr().(*net.TCPAddr).Port

	raw, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), time.Second)
	require.NoError(t, err, "TCP dial must succeed before Serve runs")
	require.NoError(t, raw.Close())

	serveStub(t, lis)

	conn := dialLocal(t, port)
	require.NoError(t, WaitExtProcReady(t.Context(), conn, nil))

	client, err := extProcPb.NewExternalProcessorClient(conn).Process(t.Context())
	require.NoError(t, err)
	res, err := SendRequest(t, client, ReqHeaderOnly(map[string]string{"hi": "mom"})[0])
	require.NoError(t, err)
	assert.NotNil(t, res)
}

func TestWaitExtProcReady(t *testing.T) {
	t.Run("ready once the server serves", func(t *testing.T) {
		lis, err := testutils.ReserveListener()
		require.NoError(t, err)
		serveStub(t, lis)

		conn := dialLocal(t, lis.Addr().(*net.TCPAddr).Port)
		assert.NoError(t, WaitExtProcReady(t.Context(), conn, nil))
	})

	t.Run("manager error returns before the timeout", func(t *testing.T) {
		// Bound but never served: a dial would succeed, so only the manager error can
		// end this wait early.
		lis, err := testutils.ReserveListener()
		require.NoError(t, err)
		t.Cleanup(func() { _ = lis.Close() })

		bindErr := fmt.Errorf("gRPC server failed to listen - %w",
			&net.OpError{Op: "listen", Net: "tcp", Err: os.NewSyscallError("bind", syscall.EADDRINUSE)})
		mgrErr := make(chan error, 1)
		mgrErr <- bindErr

		conn := dialLocal(t, lis.Addr().(*net.TCPAddr).Port)
		start := time.Now()
		got := WaitExtProcReady(t.Context(), conn, mgrErr)
		require.ErrorIs(t, got, syscall.EADDRINUSE)
		assert.Less(t, time.Since(start), extprocConnSetupTimeout)
	})

	t.Run("manager exit without error is not readiness", func(t *testing.T) {
		lis, err := testutils.ReserveListener()
		require.NoError(t, err)
		t.Cleanup(func() { _ = lis.Close() })

		mgrErr := make(chan error, 1)
		mgrErr <- nil

		conn := dialLocal(t, lis.Addr().(*net.TCPAddr).Port)
		require.ErrorContains(t, WaitExtProcReady(t.Context(), conn, mgrErr), "manager exited")
	})

	t.Run("cancelled context returns immediately", func(t *testing.T) {
		lis, err := testutils.ReserveListener()
		require.NoError(t, err)
		t.Cleanup(func() { _ = lis.Close() })

		ctx, cancel := context.WithCancel(t.Context())
		cancel()

		conn := dialLocal(t, lis.Addr().(*net.TCPAddr).Port)
		assert.ErrorIs(t, WaitExtProcReady(ctx, conn, nil), context.Canceled)
	})
}
