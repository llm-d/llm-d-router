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

package epp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"syscall"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

// echoServer is a minimal ext_proc server: it replies to every request with one
// response, which is all the client-side executors need to be exercised.
type echoServer struct {
	extProcPb.UnimplementedExternalProcessorServer
	responses int // responses to emit per request; 0 means echo one per request
}

func (s *echoServer) Process(stream extProcPb.ExternalProcessor_ProcessServer) error {
	for {
		if _, err := stream.Recv(); err != nil {
			return nil // client closed or context ended
		}
		for range max(s.responses, 1) {
			if err := stream.Send(NewResponseStreamChunk("ack", false)); err != nil {
				return err
			}
		}
	}
}

// startEchoServer serves an echoServer on a free port and returns that port.
func startEchoServer(t *testing.T, srv *echoServer) int {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	serve(t, lis, srv)

	return lis.Addr().(*net.TCPAddr).Port
}

// serve starts srv on lis and stops it at test end.
func serve(t *testing.T, lis net.Listener, srv *echoServer) {
	t.Helper()
	grpcSrv := grpc.NewServer()
	extProcPb.RegisterExternalProcessorServer(grpcSrv, srv)
	go func() { _ = grpcSrv.Serve(lis) }()
	t.Cleanup(grpcSrv.Stop)
}

func dialLocal(t *testing.T, port int) *grpc.ClientConn {
	t.Helper()
	conn, err := grpc.NewClient(fmt.Sprintf("127.0.0.1:%d", port),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

func TestExtProcServerClient(t *testing.T) {
	port := startEchoServer(t, &echoServer{})

	client, conn := ExtProcServerClient(t.Context(), t, port, logr.Discard(), nil)

	require.NotNil(t, client)
	require.Equal(t, fmt.Sprintf("127.0.0.1:%d", port), conn.Target(), "client must target the requested port over IPv4")

	// The returned client is a live ext_proc stream, not just a non-nil handle.
	require.NoError(t, client.Send(ReqHeaderOnly(nil)[0]))
	res, err := client.Recv()
	require.NoError(t, err)
	assert.Equal(t, []byte("ack"),
		res.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse().GetBody())
}

func TestStreamedRequest(t *testing.T) {
	tests := []struct {
		name              string
		requests          int
		serverResponses   int
		expectedResponses int
		wantResponses     int
	}{
		{name: "one response per request", requests: 2, expectedResponses: 2, wantResponses: 2},
		{name: "server fans out extra responses", requests: 1, serverResponses: 3, expectedResponses: 3, wantResponses: 3},
		{name: "collects fewer than the server sends", requests: 2, expectedResponses: 1, wantResponses: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			port := startEchoServer(t, &echoServer{responses: tc.serverResponses})
			client, _ := ExtProcServerClient(t.Context(), t, port, logr.Discard(), nil)

			reqs := make([]*extProcPb.ProcessingRequest, 0, tc.requests)
			for range tc.requests {
				reqs = append(reqs, ReqHeaderOnly(map[string]string{"x-a": "1"})...)
			}

			got, err := StreamedRequest(t, client, reqs, tc.expectedResponses)

			require.NoError(t, err)
			assert.Len(t, got, tc.wantResponses)
		})
	}
}

func TestStreamedRequestServerClosesStream(t *testing.T) {
	// The server stops after the first response, so the client sees io.EOF while
	// waiting for the second: a valid termination, not an error.
	port := startEchoServer(t, &echoServer{})
	client, _ := ExtProcServerClient(t.Context(), t, port, logr.Discard(), nil)

	require.NoError(t, client.CloseSend())

	got, err := StreamedRequest(t, client, nil, 1)

	require.NoError(t, err)
	assert.Empty(t, got)
}

// stubStream drives StreamedRequest's failure paths deterministically; the embedded
// nil ClientStream is never touched because only Send and Recv are called.
type stubStream struct {
	grpc.ClientStream
	sendErr  error
	recvErrs []error
}

func (s *stubStream) Send(*extProcPb.ProcessingRequest) error { return s.sendErr }

func (s *stubStream) Recv() (*extProcPb.ProcessingResponse, error) {
	if len(s.recvErrs) == 0 {
		return NewResponseStreamChunk("ack", false), nil
	}
	err := s.recvErrs[0]
	s.recvErrs = s.recvErrs[1:]
	if err != nil {
		return nil, err
	}
	return NewResponseStreamChunk("ack", false), nil
}

func TestStreamedRequestFailurePaths(t *testing.T) {
	boom := errors.New("boom")

	tests := []struct {
		name          string
		stream        *stubStream
		wantErrIs     error
		wantResponses int
	}{
		{
			name:      "send failure aborts before receiving",
			stream:    &stubStream{sendErr: boom},
			wantErrIs: boom,
		},
		{
			name:      "receive failure is returned",
			stream:    &stubStream{recvErrs: []error{boom}},
			wantErrIs: boom,
		},
		{
			// io.EOF means the server closed the stream early (e.g. it rejected the
			// request); the responses collected so far are returned without an error.
			name:          "server EOF terminates collection cleanly",
			stream:        &stubStream{recvErrs: []error{nil, io.EOF}},
			wantResponses: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := StreamedRequest(t, tc.stream, ReqHeaderOnly(nil), 2)

			if tc.wantErrIs != nil {
				require.ErrorIs(t, err, tc.wantErrIs)
				assert.Nil(t, got)
				return
			}
			require.NoError(t, err)
			assert.Len(t, got, tc.wantResponses)
		})
	}
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
	lis, err := fwknet.ReserveListener()
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
	lis, err := fwknet.ReserveListener()
	require.NoError(t, err)
	port := lis.Addr().(*net.TCPAddr).Port

	raw, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), time.Second)
	require.NoError(t, err, "TCP dial must succeed before Serve runs")
	require.NoError(t, raw.Close())

	serve(t, lis, &echoServer{})

	conn := dialLocal(t, port)
	require.NoError(t, WaitExtProcReady(t.Context(), conn, nil))

	client, err := extProcPb.NewExternalProcessorClient(conn).Process(t.Context())
	require.NoError(t, err)
	require.NoError(t, client.Send(ReqHeaderOnly(map[string]string{"hi": "mom"})[0]))
	res, err := client.Recv()
	require.NoError(t, err)
	assert.NotNil(t, res)
}

func TestWaitExtProcReady(t *testing.T) {
	t.Run("ready once the server serves", func(t *testing.T) {
		lis, err := fwknet.ReserveListener()
		require.NoError(t, err)
		serve(t, lis, &echoServer{})

		conn := dialLocal(t, lis.Addr().(*net.TCPAddr).Port)
		assert.NoError(t, WaitExtProcReady(t.Context(), conn, nil))
	})

	t.Run("manager error returns before the timeout", func(t *testing.T) {
		// Bound but never served: a dial would succeed, so only the manager error can
		// end this wait early.
		lis, err := fwknet.ReserveListener()
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
		lis, err := fwknet.ReserveListener()
		require.NoError(t, err)
		t.Cleanup(func() { _ = lis.Close() })

		mgrErr := make(chan error, 1)
		mgrErr <- nil

		conn := dialLocal(t, lis.Addr().(*net.TCPAddr).Port)
		require.ErrorContains(t, WaitExtProcReady(t.Context(), conn, mgrErr), "manager exited")
	})

	t.Run("cancelled context returns immediately", func(t *testing.T) {
		lis, err := fwknet.ReserveListener()
		require.NoError(t, err)
		t.Cleanup(func() { _ = lis.Close() })

		ctx, cancel := context.WithCancel(t.Context())
		cancel()

		conn := dialLocal(t, lis.Addr().(*net.TCPAddr).Port)
		assert.ErrorIs(t, WaitExtProcReady(ctx, conn, nil), context.Canceled)
	})
}
