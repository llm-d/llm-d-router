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
	"context"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	healthgrpc "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
)

// streamEndRecorder records when an ext_proc stream's context ends.
type streamEndRecorder struct {
	extProcPb.UnimplementedExternalProcessorServer
	started chan struct{}
	ended   chan struct{}
}

func (s *streamEndRecorder) Process(stream extProcPb.ExternalProcessor_ProcessServer) error {
	close(s.started)
	<-stream.Context().Done()
	close(s.ended)
	return nil
}

// blackholeProxy forwards TCP until cut, then drops traffic but keeps sockets open.
type blackholeProxy struct {
	listener net.Listener
	cut      atomic.Bool
	// serverClosed is closed when the server side of the proxied connection closes.
	serverClosed chan struct{}
}

func newBlackholeProxy(t *testing.T, target string) *blackholeProxy {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	p := &blackholeProxy{listener: lis, serverClosed: make(chan struct{})}
	closeOnce := sync.OnceFunc(func() { close(p.serverClosed) })
	t.Cleanup(func() { _ = lis.Close() })
	go func() {
		for {
			client, err := lis.Accept()
			if err != nil {
				return
			}
			upstream, err := net.Dial("tcp", target)
			if err != nil {
				_ = client.Close()
				return
			}
			t.Cleanup(func() { _ = client.Close(); _ = upstream.Close() })
			go p.pipe(upstream, client)
			go func() {
				p.pipe(client, upstream)
				closeOnce()
			}()
		}
	}()
	return p
}

func (p *blackholeProxy) pipe(dst io.Writer, src io.Reader) {
	buf := make([]byte, 32*1024)
	for {
		n, err := src.Read(buf)
		if err != nil {
			return
		}
		if !p.cut.Load() {
			if _, err := dst.Write(buf[:n]); err != nil {
				return
			}
		}
	}
}

// openStreamThenPartition opens a stream, cuts the network, then closes the client.
func openStreamThenPartition(t *testing.T, opts ...grpc.ServerOption) *streamEndRecorder {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	rec := &streamEndRecorder{started: make(chan struct{}), ended: make(chan struct{})}
	srv := grpc.NewServer(opts...)
	extProcPb.RegisterExternalProcessorServer(srv, rec)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	proxy := newBlackholeProxy(t, lis.Addr().String())
	conn, err := grpc.NewClient(proxy.listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := extProcPb.NewExternalProcessorClient(conn).Process(context.Background()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-rec.started:
	case <-time.After(5 * time.Second):
		t.Fatal("stream did not reach the server")
	}

	proxy.cut.Store(true)
	_ = conn.Close()
	return rec
}

func TestExtProcStreamOutlivesPartitionWithoutKeepalive(t *testing.T) {
	rec := openStreamThenPartition(t)
	select {
	case <-rec.ended:
		t.Fatal("stream ended although the client's close was dropped")
	case <-time.After(3 * time.Second):
	}
}

func TestExtProcKeepaliveEndsStreamAfterPartition(t *testing.T) {
	// gRPC raises a server ping interval below 1s to 1s.
	rec := openStreamThenPartition(t, keepaliveServerOptions(time.Second, 500*time.Millisecond)...)
	select {
	case <-rec.ended:
	case <-time.After(5 * time.Second):
		t.Fatal("stream still open 5s after the partition; keepalive did not close the dead connection")
	}
}

// TestExtProcServerRunnerClosesDeadConnections checks that AsRunnable enables keepalive.
func TestExtProcServerRunnerClosesDeadConnections(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	r := NewDefaultExtProcServerRunner()
	r.SecureServing = false
	r.GrpcListener = lis
	r.Datastore = datastore.NewDatastore(t.Context(), nil)
	r.keepaliveTime, r.keepaliveTimeout = time.Second, 500*time.Millisecond
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	go func() { _ = r.AsRunnable(logutil.NewTestLogger()).Start(ctx) }()

	proxy := newBlackholeProxy(t, lis.Addr().String())
	conn, err := grpc.NewClient(proxy.listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	// A round trip proves the server finished its handshake, which is when its keepalive starts.
	// Ready is not enough: the client can reach it from the server's SETTINGS alone.
	waitCtx, waitCancel := context.WithTimeout(ctx, 5*time.Second)
	defer waitCancel()
	_, err = healthgrpc.NewHealthClient(conn).Check(waitCtx, &healthgrpc.HealthCheckRequest{}, grpc.WaitForReady(true))
	if code := status.Code(err); code != codes.OK && code != codes.Unimplemented {
		t.Fatalf("no round trip to the ext_proc server: %v", err)
	}

	proxy.cut.Store(true)
	select {
	case <-proxy.serverClosed:
	case <-time.After(5 * time.Second):
		t.Fatal("server kept the connection open 5s after the network was cut")
	}
}
