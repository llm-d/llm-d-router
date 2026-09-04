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
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
)

// --- Execution Helpers ---

// StreamedRequest is a helper for Full-Duplex Streaming test scenarios.
// It performs the following actions:
//  1. Sends all requests in the provided slice to the server.
//  2. Listens for responses on the stream until 'expectedResponses' count is reached.
//  3. Enforces a 10-second timeout to prevent deadlocks if the server hangs.
//  4. Handles io.EOF gracefully (server closed stream).
func StreamedRequest(
	t *testing.T,
	client extProcPb.ExternalProcessor_ProcessClient,
	requests []*extProcPb.ProcessingRequest,
	expectedResponses int,
) ([]*extProcPb.ProcessingResponse, error) {
	t.Helper()

	// 1. Send Phase
	for _, req := range requests {
		t.Logf("Sending request: %v", req)
		if err := client.Send(req); err != nil {
			t.Logf("Failed to send request: %v", err)
			return nil, err
		}
	}

	// 2. Receive Phase
	// We use a channel and a separate goroutine for receiving to allow for a strict timeout via select{}.
	type recvResult struct {
		res *extProcPb.ProcessingResponse
		err error
	}

	// Buffered channel avoids blocking the goroutine on the last read.
	recvChan := make(chan recvResult, expectedResponses+1)

	// Start reading in background.
	go func() {
		for range expectedResponses {
			res, err := client.Recv()
			recvChan <- recvResult{res, err}
			if err != nil {
				return // Stop reading on error or EOF.
			}
		}
	}()

	var responses []*extProcPb.ProcessingResponse

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	// Collect results with timeout.
	for i := range expectedResponses {
		select {
		case <-ctx.Done():
			t.Logf("Timeout waiting for response %d of %d: %v", i+1, expectedResponses, ctx.Err())
			return responses, fmt.Errorf("timeout waiting for responses: %w", ctx.Err())

		case result := <-recvChan:
			if result.err != nil {
				// io.EOF is a valid termination from the server side (e.g. rejection).
				if result.err == io.EOF {
					return responses, nil
				}
				t.Logf("Failed to receive: %v", result.err)
				return nil, result.err
			}
			t.Logf("Received response: %+v", result.res)
			responses = append(responses, result.res)
		}
	}

	return responses, nil
}

// --- System Utilities ---

// ExtProcServerClient returns a ExternalProcessor_ProcessClient listen to localhost on given port.
// mgrErr, when non-nil, carries the early exit of the manager hosting the server so a start
// failure fails the test immediately instead of waiting out extprocConnSetupTimeout.
func ExtProcServerClient(
	ctx context.Context,
	t *testing.T,
	port int,
	logger logr.Logger,
	mgrErr <-chan error,
) (extProcPb.ExternalProcessor_ProcessClient, *grpc.ClientConn) {
	t.Helper()

	// Force IPv4: the server binds on IPv4 localhost, and a dual-stack target would let
	// the client reach for an IPv6 address nothing is listening on.
	serverAddr := fmt.Sprintf("127.0.0.1:%d", port)

	conn, err := grpc.NewClient(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err, "failed to create grpc connection")
	// Registered before any require below can call FailNow, so the conn and its
	// goroutines do not leak on the failure path.
	t.Cleanup(func() { _ = conn.Close() })

	require.NoError(t, WaitExtProcReady(ctx, conn, mgrErr), "ext-proc server did not become ready")

	extProcClient, err := extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	require.NoError(t, err, "failed to initialize ext_proc stream client")

	return extProcClient, conn
}

// WaitExtProcReady blocks until conn reaches connectivity.Ready (nil), the manager
// reports an early exit via mgrErr, ctx is done, or extprocConnSetupTimeout elapses.
//
// The server serves on a listener bound before it starts, so the kernel completes the
// TCP handshake out of the listen backlog and a dial succeeds even while grpc.Serve has
// not run yet. Only Ready, which requires the HTTP/2 handshake, proves it is serving.
// The state wait is bounded by the poll interval so an early mgrErr is still observed
// promptly rather than after the full timeout.
func WaitExtProcReady(ctx context.Context, conn *grpc.ClientConn, mgrErr <-chan error) error {
	deadline := time.Now().Add(extprocConnSetupTimeout)
	for {
		state := conn.GetState()
		if state == connectivity.Ready {
			return nil
		}
		if state == connectivity.Idle {
			// A ClientConn only leaves Idle on Connect or a started RPC, so a
			// Ready to Idle flap would otherwise stall until the deadline.
			conn.Connect()
		}

		select {
		case err := <-mgrErr:
			if err == nil {
				return errors.New("manager exited before ext-proc server became ready")
			}
			return err
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if time.Now().After(deadline) {
			return fmt.Errorf("ext-proc server at %s not ready within %s (state %s)",
				conn.Target(), extprocConnSetupTimeout, state)
		}

		waitCtx, cancel := context.WithTimeout(ctx, extPorcConnSetupPollInterval)
		conn.WaitForStateChange(waitCtx, state)
		cancel()
	}
}
