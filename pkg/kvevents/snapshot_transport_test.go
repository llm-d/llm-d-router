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

package kvevents_test

import (
	"context"
	"fmt"
	"net"
	"testing"
	"time"

	zmq "github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
	"github.com/stretchr/testify/require"
)

func TestSnapshotShutdownCancelsStalledHandshake(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer listener.Close()
	accepted := make(chan net.Conn, 1)
	go func() {
		conn, err := listener.Accept()
		if err == nil {
			accepted <- conn
		}
	}()
	cfg := kvevents.DefaultConfig()
	cfg.SnapshotPort = 5559
	tokens, err := kvblock.NewChunkedTokenDatabase(nil)
	require.NoError(t, err)
	manager, err := kvevents.NewSnapshotManager(cfg, nil, tokens, engineadapter.NewVLLMAdapter(), nil)
	require.NoError(t, err)
	require.NoError(t, manager.EnsureSubscriber(ctx, "pod", "pod:8000", "tcp://"+listener.Addr().String(), "", "kv@", true))
	var conn net.Conn
	select {
	case conn = <-accepted:
	case <-time.After(time.Second):
		t.Fatal("no connection")
	}
	defer conn.Close()
	done := make(chan struct{})
	go func() { manager.Shutdown(ctx); close(done) }()
	select {
	case <-done:
	case <-time.After(time.Second):
		conn.Close()
		<-done
		t.Fatal("shutdown waited for a stalled ZMQ handshake")
	}
}

func TestSnapshotRetriesStalledHandshake(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer listener.Close()
	accepted := make(chan net.Conn, 4)
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			accepted <- conn
		}
	}()
	cfg := kvevents.DefaultConfig()
	cfg.SnapshotPort = 5559
	tokens, err := kvblock.NewChunkedTokenDatabase(nil)
	require.NoError(t, err)
	manager, err := kvevents.NewSnapshotManager(cfg, nil, tokens, engineadapter.NewVLLMAdapter(), nil)
	require.NoError(t, err)
	defer manager.Shutdown(ctx)
	require.NoError(t, manager.EnsureSubscriber(ctx, "pod", "pod:8000", "tcp://"+listener.Addr().String(), "", "kv@", true))
	for i := range 2 {
		select {
		case conn := <-accepted:
			defer conn.Close()
		case <-time.After(15 * time.Second):
			t.Fatalf("connection %d not attempted; a stalled greeting must not block the subscriber", i+1)
		}
	}
}

func TestSilentPublishersDoNotHoldRecoverySlots(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cfg := kvevents.DefaultConfig()
	cfg.PodDiscoveryConfig.SocketPort = 1024
	cfg.SnapshotPort = 1
	tokens, err := kvblock.NewChunkedTokenDatabase(nil)
	require.NoError(t, err)
	manager, err := kvevents.NewSnapshotManager(cfg, nil, tokens, engineadapter.NewVLLMAdapter(), nil)
	require.NoError(t, err)
	defer manager.Shutdown(ctx)
	listen := func() (zmq.Socket, string) {
		pub := zmq.NewPub(ctx)
		require.NoError(t, pub.Listen("tcp://127.0.0.1:0"))
		t.Cleanup(func() { pub.Close() })
		return pub, "tcp://" + pub.Addr().String()
	}
	// Engines without a snapshot endpoint send nothing while idle; more of them
	// than recovery slots must not keep an active publisher from connecting.
	for i := range 4 {
		_, endpoint := listen()
		require.NoError(t, manager.EnsureSubscriber(ctx, fmt.Sprintf("idle-%d", i), fmt.Sprintf("idle-%d:8000", i), endpoint, "", "kv@", true))
	}
	time.Sleep(200 * time.Millisecond)
	active, endpoint := listen()
	require.NoError(t, manager.EnsureSubscriber(ctx, "active", "active:8000", endpoint, "", "kv@", true))
	batch, err := msgpack.Marshal([]any{1.0, []any{}, nil})
	require.NoError(t, err)
	seq := make([]byte, 8)
	require.Eventually(t, func() bool {
		_ = active.Send(zmq.NewMsgFrom([]byte("kv@127.0.0.1:8000@test-model"), seq, batch))
		return manager.Status().LiveOnly == 1
	}, 10*time.Second, 50*time.Millisecond)
}
