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
	"net"
	"testing"
	"time"

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
