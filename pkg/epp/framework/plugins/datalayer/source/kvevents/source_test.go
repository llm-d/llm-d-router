// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvevents //nolint:testpackage // verifies the shared subscriber and consumer list

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"net"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-zeromq/zmq4"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	events "github.com/llm-d/llm-d-router/pkg/kvevents"
)

type recordingExtractor struct {
	name       string
	mu         sync.Mutex
	operations []string
	sequences  []uint64
	err        error
	panics     bool
}

func (r *recordingExtractor) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: r.name, Name: r.name}
}
func (r *recordingExtractor) ProcessEvents(_ context.Context, source events.EventSource, _ events.EventBatch) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.operations = append(r.operations, "batch:"+source.Endpoint)
	r.sequences = append(r.sequences, source.Sequence)
	if r.panics {
		panic("consumer failure")
	}
	return r.err
}
func (r *recordingExtractor) Reset(_ context.Context, source events.EventSource) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.operations = append(r.operations, "reset:"+source.Endpoint)
	return nil
}
func (r *recordingExtractor) lastSequence() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.sequences) == 0 {
		return 0
	}
	return r.sequences[len(r.sequences)-1]
}

func TestSourceIsolatesConsumerFailures(t *testing.T) {
	for _, panics := range []bool{false, true} {
		t.Run(strconv.FormatBool(panics), func(t *testing.T) {
			bad := &recordingExtractor{name: "bad", err: errors.New("failed"), panics: panics}
			good := &recordingExtractor{name: "good"}
			s := &Source{extractors: []extractor{bad, good}, subscribers: events.NewSubscriberManager(nil)}
			source := events.EventSource{Endpoint: "engine:8000", Sequence: 2}
			require.NoError(t, s.ProcessEvents(t.Context(), source, events.EventBatch{}))
			require.Equal(t, []string{"batch:engine:8000", "reset:engine:8000"}, bad.operations)
			require.Equal(t, []string{"batch:engine:8000"}, good.operations)
			require.NoError(t, s.Reset(t.Context(), source))
			require.Equal(t, []string{"batch:engine:8000", "reset:engine:8000"}, good.operations)
		})
	}
}

func TestSourceSharesOneSubscriber(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	pub := zmq4.NewPub(ctx)
	require.NoError(t, pub.Listen("tcp://127.0.0.1:0"))
	defer pub.Close()
	_, port, err := net.SplitHostPort(pub.Addr().String())
	require.NoError(t, err)
	socketPort, err := strconv.Atoi(port)
	require.NoError(t, err)
	cfg := events.DefaultConfig()
	cfg.PodDiscoveryConfig.PodLabelSelector = ""
	cfg.PodDiscoveryConfig.SocketPort = socketPort
	s, err := New(ctx, "events", cfg)
	require.NoError(t, err)
	a, b := &recordingExtractor{name: "cache"}, &recordingExtractor{name: "sessions"}
	require.NoError(t, s.AppendExtractor(a))
	require.NoError(t, s.AppendExtractor(b))
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine"}, Address: "127.0.0.1", Port: "8000"}, nil)
	event := fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}
	for range 2 {
		out, err := s.NotifyEndpoint(ctx, event)
		require.NoError(t, err)
		require.Nil(t, out)
	}
	pods, _ := s.subscribers.GetActiveSubscribers()
	require.Len(t, pods, 1)
	require.ErrorContains(t, s.AppendExtractor(&recordingExtractor{name: "late"}), "before endpoint discovery")
	payload, err := msgpack.Marshal([]any{1.0, []any{[]any{string(events.EventTypeAllBlocksCleared)}}, nil})
	require.NoError(t, err)
	var seq uint64
	require.Eventually(t, func() bool {
		seq++
		frame := make([]byte, 8)
		binary.BigEndian.PutUint64(frame, seq)
		require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte("kv@wire:8000@model"), frame, payload)))
		return a.lastSequence() > 0 && a.lastSequence() == b.lastSequence()
	}, 5*time.Second, 20*time.Millisecond)
	_, err = s.NotifyEndpoint(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: ep})
	require.NoError(t, err)
	pods, _ = s.subscribers.GetActiveSubscribers()
	require.Empty(t, pods)
	require.Eventually(t, func() bool {
		a.mu.Lock()
		defer a.mu.Unlock()
		b.mu.Lock()
		defer b.mu.Unlock()
		return a.operations[len(a.operations)-1] == "reset:127.0.0.1:8000" && b.operations[len(b.operations)-1] == "reset:127.0.0.1:8000"
	}, time.Second, time.Millisecond)
}

func TestNewRejectsNonPositiveSocketPort(t *testing.T) {
	for _, port := range []int{0, -1} {
		cfg := events.DefaultConfig()
		cfg.PodDiscoveryConfig.SocketPort = port
		_, err := New(t.Context(), "events", cfg)
		require.ErrorContains(t, err, "socketPort must be positive")
	}
}

func TestNotifyEndpointDoesNotWaitForConsumers(t *testing.T) {
	s := &Source{subscribers: events.NewSubscriberManager(nil)}
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)
	event := fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}
	_, err := s.NotifyEndpoint(t.Context(), event)
	require.NoError(t, err)
	// A consumer processing a batch holds the read lock.
	s.mu.RLock()
	defer s.mu.RUnlock()
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = s.NotifyEndpoint(t.Context(), event)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("endpoint notification waited for a consumer")
	}
}

func TestSourceDumpsSubscribersAndConsumers(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	cfg := events.DefaultConfig()
	cfg.PodDiscoveryConfig.PodLabelSelector = ""
	s, err := New(ctx, "events", cfg)
	require.NoError(t, err)
	require.NoError(t, s.AppendExtractor(&recordingExtractor{name: "cache"}))
	for _, name := range []string{"engine-b", "engine-a"} {
		ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name}, Address: "127.0.0.1", Port: "8000"}, nil)
		_, err := s.NotifyEndpoint(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep})
		require.NoError(t, err)
	}
	raw, err := s.DumpState()
	require.NoError(t, err)
	var state sourceState
	require.NoError(t, json.Unmarshal(raw, &state))
	require.Equal(t, sourceState{
		Subscribers: []string{"/engine-a", "/engine-b"}, TotalSubscribers: 2,
		MaxSubscribers: maxDumpSubscribers, Consumers: []string{"cache/cache"},
	}, state)
}

func TestSourceResyncsEndpointAfterConsumerFailure(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	listen := func(sock zmq4.Socket) int {
		require.NoError(t, sock.Listen("tcp://127.0.0.1:0"))
		_, port, err := net.SplitHostPort(sock.Addr().String())
		require.NoError(t, err)
		n, err := strconv.Atoi(port)
		require.NoError(t, err)
		return n
	}
	pub, rep := zmq4.NewPub(ctx), zmq4.NewRep(ctx)
	defer pub.Close()
	defer rep.Close()
	cfg := events.DefaultConfig()
	cfg.PodDiscoveryConfig.PodLabelSelector = ""
	cfg.PodDiscoveryConfig.SocketPort = listen(pub)
	cfg.PodDiscoveryConfig.SnapshotSocketPort = listen(rep)

	// The snapshot is empty and cut at the latest published sequence.
	var seq atomic.Uint64
	var snapshots atomic.Int32
	go func() {
		for {
			if _, err := rep.Recv(); err != nil {
				return
			}
			snapshots.Add(1)
			cut := make([]byte, 8)
			binary.BigEndian.PutUint64(cut, seq.Load())
			if err := rep.Send(zmq4.NewMsgFrom(cut, make([]byte, 16))); err != nil {
				return
			}
		}
	}()

	s, err := New(ctx, "events", cfg)
	require.NoError(t, err)
	bad, good := &recordingExtractor{name: "bad"}, &recordingExtractor{name: "good"}
	require.NoError(t, s.AppendExtractor(bad))
	require.NoError(t, s.AppendExtractor(good))
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine"}, Address: "127.0.0.1", Port: "8000"}, nil)
	_, err = s.NotifyEndpoint(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep})
	require.NoError(t, err)

	payload, err := msgpack.Marshal([]any{1.0, []any{[]any{string(events.EventTypeAllBlocksCleared)}}, nil})
	require.NoError(t, err)
	send := func() {
		frame := make([]byte, 8)
		binary.BigEndian.PutUint64(frame, seq.Add(1))
		require.NoError(t, pub.Send(zmq4.NewMsgFrom([]byte("kv@wire:8000@model"), frame, payload)))
	}
	require.Eventually(t, func() bool { send(); return good.lastSequence() > 0 }, 5*time.Second, 20*time.Millisecond)
	require.Equal(t, int32(1), snapshots.Load())

	bad.mu.Lock()
	bad.err = errors.New("failed")
	bad.mu.Unlock()
	require.Eventually(t, func() bool { send(); return snapshots.Load() == 2 }, 5*time.Second, 20*time.Millisecond)
	bad.mu.Lock()
	bad.err = nil
	bad.mu.Unlock()
	// Both consumers continue from the rebuilt endpoint state.
	resumeFrom := seq.Load()
	require.Eventually(t, func() bool {
		send()
		return bad.lastSequence() > resumeFrom && bad.lastSequence() == good.lastSequence()
	}, 5*time.Second, 20*time.Millisecond)
	require.Equal(t, int32(2), snapshots.Load())
}
