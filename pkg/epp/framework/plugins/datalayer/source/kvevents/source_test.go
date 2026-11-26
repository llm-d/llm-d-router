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
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	events "github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

type recordingRegistrar struct{ registrations []fwkdl.PendingRegistration }

func (r *recordingRegistrar) Register(p fwkdl.PendingRegistration) error {
	r.registrations = append(r.registrations, p)
	return nil
}

func TestSourceRegistersForEndpointEvents(t *testing.T) {
	s := &Source{typedName: plugin.TypedName{Type: PluginType, Name: "events"}}
	r := &recordingRegistrar{}
	require.NoError(t, s.RegisterDependencies(r))
	require.Len(t, r.registrations, 1)
	registration := r.registrations[0]
	require.Equal(t, s.typedName, registration.Owner)
	require.Equal(t, sourcenotifications.EndpointNotificationSourceType, registration.SourceType)
	require.Same(t, s, registration.Extractor)
	require.NotNil(t, registration.DefaultSource)
}

func TestSourceEngineType(t *testing.T) {
	require.Equal(t, engineadapter.EngineTypeVLLM, (&Source{}).EngineType())
	require.Equal(t, engineadapter.EngineTypeVLLM, (&Source{config: &events.Config{}}).EngineType())
	require.Equal(t, engineadapter.EngineTypeSGLang, (&Source{config: &events.Config{EngineType: engineadapter.EngineTypeSGLang}}).EngineType())
}

type recordingConsumer struct {
	name       string
	mu         sync.Mutex
	operations []string
	sequences  []uint64
	err        error
	panics     bool
}

func (r *recordingConsumer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: r.name, Name: r.name}
}
func (r *recordingConsumer) ProcessEvents(_ context.Context, source events.EventSource, _ events.EventBatch) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.operations = append(r.operations, "batch:"+source.Endpoint)
	r.sequences = append(r.sequences, source.Sequence)
	if r.panics {
		panic("consumer failure")
	}
	return r.err
}
func (r *recordingConsumer) Reset(_ context.Context, source events.EventSource) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.operations = append(r.operations, "reset:"+source.Endpoint)
	return nil
}
func (r *recordingConsumer) lastSequence() uint64 {
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
			bad := &recordingConsumer{name: "bad", err: errors.New("failed"), panics: panics}
			good := &recordingConsumer{name: "good"}
			s := &Source{consumers: []consumer{{owner: bad.TypedName(), EventConsumer: bad}, {owner: good.TypedName(), EventConsumer: good}}, subscribers: events.NewSubscriberManager(nil)}
			source := events.EventSource{Endpoint: "engine:8000", Sequence: 2}
			require.NoError(t, s.ProcessEvents(t.Context(), source, events.EventBatch{}))
			require.Equal(t, []string{"batch:engine:8000", "reset:engine:8000"}, bad.operations)
			require.Equal(t, []string{"batch:engine:8000"}, good.operations)
			require.NoError(t, s.Reset(t.Context(), source))
			require.Equal(t, []string{"batch:engine:8000", "reset:engine:8000"}, good.operations)
		})
	}
}

func TestSourceResyncReachesOnlyTheFailedConsumer(t *testing.T) {
	bad := &recordingConsumer{name: "bad", err: errors.New("failed")}
	good := &recordingConsumer{name: "good"}
	s := &Source{consumers: []consumer{{owner: bad.TypedName(), EventConsumer: bad}, {owner: good.TypedName(), EventConsumer: good}}, subscribers: events.NewSubscriberManager(nil)}
	ctx := t.Context()
	at := func(seq uint64, resync bool) events.EventSource {
		return events.EventSource{Endpoint: "engine:8000", Sequence: seq, Resync: resync}
	}
	require.NoError(t, s.ProcessEvents(ctx, at(2, false), events.EventBatch{}))
	bad.err = nil
	bad.operations, good.operations, bad.sequences, good.sequences = nil, nil, nil, nil

	// The rebuild covers the stream up to sequence 4.
	require.NoError(t, s.Reset(ctx, at(4, true)))
	require.NoError(t, s.ProcessEvents(ctx, at(0, true), events.EventBatch{}))
	require.Equal(t, []string{"reset:engine:8000", "batch:engine:8000"}, bad.operations)
	require.Empty(t, good.operations, "a resync leaves the other consumers' state alone")

	for _, seq := range []uint64{3, 4, 5} {
		require.NoError(t, s.ProcessEvents(ctx, at(seq, false), events.EventBatch{}))
	}
	require.Equal(t, []uint64{0, 5}, bad.sequences, "the resynced consumer skips batches the rebuild covers")
	require.Equal(t, []uint64{3, 4, 5}, good.sequences)

	// A later resync without a failure resets no consumer.
	require.NoError(t, s.Reset(ctx, at(6, true)))
	require.Equal(t, []string{"reset:engine:8000", "batch:engine:8000", "batch:engine:8000"}, bad.operations)
	require.NoError(t, s.Reset(ctx, at(0, false)))
	require.Equal(t, "reset:engine:8000", good.operations[len(good.operations)-1])
}

func TestSourceResyncRecovery(t *testing.T) {
	ctx := t.Context()
	at := func(seq uint64, resync bool) events.EventSource {
		return events.EventSource{Endpoint: "engine:8000", Sequence: seq, Resync: resync}
	}
	// setup returns a source whose first consumer failed on sequence 2 and
	// waits for a resync.
	setup := func(t *testing.T) (*Source, *recordingConsumer, *recordingConsumer) {
		bad := &recordingConsumer{name: "bad", err: errors.New("failed")}
		other := &recordingConsumer{name: "other"}
		s := &Source{consumers: []consumer{{owner: bad.TypedName(), EventConsumer: bad}, {owner: other.TypedName(), EventConsumer: other}}, subscribers: events.NewSubscriberManager(nil)}
		require.NoError(t, s.ProcessEvents(ctx, at(2, false), events.EventBatch{}))
		bad.err = nil
		bad.operations, other.operations, bad.sequences, other.sequences = nil, nil, nil, nil
		return s, bad, other
	}
	batch, reset := "batch:engine:8000", "reset:engine:8000"

	t.Run("an aborted partial resync rebuilds on the next resync", func(t *testing.T) {
		s, bad, other := setup(t)
		require.NoError(t, s.Reset(ctx, at(4, true)))
		require.NoError(t, s.ProcessEvents(ctx, at(1, true), events.EventBatch{}))
		aborted := at(0, true)
		aborted.ResyncAborted = true
		require.NoError(t, s.Reset(ctx, aborted))
		require.Equal(t, []string{reset, batch, reset}, bad.operations, "the abort drops the partial rebuild")
		require.NoError(t, s.ProcessEvents(ctx, at(5, false), events.EventBatch{}))

		require.NoError(t, s.Reset(ctx, at(5, true)))
		require.NoError(t, s.ProcessEvents(ctx, at(1, true), events.EventBatch{}))
		require.Equal(t, []string{reset, batch, reset, batch, reset, batch}, bad.operations)
		require.Equal(t, []string{batch}, other.operations)
	})

	t.Run("a second resync leaves a recovered consumer alone", func(t *testing.T) {
		s, bad, other := setup(t)
		// The rebuild covers the stream up to sequence 6.
		require.NoError(t, s.Reset(ctx, at(6, true)))
		require.NoError(t, s.ProcessEvents(ctx, at(0, true), events.EventBatch{}))
		other.err = errors.New("failed")
		require.NoError(t, s.ProcessEvents(ctx, at(5, false), events.EventBatch{}))
		other.err = nil
		bad.operations, bad.sequences = nil, nil

		require.NoError(t, s.Reset(ctx, at(8, true)))
		require.NoError(t, s.ProcessEvents(ctx, at(0, true), events.EventBatch{}))
		require.Empty(t, bad.operations, "the recovered consumer is neither reset nor rebuilt again")
		for _, seq := range []uint64{6, 7, 8, 9} {
			require.NoError(t, s.ProcessEvents(ctx, at(seq, false), events.EventBatch{}))
		}
		require.Equal(t, []uint64{7, 8, 9}, bad.sequences, "each consumer skips up to its own rebuild")
		require.Equal(t, []uint64{5, 0, 9}, other.sequences)
	})

	t.Run("a consumer that fails during a resync waits for the next one", func(t *testing.T) {
		s, bad, _ := setup(t)
		require.NoError(t, s.Reset(ctx, at(4, true)))
		bad.err = errors.New("failed")
		require.NoError(t, s.ProcessEvents(ctx, at(1, true), events.EventBatch{}))
		bad.err = nil
		require.NoError(t, s.ProcessEvents(ctx, at(2, true), events.EventBatch{}))
		require.Equal(t, []string{reset, batch, reset}, bad.operations, "later rebuilt batches skip the failed consumer")
		require.NoError(t, s.ProcessEvents(ctx, at(3, false), events.EventBatch{}))
		require.Equal(t, []uint64{1, 3}, bad.sequences, "the failed consumer follows the shared stream until the next resync")

		require.NoError(t, s.Reset(ctx, at(5, true)))
		require.NoError(t, s.ProcessEvents(ctx, at(1, true), events.EventBatch{}))
		require.Equal(t, []string{reset, batch, reset, batch, reset, batch}, bad.operations)
	})
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
	a, b := &recordingConsumer{name: "cache"}, &recordingConsumer{name: "sessions"}
	require.NoError(t, s.AddConsumer(a.TypedName(), a))
	require.NoError(t, s.AddConsumer(b.TypedName(), b))
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine"}, Address: "127.0.0.1", Port: "8000"}, nil)
	event := fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}
	for range 2 {
		require.NoError(t, s.Extract(ctx, event))
	}
	pods, _ := s.subscribers.GetActiveSubscribers()
	require.Len(t, pods, 1)
	late := &recordingConsumer{name: "late"}
	require.ErrorContains(t, s.AddConsumer(late.TypedName(), late), "after endpoint discovery started")
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
	err = s.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: ep})
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

func TestExtractDoesNotWaitForConsumers(t *testing.T) {
	s := &Source{subscribers: events.NewSubscriberManager(nil)}
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)
	event := fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}
	err := s.Extract(t.Context(), event)
	require.NoError(t, err)
	// A consumer processing a batch holds the read lock.
	s.mu.RLock()
	defer s.mu.RUnlock()
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = s.Extract(t.Context(), event)
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
	cache := &recordingConsumer{name: "cache"}
	require.NoError(t, s.AddConsumer(cache.TypedName(), cache))
	for _, name := range []string{"engine-b", "engine-a"} {
		ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name}, Address: "127.0.0.1", Port: "8000"}, nil)
		err := s.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep})
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
	bad, good := &recordingConsumer{name: "bad"}, &recordingConsumer{name: "good"}
	require.NoError(t, s.AddConsumer(bad.TypedName(), bad))
	require.NoError(t, s.AddConsumer(good.TypedName(), good))
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine"}, Address: "127.0.0.1", Port: "8000"}, nil)
	err = s.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep})
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
