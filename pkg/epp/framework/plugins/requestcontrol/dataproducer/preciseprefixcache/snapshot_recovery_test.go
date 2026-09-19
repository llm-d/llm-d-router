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
	"encoding/binary"
	"net"
	"strconv"
	"sync"
	"testing"
	"time"

	zmq "github.com/go-zeromq/zmq4"
	dl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	rh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	scorer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
	k8stypes "k8s.io/apimachinery/pkg/types"
)

type recoveryPublisher struct {
	topic                  string
	mu                     sync.Mutex
	pub                    zmq.Socket
	livePort, snapshotPort int
	seq                    uint64
	epoch                  byte
	chunks                 [][]byte
	empty                  []byte
	paused                 bool
	unavailable            bool
	hold                   <-chan struct{}
	requests               int
}

func newRecoveryPublisher(ctx context.Context, t *testing.T, host string, livePort, snapshotPort int) *recoveryPublisher {
	t.Helper()
	pub, rep := zmq.NewPub(ctx), zmq.NewRep(ctx)
	p := &recoveryPublisher{pub: pub, livePort: livePort, snapshotPort: snapshotPort, epoch: 1, topic: "kv@" + host + ":8000@test-model"}
	require.NoError(t, pub.Listen("tcp://"+net.JoinHostPort(host, strconv.Itoa(p.livePort))))
	require.NoError(t, rep.Listen("tcp://"+net.JoinHostPort(host, strconv.Itoa(p.snapshotPort))))
	p.empty = snapshotBatch(t)
	p.chunks = [][]byte{snapshotBatch(t, map[string]any{"type": "BlockStored", "block_hashes": []uint64{101}, "parent_block_hash": nil, "token_ids": []uint32{1, 2, 3, 4}, "block_size": 4})}
	var wg sync.WaitGroup
	wg.Add(2)
	t.Cleanup(func() { pub.Close(); rep.Close(); wg.Wait() })
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				p.mu.Lock()
				if !p.paused {
					p.send(p.empty)
				}
				p.mu.Unlock()
			}
		}
	}()
	go func() {
		defer wg.Done()
		for {
			_, err := rep.Recv()
			if err != nil {
				return
			}
			p.mu.Lock()
			p.requests++
			hold := p.hold
			cut := p.seq
			if p.unavailable {
				cut = ^uint64(1)
			}
			seq := make([]byte, 8)
			binary.BigEndian.PutUint64(seq, cut)
			epoch := make([]byte, 16)
			epoch[0] = p.epoch
			frames := [][]byte{seq, epoch}
			if !p.unavailable {
				frames = append(frames, p.chunks...)
			}
			p.mu.Unlock()
			if hold != nil {
				select {
				case <-hold:
				case <-ctx.Done():
					return
				}
			}
			if rep.Send(zmq.NewMsgFrom(frames...)) != nil {
				return
			}
		}
	}()
	return p
}

func snapshotBatch(t *testing.T, events ...any) []byte {
	t.Helper()
	if events == nil {
		events = []any{}
	}
	data, err := msgpack.Marshal([]any{1.0, events, nil})
	require.NoError(t, err)
	return data
}
func (p *recoveryPublisher) send(payload []byte) {
	p.seq++
	seq := make([]byte, 24)
	binary.BigEndian.PutUint64(seq, p.seq)
	seq[8] = p.epoch
	_ = p.pub.Send(zmq.NewMsgFrom([]byte(p.topic), seq, payload))
}

func TestSnapshotRecoveryGatesRealScorer(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stableAddress := "::1"
	server := newRecoveryPublisher(ctx, t, "127.0.0.1", snapshotTestPort(t), snapshotTestPort(t))
	newRecoveryPublisher(ctx, t, stableAddress, server.livePort, server.snapshotPort)
	// Cancel before the fixture waits for its heartbeat goroutine.
	t.Cleanup(cancel)
	cfg := kvevents.DefaultConfig()
	cfg.SnapshotPort = server.snapshotPort
	cfg.PodDiscoveryConfig.SocketPort = server.livePort
	indexCfg, err := kvcache.NewDefaultConfig()
	require.NoError(t, err)
	p, err := New(ctx, "recovery", PluginConfig{IndexerConfig: indexCfg, KVEventsConfig: cfg, TokenProcessorConfig: &kvblock.TokenProcessorConfig{BlockSizeTokens: 4}})
	require.NoError(t, err)
	defer p.subscribersManager.Shutdown(ctx)
	ep := scheduling.NewEndpoint(&dl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod"}, Address: "127.0.0.1", Port: "8000"}, nil, nil)
	require.NoError(t, p.Extract(ctx, dl.EndpointEvent{Type: dl.EventAddOrUpdate, Endpoint: dl.NewEndpoint(ep.GetMetadata(), nil)}))
	stable := scheduling.NewEndpoint(&dl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "stable-pod"}, Address: stableAddress, Port: "8000"}, nil, nil)
	require.NoError(t, p.Extract(ctx, dl.EndpointEvent{Type: dl.EventAddOrUpdate, Endpoint: dl.NewEndpoint(stable.GetMetadata(), nil)}))
	req := &scheduling.InferenceRequest{TargetModel: "test-model", Body: &rh.InferenceRequestBody{TokenizedRequest: &rh.TokenizedRequest{Prompts: []rh.PromptTokens{{TokenIDs: []uint32{1, 2, 3, 4}}}}}}
	prefix, err := scorer.New(ctx, "prefix", "recovery")
	require.NoError(t, err)
	score := func() float64 {
		require.NoError(t, p.Produce(ctx, req, []scheduling.Endpoint{ep, stable}))
		scores := prefix.Score(ctx, req, []scheduling.Endpoint{ep, stable})
		return scores[ep]
	}
	check := func(t *testing.T, want float64) {
		t.Helper()
		require.Eventually(t, func() bool { return score() == want }, 7*time.Second, 20*time.Millisecond)
		if want == 0 {
			scores := prefix.Score(ctx, req, []scheduling.Endpoint{ep, stable})
			picked := maxscore.NewMaxScorePicker(1).Pick(ctx, []*scheduling.ScoredEndpoint{{Endpoint: ep, Score: scores[ep]}, {Endpoint: stable, Score: scores[stable]}})
			require.Len(t, picked.TargetEndpoints, 1)
			require.Equal(t, stable.GetMetadata(), picked.TargetEndpoints[0].GetMetadata(), "picker must select the healthy cached endpoint during recovery")
		}
	}
	check(t, 1)
	require.Eventually(t, func() bool { score(); return prefix.Score(ctx, req, []scheduling.Endpoint{stable})[stable] == 1 }, 3*time.Second, 20*time.Millisecond)
	for _, fault := range []string{"gap", "epoch", "malformed-live", "decode-failure", "missing-parent", "heartbeat-timeout"} {
		t.Run(fault, func(t *testing.T) {
			held := make(chan struct{})
			server.mu.Lock()
			server.hold = held
			before := server.requests
			switch fault {
			case "gap":
				server.seq++
				server.send(server.empty)
			case "epoch":
				server.epoch++
				server.seq = 0
				server.send(server.empty)
			case "malformed-live":
				_ = server.pub.Send(zmq.NewMsgFrom([]byte("kv@127.0.0.1:8000@test-model")))
			case "decode-failure":
				server.send([]byte{0xc1})
			case "missing-parent":
				server.send(snapshotBatch(t, map[string]any{"type": "BlockStored", "block_hashes": []uint64{202}, "parent_block_hash": uint64(999), "token_ids": []uint32{5, 6, 7, 8}, "block_size": 4}))
			case "heartbeat-timeout":
				server.paused = true
			}
			server.mu.Unlock()
			check(t, 0)
			require.Equal(t, 1.0, prefix.Score(ctx, req, []scheduling.Endpoint{stable})[stable], "unaffected publisher must remain routable with cache affinity")
			// Rebuilds must not expose partial cache affinity.
			require.Never(t, func() bool { return score() != 0 }, 100*time.Millisecond, 10*time.Millisecond)
			server.mu.Lock()
			server.paused = false
			server.mu.Unlock()
			require.Eventually(t, func() bool { server.mu.Lock(); defer server.mu.Unlock(); return server.requests > before }, 4*time.Second, 20*time.Millisecond)
			close(held)
			server.mu.Lock()
			server.hold = nil
			server.mu.Unlock()
			check(t, 1)
		})
	}
	t.Run("unavailable", func(t *testing.T) {
		server.mu.Lock()
		server.unavailable = true
		server.seq++
		server.send(server.empty)
		before := server.requests
		server.mu.Unlock()
		check(t, 0)
		require.Equal(t, 1.0, prefix.Score(ctx, req, []scheduling.Endpoint{stable})[stable], "unaffected publisher must remain routable with cache affinity")
		require.Eventually(t, func() bool { server.mu.Lock(); defer server.mu.Unlock(); return server.requests >= before+2 }, 4*time.Second, 20*time.Millisecond)
		require.Equal(t, 0.0, score())
		server.mu.Lock()
		server.unavailable = false
		server.mu.Unlock()
		check(t, 1)
	})
	t.Run("malformed-snapshot", func(t *testing.T) {
		server.mu.Lock()
		original := server.chunks
		server.chunks = [][]byte{{0xc1}}
		server.seq++
		server.send(server.empty)
		before := server.requests
		server.mu.Unlock()
		check(t, 0)
		require.Equal(t, 1.0, prefix.Score(ctx, req, []scheduling.Endpoint{stable})[stable], "unaffected publisher must remain routable with cache affinity")
		require.Eventually(t, func() bool { server.mu.Lock(); defer server.mu.Unlock(); return server.requests > before }, 4*time.Second, 20*time.Millisecond)
		require.Equal(t, 0.0, score())
		server.mu.Lock()
		server.chunks = original
		server.mu.Unlock()
		check(t, 1)
	})
}
