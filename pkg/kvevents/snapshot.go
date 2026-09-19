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

package kvevents

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	zmq "github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	snapshotTimeout     = 10 * time.Second
	heartbeatTimeout    = 5 * time.Second
	snapshotBufferBytes = 64 << 20
	snapshotReplyBytes  = 256 << 20
)

// SnapshotManager owns an independent, replaceable event index per publisher.
// Only complete snapshots with a consecutive live suffix are visible to the matcher.
type SnapshotManager struct {
	matcher     *kvcache.Indexer
	mu          sync.RWMutex
	subscribers map[string]*snapshotSubscriber
	port        int
	indexConfig kvblock.InMemoryIndexConfig
	tokens      kvblock.TokenProcessor
	adapter     EngineAdapter
}

type snapshotSubscriber struct {
	manager                                 *SnapshotManager
	endpoint, snapshotEndpoint, topicFilter string
	sourceEndpoint                          string
	cancel                                  context.CancelFunc
	done                                    chan struct{}
	mu                                      sync.RWMutex
	index                                   kvblock.Index
	lastReceive                             atomic.Int64
}

func NewSnapshotManager(cfg *Config, indexCfg *kvblock.IndexConfig, tokens kvblock.TokenProcessor, adapter EngineAdapter, matcher *kvcache.Indexer) (*SnapshotManager, error) {
	if err := registerSnapshotTransport(); err != nil {
		return nil, err
	}
	if cfg.SnapshotPort < 1 || cfg.SnapshotPort > 65535 {
		return nil, fmt.Errorf("invalid snapshotPort: %d", cfg.SnapshotPort)
	}
	if indexCfg == nil {
		indexCfg = kvblock.DefaultIndexConfig()
	}
	if indexCfg.InMemoryConfig == nil || indexCfg.RedisConfig != nil || indexCfg.CostAwareMemoryConfig != nil {
		return nil, fmt.Errorf("snapshot recovery requires the in-memory index")
	}
	return &SnapshotManager{subscribers: make(map[string]*snapshotSubscriber), port: cfg.SnapshotPort, indexConfig: *indexCfg.InMemoryConfig, tokens: tokens, adapter: adapter, matcher: matcher}, nil
}

func (m *SnapshotManager) EnsureSubscriber(ctx context.Context, id, sourceEndpoint, endpoint, replayEndpoint, topic string, remote bool) error {
	if sourceEndpoint == "" || replayEndpoint != "" {
		return fmt.Errorf("snapshot recovery requires an identified endpoint without replay")
	}
	if !remote || !strings.HasPrefix(endpoint, "tcp://") {
		return fmt.Errorf("snapshot recovery requires a remote TCP publisher")
	}
	host, _, err := net.SplitHostPort(strings.TrimPrefix(endpoint, "tcp://"))
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if old := m.subscribers[id]; old != nil {
		if old.endpoint == endpoint && old.topicFilter == topic && old.sourceEndpoint == sourceEndpoint {
			return nil
		}
		old.cancel()
		delete(m.subscribers, id)
	}
	subCtx, cancel := context.WithCancel(ctx)
	s := &snapshotSubscriber{manager: m, endpoint: endpoint, sourceEndpoint: sourceEndpoint, snapshotEndpoint: "tcp://" + net.JoinHostPort(host, strconv.Itoa(m.port)), topicFilter: topic, cancel: cancel, done: make(chan struct{})}
	m.subscribers[id] = s
	go s.run(subCtx)
	return nil
}

func (m *SnapshotManager) RemoveSubscriber(_ context.Context, id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if s := m.subscribers[id]; s != nil {
		s.cancel()
		delete(m.subscribers, id)
		return true
	}
	return false
}

func (m *SnapshotManager) Shutdown(_ context.Context) {
	m.mu.Lock()
	old := m.subscribers
	m.subscribers = make(map[string]*snapshotSubscriber)
	m.mu.Unlock()
	for _, s := range old {
		s.cancel()
	}
	for _, s := range old {
		<-s.done
	}
}

func (m *SnapshotManager) GetActiveSubscribers() ([]string, []string) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	ids, endpoints := []string{}, []string{}
	for id, s := range m.subscribers {
		ids = append(ids, id)
		endpoints = append(endpoints, s.endpoint)
	}
	return ids, endpoints
}

func (m *SnapshotManager) MatchBlockKeys(ctx context.Context, keys []kvblock.BlockHash, pods sets.Set[string]) (map[string]kvcache.PodMatch, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make(map[string]kvcache.PodMatch)
	for _, s := range m.subscribers {
		if pods.Len() > 0 && !pods.Has(s.sourceEndpoint) {
			continue
		}
		s.mu.RLock()
		if s.index == nil || time.Since(time.Unix(0, s.lastReceive.Load())) > heartbeatTimeout {
			s.mu.RUnlock()
			continue
		}
		matches, err := m.matcher.WithIndex(s.index).MatchBlockKeys(ctx, keys, pods)
		s.mu.RUnlock()
		if err != nil {
			return nil, err
		}
		for pod, match := range matches {
			result[pod] = match
		}
	}
	return result, nil
}

func (s *snapshotSubscriber) invalidate() { s.mu.Lock(); s.index = nil; s.mu.Unlock() }
func (s *snapshotSubscriber) run(ctx context.Context) {
	defer close(s.done)
	defer s.invalidate()
	for ctx.Err() == nil {
		err := s.consume(ctx)
		s.invalidate()
		if ctx.Err() != nil {
			return
		}
		log.FromContext(ctx).Error(err, "KV snapshot recovery failed", "endpoint", s.endpoint)
		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Second):
		}
	}
}

type snapshotLive struct {
	topic    string
	sequence uint64
	epoch    []byte
	payload  []byte
}

func decodeSnapshotLive(msg zmq.Msg) (snapshotLive, error) {
	if len(msg.Frames) != 3 || len(msg.Frames[1]) != 24 {
		return snapshotLive{}, fmt.Errorf("expected snapshot-enabled live frames")
	}
	return snapshotLive{string(msg.Frames[0]), binary.BigEndian.Uint64(msg.Frames[1][:8]), msg.Frames[1][8:], msg.Frames[2]}, nil
}

func (s *snapshotSubscriber) consume(parent context.Context) error {
	ctx, cancel := context.WithCancel(parent)
	sub := zmq.NewSub(ctx, zmq.WithDialerMaxRetries(0), zmq.WithDialerTimeout(time.Second))
	defer sub.Close()
	defer cancel()
	if err := sub.SetOption(zmq.OptionSubscribe, s.topicFilter); err != nil {
		return err
	}
	if err := sub.Dial(snapshotTCP(s.endpoint)); err != nil {
		return err
	}
	live := make(chan snapshotLive, 256)
	failures := make(chan error, 1)
	var queuedBytes atomic.Int64
	receiverDone := make(chan struct{})
	defer func() { cancel(); sub.Close(); <-receiverDone }()
	go func() {
		defer close(receiverDone)
		fail := func(err error) {
			s.invalidate()
			select {
			case failures <- err:
			default:
			}
			cancel()
		}
		for {
			msg, err := sub.Recv()
			if err != nil {
				fail(err)
				return
			}
			event, err := decodeSnapshotLive(msg)
			if err != nil {
				fail(err)
				return
			}
			s.lastReceive.Store(time.Now().UnixNano())
			if queuedBytes.Add(int64(len(event.payload))) > snapshotBufferBytes {
				fail(fmt.Errorf("live buffer byte limit exceeded"))
				return
			}
			select {
			case live <- event:
			case <-ctx.Done():
				return
			default:
				fail(fmt.Errorf("live buffer message limit exceeded"))
				return
			}
		}
	}()
	take := func() (snapshotLive, error) {
		select {
		case msg := <-live:
			queuedBytes.Add(-int64(len(msg.payload)))
			return msg, nil
		case err := <-failures:
			return snapshotLive{}, err
		case <-ctx.Done():
			return snapshotLive{}, ctx.Err()
		case <-time.After(heartbeatTimeout):
			return snapshotLive{}, fmt.Errorf("publisher heartbeat timeout")
		}
	}
	first, err := take()
	if err != nil {
		return err
	}
	if !strings.HasPrefix(first.topic, "kv@") || strings.Count(first.topic, "@") != 2 {
		return fmt.Errorf("snapshot recovery requires kv@<address:port>@<model> topic")
	}
	reqCtx, reqCancel := context.WithTimeout(ctx, snapshotTimeout)
	defer reqCancel()
	req := zmq.NewReq(reqCtx, zmq.WithDialerMaxRetries(0), zmq.WithDialerTimeout(time.Second), zmq.WithTimeout(snapshotTimeout))
	defer req.Close()
	if err := req.Dial(snapshotTCP(s.snapshotEndpoint)); err != nil {
		return err
	}
	if err := req.Send(zmq.NewMsg([]byte("snapshot"))); err != nil {
		return err
	}
	type reply struct {
		msg zmq.Msg
		err error
	}
	replies := make(chan reply, 1)
	go func() { msg, err := req.Recv(); replies <- reply{msg, err} }()
	buffered := []snapshotLive{first}
	bufferBytes := len(first.payload)
	var response zmq.Msg
waiting:
	for {
		select {
		case r := <-replies:
			if r.err != nil {
				return r.err
			}
			response = r.msg
			break waiting
		case msg := <-live:
			queuedBytes.Add(-int64(len(msg.payload)))
			bufferBytes += len(msg.payload)
			if bufferBytes > snapshotBufferBytes || len(buffered) >= 4096 {
				return fmt.Errorf("bootstrap buffer exhausted")
			}
			buffered = append(buffered, msg)
		case <-reqCtx.Done():
			return reqCtx.Err()
		}
	}
	f := response.Frames
	if len(f) < 2 || len(f[0]) != 8 || len(f[1]) != 16 {
		return fmt.Errorf("malformed snapshot reply")
	}
	// Signed -1 is the initial empty cut; -2 and lower mean unavailable.
	cut := int64(binary.BigEndian.Uint64(f[0]))
	if cut < -1 {
		return fmt.Errorf("snapshot unavailable (%d)", cut)
	}
	if !bytes.Equal(f[1], first.epoch) {
		return fmt.Errorf("publisher changed during bootstrap")
	}
	size := 0
	for _, chunk := range f[2:] {
		size += len(chunk)
	}
	if size > snapshotReplyBytes {
		return fmt.Errorf("snapshot reply limit exceeded")
	}
	idx, err := kvblock.NewInMemoryIndex(&s.manager.indexConfig)
	if err != nil {
		return err
	}
	pool, err := NewPool(&Config{Concurrency: 1}, idx, s.manager.tokens, s.manager.adapter)
	if err != nil {
		return err
	}
	defer pool.queues[0].ShutDown()
	pool.strict = true
	apply := func(payload []byte) error {
		_, model, batch, err := s.manager.adapter.ParseMessage(&RawMessage{Topic: first.topic, Payload: payload})
		if err != nil {
			return err
		}
		return pool.processEventBatch(ctx, &batch, s.sourceEndpoint, model)
	}
	for _, chunk := range f[2:] {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err := apply(chunk); err != nil {
			return err
		}
	}
	next := uint64(cut + 1)
	advance := func(msg snapshotLive) error {
		if msg.topic != first.topic || !bytes.Equal(msg.epoch, f[1]) {
			return fmt.Errorf("publisher identity changed")
		}
		if msg.sequence < next {
			return nil
		}
		if msg.sequence != next {
			return fmt.Errorf("live sequence gap: got %d, want %d", msg.sequence, next)
		}
		if err := apply(msg.payload); err != nil {
			return err
		}
		next++
		return nil
	}
	for _, msg := range buffered {
		if err := advance(msg); err != nil {
			return err
		}
	}
	// A finite cut prevents a busy publisher from indefinitely delaying install.
	for remaining := len(live); remaining > 0; remaining-- {
		msg, err := take()
		if err != nil {
			return err
		}
		if err := advance(msg); err != nil {
			return err
		}
	}
	s.mu.Lock()
	if ctx.Err() != nil {
		s.mu.Unlock()
		return ctx.Err()
	}
	s.index = idx
	s.mu.Unlock()
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		msg, err := take()
		if err != nil {
			return err
		}
		s.mu.Lock()
		err = advance(msg)
		if err != nil {
			s.index = nil
		}
		s.mu.Unlock()
		if err != nil {
			return err
		}
	}
}

// GPU resets preserve offloaded residency and its reconstruction mappings.
func (p *Pool) clearSnapshotGPU(ctx context.Context, pod string) error {
	p.dedup.mu.Lock()
	defer p.dedup.mu.Unlock()
	for key := range p.dedup.refs[pod] {
		if key.deviceTier != "gpu" {
			continue
		}
		entry := kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "gpu"}
		if key.groupIdx != noGroupIdx {
			entry.HasGroup = true
			entry.GroupIdx = kvblock.GroupID(key.groupIdx)
		}
		if err := p.index.Evict(ctx, kvblock.BlockHash(key.blockHash), kvblock.EngineKey, []kvblock.PodEntry{entry}); err != nil {
			return err
		}
		delete(p.dedup.refs[pod], key)
	}
	return nil
}
