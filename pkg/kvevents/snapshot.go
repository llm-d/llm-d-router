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
	"errors"
	"fmt"
	"math/rand/v2"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	zmq "github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	snapshotTimeout     = 10 * time.Second
	heartbeatTimeout    = 5 * time.Second
	snapshotBufferBytes = 64 << 20
	snapshotReplyBytes  = 256 << 20
	retryInitial        = time.Second
	retryMaximum        = 30 * time.Second
)

// SnapshotManager owns a shared index with replaceable publisher generations.
// Only complete generations with a consecutive live suffix are visible to the matcher.
type SnapshotManager struct {
	matcher      *kvcache.Indexer
	index        kvblock.Index
	mu           sync.RWMutex
	subscribers  map[string]*snapshotSubscriber
	livePort     int
	snapshotPort int
	tokens       kvblock.TokenProcessor
	adapter      EngineAdapter
	nextGen      atomic.Uint64
}

type snapshotSubscriber struct {
	manager                                 *SnapshotManager
	endpoint, snapshotEndpoint, topicFilter string
	sourceEndpoint                          string
	cancel                                  context.CancelFunc
	done                                    chan struct{}
	mu                                      sync.RWMutex
	generationID                            string
	activeID                                string
	lastReceive                             atomic.Int64
	successes                               atomic.Uint64
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
	index, err := kvblock.NewInMemoryIndex(indexCfg.InMemoryConfig)
	if err != nil {
		return nil, err
	}
	if matcher != nil {
		matcher = matcher.WithIndex(index)
	}
	metrics.Register()
	return &SnapshotManager{
		subscribers:  make(map[string]*snapshotSubscriber),
		livePort:     cfg.PodDiscoveryConfig.SocketPort,
		snapshotPort: cfg.SnapshotPort,
		index:        index,
		tokens:       tokens,
		adapter:      adapter,
		matcher:      matcher,
	}, nil
}

func (m *SnapshotManager) EnsureSubscriber(ctx context.Context, id, sourceEndpoint, endpoint, replayEndpoint, topic string, remote bool) error {
	if sourceEndpoint == "" || replayEndpoint != "" {
		return fmt.Errorf("snapshot recovery requires an identified endpoint without replay")
	}
	if !remote || !strings.HasPrefix(endpoint, "tcp://") {
		return fmt.Errorf("snapshot recovery requires a remote TCP publisher")
	}
	host, livePortText, err := net.SplitHostPort(strings.TrimPrefix(endpoint, "tcp://"))
	if err != nil {
		return err
	}
	livePort, err := strconv.Atoi(livePortText)
	if err != nil {
		return fmt.Errorf("invalid live publisher port %q: %w", livePortText, err)
	}
	rankIndex := livePort - m.livePort
	snapshotPort := m.snapshotPort + rankIndex
	if rankIndex < 0 || snapshotPort > 65535 {
		return fmt.Errorf("live publisher port %d is outside the snapshot port range", livePort)
	}
	m.mu.Lock()
	if old := m.subscribers[id]; old != nil {
		if old.endpoint == endpoint && old.topicFilter == topic && old.sourceEndpoint == sourceEndpoint {
			m.mu.Unlock()
			return nil
		}
		old.cancel()
		delete(m.subscribers, id)
	}
	subCtx, cancel := context.WithCancel(ctx)
	generationID := fmt.Sprintf("%s#snapshot-%d", sourceEndpoint, m.nextGen.Add(1))
	s := &snapshotSubscriber{manager: m, endpoint: endpoint, sourceEndpoint: sourceEndpoint, snapshotEndpoint: "tcp://" + net.JoinHostPort(host, strconv.Itoa(snapshotPort)), topicFilter: topic, cancel: cancel, done: make(chan struct{}), generationID: generationID}
	m.subscribers[id] = s
	registered := len(m.subscribers)
	m.mu.Unlock()
	metrics.SubscriberActive.Set(float64(registered))
	go s.run(subCtx)
	return nil
}

func (m *SnapshotManager) RemoveSubscriber(_ context.Context, id string) bool {
	m.mu.Lock()
	if s := m.subscribers[id]; s != nil {
		s.cancel()
		delete(m.subscribers, id)
		registered := len(m.subscribers)
		m.mu.Unlock()
		metrics.SubscriberActive.Set(float64(registered))
		m.updateReadyMetric()
		return true
	}
	m.mu.Unlock()
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
	metrics.SubscriberActive.Set(0)
	metrics.SnapshotReady.Set(0)
}

// SnapshotStatus reports publisher recovery states.
type SnapshotStatus struct {
	Registered int
	Ready      int
	Recovering int
	Stale      int
}

// Status returns current publisher recovery counts.
func (m *SnapshotManager) Status() SnapshotStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()
	status := SnapshotStatus{Registered: len(m.subscribers)}
	for _, s := range m.subscribers {
		s.mu.RLock()
		active := s.activeID != ""
		fresh := time.Since(time.Unix(0, s.lastReceive.Load())) <= heartbeatTimeout
		s.mu.RUnlock()
		switch {
		case active && fresh:
			status.Ready++
		case active:
			status.Stale++
		default:
			status.Recovering++
		}
	}
	return status
}

func (m *SnapshotManager) updateReadyMetric() {
	metrics.SnapshotReady.Set(float64(m.Status().Ready))
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

// GetReadySubscribers returns publishers with a complete current index.
func (m *SnapshotManager) GetReadySubscribers() ([]string, []string) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	ids, endpoints := []string{}, []string{}
	for id, s := range m.subscribers {
		s.mu.RLock()
		ready := s.activeID != "" && time.Since(time.Unix(0, s.lastReceive.Load())) <= heartbeatTimeout
		s.mu.RUnlock()
		if !ready {
			continue
		}
		ids = append(ids, id)
		endpoints = append(endpoints, s.endpoint)
	}
	return ids, endpoints
}

func (m *SnapshotManager) MatchBlockKeys(ctx context.Context, keys []kvblock.BlockHash, pods sets.Set[string]) (map[string]kvcache.PodMatch, error) {
	m.mu.RLock()
	active := make(map[string]string, len(m.subscribers))
	for _, s := range m.subscribers {
		if pods.Len() > 0 && !pods.Has(s.sourceEndpoint) {
			continue
		}
		s.mu.RLock()
		if s.activeID != "" && time.Since(time.Unix(0, s.lastReceive.Load())) <= heartbeatTimeout {
			active[s.activeID] = s.sourceEndpoint
		}
		s.mu.RUnlock()
	}
	m.mu.RUnlock()
	if len(active) == 0 {
		return map[string]kvcache.PodMatch{}, nil
	}
	matches, err := m.matcher.MatchBlockKeys(ctx, keys, sets.KeySet(active))
	if err != nil {
		return nil, err
	}
	result := make(map[string]kvcache.PodMatch, len(matches))
	for generation, match := range matches {
		result[active[generation]] = match
	}
	return result, nil
}

func (s *snapshotSubscriber) deactivate(generation string) {
	s.mu.Lock()
	changed := false
	if s.activeID == generation {
		s.activeID = ""
		changed = true
	}
	s.mu.Unlock()
	if changed {
		s.manager.updateReadyMetric()
	}
}

func (s *snapshotSubscriber) run(ctx context.Context) {
	defer close(s.done)
	backoff := retryInitial
	for ctx.Err() == nil {
		successes := s.successes.Load()
		err := s.consume(ctx)
		if ctx.Err() != nil {
			return
		}
		metrics.SnapshotRecoveries.WithLabelValues("failure", snapshotFailureReason(err)).Inc()
		log.FromContext(ctx).Error(err, "KV snapshot recovery failed", "endpoint", s.endpoint)
		if s.successes.Load() != successes {
			backoff = retryInitial
		}
		delay := backoff/2 + time.Duration(rand.Int64N(int64(backoff/2)))
		if s.successes.Load() == successes {
			backoff = min(2*backoff, retryMaximum)
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func snapshotFailureReason(err error) string {
	if errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "timeout") {
		return "timeout"
	}
	message := err.Error()
	switch {
	case strings.Contains(message, "unavailable"):
		return "unavailable"
	case strings.Contains(message, "sequence gap"), strings.Contains(message, "identity changed"), strings.Contains(message, "publisher changed"):
		return "stream"
	case strings.Contains(message, "malformed"), strings.Contains(message, "parse"), strings.Contains(message, "decode"), strings.Contains(message, "engine key not found"):
		return "invalid"
	default:
		return "transport"
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
	bootstrapStarted := time.Now()
	ctx, cancel := context.WithCancel(parent)
	generation := s.generationID
	defer func() {
		s.deactivate(generation)
		if err := s.manager.index.Clear(context.Background(), generation); err != nil {
			log.FromContext(parent).Error(err, "Failed to clear KV snapshot generation", "endpoint", s.endpoint)
		}
	}()
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
			s.deactivate(generation)
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
	pool, err := NewPool(&Config{Concurrency: 1}, s.manager.index, s.manager.tokens, s.manager.adapter)
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
		return pool.processEventBatch(ctx, &batch, generation, model)
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
	s.activeID = generation
	s.successes.Add(1)
	s.mu.Unlock()
	s.manager.updateReadyMetric()
	metrics.SnapshotRecoveries.WithLabelValues("success", "none").Inc()
	metrics.SnapshotBootstrapDuration.Observe(time.Since(bootstrapStarted).Seconds())
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		msg, err := take()
		if err != nil {
			return err
		}
		err = advance(msg)
		if err != nil {
			s.deactivate(generation)
		}
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
