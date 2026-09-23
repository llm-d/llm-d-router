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
	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	snapshotTimeout       = 10 * time.Second
	heartbeatTimeout      = 5 * time.Second
	snapshotBufferBytes   = 64 << 20
	snapshotReplyBytes    = 256 << 20
	retryInitial          = time.Second
	retryMaximum          = 30 * time.Second
	maxConcurrentRecovery = 4
	// Snapshot generations use one entry per publisher, tier, and cache group.
	// Keep the shared index well above the ordinary request index's small
	// per-key routing cache so recovery never silently drops publishers.
	snapshotEntriesPerKey = 1 << 20
	// Eviction is fail closed: a later event that references metadata older than
	// this bounded history deactivates the generation and starts a fresh snapshot.
	snapshotEngineMappings = 1 << 20
)

// SnapshotManager owns a shared index with replaceable publisher generations.
// Only complete generations with a consecutive live suffix are visible to the matcher.
type SnapshotManager struct {
	matcher       *kvcache.Indexer
	index         kvblock.Index
	mu            sync.RWMutex
	metricsMu     sync.Mutex
	subscribers   map[string]*snapshotSubscriber
	livePort      int
	snapshotPort  int
	tokens        kvblock.TokenProcessor
	adapter       EngineAdapter
	nextGen       atomic.Uint64
	recoverySlots chan struct{}
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
	liveOnly                                bool
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
	snapshotIndexCfg := *indexCfg.InMemoryConfig
	if snapshotIndexCfg.PodCacheSize < snapshotEntriesPerKey {
		snapshotIndexCfg.PodCacheSize = snapshotEntriesPerKey
	}
	index, err := kvblock.NewInMemoryIndex(&snapshotIndexCfg)
	if err != nil {
		return nil, err
	}
	if matcher != nil {
		matcher = matcher.WithIndex(index)
	}
	metrics.Register()
	return &SnapshotManager{
		subscribers:   make(map[string]*snapshotSubscriber),
		livePort:      cfg.PodDiscoveryConfig.SocketPort,
		snapshotPort:  cfg.SnapshotPort,
		index:         index,
		tokens:        tokens,
		adapter:       adapter,
		matcher:       matcher,
		recoverySlots: make(chan struct{}, maxConcurrentRecovery),
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
	m.mu.Unlock()
	m.updateSubscriberMetric()
	go s.run(subCtx)
	return nil
}

func (m *SnapshotManager) RemoveSubscriber(_ context.Context, id string) bool {
	m.mu.Lock()
	if s := m.subscribers[id]; s != nil {
		s.cancel()
		delete(m.subscribers, id)
		m.mu.Unlock()
		m.updateSubscriberMetric()
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
	metrics.LiveOnlyPublishers.Set(0)
}

// SnapshotStatus reports publisher recovery states.
type SnapshotStatus struct {
	Registered int
	Ready      int
	LiveOnly   int
	Recovering int
	Stale      int
}

// Status returns current publisher recovery counts.
func (m *SnapshotManager) Status() SnapshotStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()
	status := SnapshotStatus{Registered: len(m.subscribers)}
	for _, s := range m.subscribers {
		switch state, _ := s.state(); state {
		case publisherReady:
			status.Ready++
		case publisherLiveOnly:
			status.LiveOnly++
		case publisherStale:
			status.Stale++
		default:
			status.Recovering++
		}
	}
	return status
}

func (m *SnapshotManager) updateReadyMetric() {
	m.metricsMu.Lock()
	defer m.metricsMu.Unlock()
	status := m.Status()
	metrics.SnapshotReady.Set(float64(status.Ready))
	metrics.LiveOnlyPublishers.Set(float64(status.LiveOnly))
}

func (m *SnapshotManager) updateSubscriberMetric() {
	m.metricsMu.Lock()
	defer m.metricsMu.Unlock()
	m.mu.RLock()
	registered := len(m.subscribers)
	m.mu.RUnlock()
	metrics.SubscriberActive.Set(float64(registered))
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
		if state, _ := s.state(); !state.routable() {
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
		if state, generation := s.state(); state.routable() {
			active[generation] = s.sourceEndpoint
		}
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

type publisherState int

const (
	publisherRecovering publisherState = iota
	// publisherReady has a complete snapshot generation and a recent heartbeat.
	publisherReady
	// publisherLiveOnly has no snapshot endpoint and is indexed from live events.
	publisherLiveOnly
	// publisherStale has a snapshot generation without a recent heartbeat.
	publisherStale
)

func (p publisherState) routable() bool { return p == publisherReady || p == publisherLiveOnly }

// state reports the publisher's lookup state and its active generation.
func (s *snapshotSubscriber) state() (publisherState, string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	switch {
	case s.activeID == "":
		return publisherRecovering, ""
	case s.liveOnly:
		return publisherLiveOnly, s.activeID
	case time.Since(time.Unix(0, s.lastReceive.Load())) <= heartbeatTimeout:
		return publisherReady, s.activeID
	}
	return publisherStale, s.activeID
}

func (s *snapshotSubscriber) activate(ctx context.Context, generation string, liveOnly bool) error {
	s.mu.Lock()
	if err := ctx.Err(); err != nil {
		s.mu.Unlock()
		return err
	}
	s.activeID = generation
	s.liveOnly = liveOnly
	s.successes.Add(1)
	s.mu.Unlock()
	s.manager.updateReadyMetric()
	return nil
}

func (s *snapshotSubscriber) deactivate(generation string) {
	s.mu.Lock()
	changed := false
	if s.activeID == generation {
		s.activeID = ""
		s.liveOnly = false
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
		select {
		case s.manager.recoverySlots <- struct{}{}:
		case <-ctx.Done():
			return
		}
		release := sync.OnceFunc(func() { <-s.manager.recoverySlots })
		successes := s.successes.Load()
		err := s.consume(ctx, release)
		release()
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

// snapshotEnabled reports whether the publisher serves snapshots: vLLM appends
// its 16-byte identity to the sequence frame, and heartbeats, only then.
func (m snapshotLive) snapshotEnabled() bool { return m.epoch != nil }

func decodeSnapshotLive(msg zmq.Msg) (snapshotLive, error) {
	if len(msg.Frames) != 3 {
		return snapshotLive{}, fmt.Errorf("malformed live frames")
	}
	switch len(msg.Frames[1]) {
	case 8:
		return snapshotLive{string(msg.Frames[0]), binary.BigEndian.Uint64(msg.Frames[1]), nil, msg.Frames[2]}, nil
	case 24:
		return snapshotLive{string(msg.Frames[0]), binary.BigEndian.Uint64(msg.Frames[1][:8]), msg.Frames[1][8:], msg.Frames[2]}, nil
	}
	return snapshotLive{}, fmt.Errorf("malformed live sequence frame (%d bytes)", len(msg.Frames[1]))
}

// follow applies live messages until the stream fails or step rejects one.
func follow(ctx context.Context, take func(time.Duration) (snapshotLive, error), timeout time.Duration,
	step func(snapshotLive) error,
) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		msg, err := take(timeout)
		if err != nil {
			return err
		}
		if err := step(msg); err != nil {
			return err
		}
	}
}

func (s *snapshotSubscriber) consume(parent context.Context, recoveryComplete func()) error {
	bootstrapStarted := time.Now()
	ctx, cancel := context.WithCancel(parent)
	generation := s.generationID
	defer func() {
		s.deactivate(generation)
	}()
	sub := zmq.NewSub(ctx, zmq.WithDialerMaxRetries(0), zmq.WithDialerTimeout(time.Second))
	defer sub.Close()
	defer cancel()
	if err := sub.SetOption(zmq.OptionSubscribe, s.topicFilter); err != nil {
		return err
	}
	// Live frames carry no deadline: idle publishers without a snapshot
	// endpoint are silent, and snapshot publishers are bounded by heartbeats.
	dialing := time.AfterFunc(snapshotTimeout, cancel)
	err := sub.Dial(snapshotTCP(s.endpoint))
	if !dialing.Stop() {
		return fmt.Errorf("dial %s: handshake timeout", s.endpoint)
	}
	if err != nil {
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
	// take waits for the next live message; a zero timeout waits indefinitely.
	take := func(timeout time.Duration) (snapshotLive, error) {
		var expired <-chan time.Time
		if timeout > 0 {
			expired = time.After(timeout)
		}
		select {
		case msg := <-live:
			queuedBytes.Add(-int64(len(msg.payload)))
			return msg, nil
		case err := <-failures:
			return snapshotLive{}, err
		case <-ctx.Done():
			return snapshotLive{}, ctx.Err()
		case <-expired:
			return snapshotLive{}, fmt.Errorf("publisher heartbeat timeout")
		}
	}
	// The first frame shows whether the engine serves snapshots. Only
	// snapshot-enabled engines heartbeat, so it has no deadline.
	first, err := take(0)
	if err != nil {
		return err
	}
	pool, err := newGenerationPool(s.manager.index, s.manager.tokens, s.manager.adapter)
	if err != nil {
		return err
	}
	defer pool.queues[0].ShutDown()
	defer func() {
		if err := pool.clearSnapshotGeneration(context.Background()); err != nil {
			log.FromContext(parent).Error(err, "Failed to clear KV snapshot generation", "endpoint", s.endpoint)
		}
	}()
	apply := func(msg snapshotLive) error {
		_, model, batch, err := s.manager.adapter.ParseMessage(&RawMessage{Topic: msg.topic, Payload: msg.payload})
		if err != nil {
			return err
		}
		return pool.processEventBatch(ctx, &batch, generation, model)
	}
	if !first.snapshotEnabled() {
		// Without a snapshot endpoint there is nothing to recover: index live
		// events as they arrive, as live mode does, and allow idle periods.
		if err := s.activate(ctx, generation, true); err != nil {
			return err
		}
		log.FromContext(ctx).Info("KV publisher has no snapshot endpoint, indexing live events only", "endpoint", s.endpoint)
		recoveryComplete()
		index := func(msg snapshotLive) error {
			if msg.snapshotEnabled() {
				return fmt.Errorf("publisher changed to snapshot-enabled frames")
			}
			if err := apply(msg); err != nil {
				log.FromContext(ctx).Error(err, "Failed to apply KV event batch", "endpoint", s.endpoint)
			}
			return nil
		}
		if err := index(first); err != nil {
			return err
		}
		return follow(ctx, take, 0, index)
	}
	pool.strict = true
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
	for _, chunk := range f[2:] {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err := apply(snapshotLive{topic: first.topic, payload: chunk}); err != nil {
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
		if err := apply(msg); err != nil {
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
		msg, err := take(heartbeatTimeout)
		if err != nil {
			return err
		}
		if err := advance(msg); err != nil {
			return err
		}
	}
	if err := s.activate(ctx, generation, false); err != nil {
		return err
	}
	metrics.SnapshotRecoveries.WithLabelValues("success", "none").Inc()
	metrics.SnapshotBootstrapDuration.Observe(time.Since(bootstrapStarted).Seconds())
	recoveryComplete()
	return follow(ctx, take, heartbeatTimeout, advance)
}

// newGenerationPool returns a single-worker pool whose index entries belong to
// one publisher generation and are removed by clearSnapshotGeneration.
func newGenerationPool(index kvblock.Index, tokens kvblock.TokenProcessor, adapter EngineAdapter) (*Pool, error) {
	pool, err := NewPool(&Config{Concurrency: 1}, index, tokens, adapter)
	if err != nil {
		return nil, err
	}
	pool.ownsEntries = true
	return pool, nil
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
		engineKey := kvblock.BlockHash(key.blockHash)
		requestKeys, err := p.snapshotRequestKeys(ctx, engineKey)
		if err != nil {
			return err
		}
		for _, requestKey := range requestKeys {
			if err := p.index.Evict(ctx, requestKey, kvblock.RequestKey, []kvblock.PodEntry{entry}); err != nil {
				return err
			}
		}
		p.untrackSnapshotStore(requestKeys, []kvblock.PodEntry{entry})
		delete(p.dedup.refs[pod], key)
	}
	return nil
}

func (p *Pool) snapshotRequestKeys(ctx context.Context, engineKey kvblock.BlockHash) ([]kvblock.BlockHash, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if p.snapshotEngineKeys == nil {
		return nil, fmt.Errorf("snapshot engine key not found: %s", engineKey.String())
	}
	keys, found := p.snapshotEngineKeys.Get(engineKey)
	if !found || len(keys) == 0 {
		return nil, fmt.Errorf("snapshot engine key not found: %s", engineKey.String())
	}
	return append([]kvblock.BlockHash(nil), keys...), nil
}

func (p *Pool) trackSnapshotEngineKeys(engineKeys, requestKeys []kvblock.BlockHash) error {
	if len(engineKeys) == 0 || len(requestKeys) == 0 {
		return fmt.Errorf("snapshot engine mapping requires engine and request keys")
	}
	if p.snapshotEngineKeys == nil {
		cache, err := lru.New[kvblock.BlockHash, []kvblock.BlockHash](snapshotEngineMappings)
		if err != nil {
			return err
		}
		p.snapshotEngineKeys = cache
	}
	mappings := make(map[kvblock.BlockHash][]kvblock.BlockHash, len(engineKeys))
	n := max(len(engineKeys), len(requestKeys))
	for i := range n {
		engineKey := engineKeys[i*len(engineKeys)/n]
		requestKey := requestKeys[i*len(requestKeys)/n]
		mappings[engineKey] = append(mappings[engineKey], requestKey)
	}
	for engineKey, keys := range mappings {
		p.snapshotEngineKeys.Add(engineKey, keys)
	}
	return nil
}

func (p *Pool) trackSnapshotStore(keys []kvblock.BlockHash, entries []kvblock.PodEntry) {
	if p.snapshotEntries == nil {
		p.snapshotEntries = make(map[snapshotOwnedEntry]struct{})
	}
	for _, key := range keys {
		for _, entry := range entries {
			p.snapshotEntries[snapshotOwnedEntry{key: key, entry: entry}] = struct{}{}
		}
	}
}

func (p *Pool) untrackSnapshotStore(keys []kvblock.BlockHash, entries []kvblock.PodEntry) {
	for _, key := range keys {
		for _, entry := range entries {
			delete(p.snapshotEntries, snapshotOwnedEntry{key: key, entry: entry})
		}
	}
}

func (p *Pool) clearSnapshotGeneration(ctx context.Context) error {
	for owned := range p.snapshotEntries {
		if err := p.index.Evict(ctx, owned.key, kvblock.RequestKey, []kvblock.PodEntry{owned.entry}); err != nil {
			return err
		}
	}
	clear(p.snapshotEntries)
	return nil
}
