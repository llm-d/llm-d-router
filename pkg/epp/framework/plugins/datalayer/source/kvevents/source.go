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

package kvevents

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"sort"
	"strconv"
	"sync"

	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	events "github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

const PluginType = "kv-events-source"

// consumer is a bound EventConsumer and the plugin that bound it.
type consumer struct {
	owner plugin.TypedName
	events.EventConsumer
}

// Source shares one ordered, decoded event stream among the consumers that
// plugins bind to it. It follows endpoint lifecycle events as an extractor of
// the endpoint-notification-source and opens one subscriber per matching
// endpoint.
type Source struct {
	typedName   plugin.TypedName
	ctx         context.Context
	config      *events.Config
	selector    labels.Selector
	pool        *events.Pool
	subscribers *events.SubscriberManager
	mu          sync.RWMutex
	started     bool
	consumers   []consumer
	// resyncMu guards resyncs. Each endpoint's entry is read and written by
	// the pool worker that owns the endpoint.
	resyncMu sync.Mutex
	resyncs  map[string]*resyncState
}

// resyncState tracks the consumers of one endpoint that dropped their state
// on a continuous stream.
type resyncState struct {
	// pending consumers failed and wait for the next resync.
	pending map[int]bool
	// active consumers receive the running resync's rebuilt batches.
	active map[int]bool
	// skipThrough is the last shared sequence each resynced consumer's
	// rebuild covers.
	skipThrough map[int]uint64
}

// delivers reports whether consumer i receives the batch from source.
func (r *resyncState) delivers(i int, source events.EventSource) bool {
	if r == nil {
		return !source.Resync
	}
	if source.Resync {
		return r.active[i]
	}
	if through, ok := r.skipThrough[i]; ok {
		if source.Sequence <= through {
			return false
		}
		delete(r.skipThrough, i)
	}
	return true
}

var (
	_ fwkdl.EndpointExtractor = &Source{}
	_ fwkdl.Registrant        = &Source{}
	_ plugin.StateDumper      = &Source{}
	_ events.EventConsumer    = &Source{}
)

func Factory(name string, parameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	cfg := events.DefaultConfig()
	if parameters != nil {
		if err := parameters.Decode(cfg); err != nil {
			return nil, fmt.Errorf("parse %s config: %w", PluginType, err)
		}
	}
	return New(handle.Context(), name, cfg)
}

func New(ctx context.Context, name string, cfg *events.Config) (*Source, error) {
	if cfg == nil || !cfg.DiscoverPods || cfg.PodDiscoveryConfig == nil || cfg.ZMQEndpoint != "" {
		return nil, errors.New("kv-events-source requires per-endpoint pod discovery without a global zmqEndpoint")
	}
	if cfg.PodDiscoveryConfig.SocketPort <= 0 {
		return nil, fmt.Errorf("kv-events-source socketPort must be positive, got %d", cfg.PodDiscoveryConfig.SocketPort)
	}
	selector, err := labels.Parse(cfg.PodDiscoveryConfig.PodLabelSelector)
	if err != nil {
		return nil, fmt.Errorf("parse pod label selector: %w", err)
	}
	adapter, err := engineadapter.NewAdapter(cfg.EngineType)
	if err != nil {
		return nil, err
	}
	s := &Source{typedName: plugin.TypedName{Type: PluginType, Name: name}, ctx: ctx, config: cfg, selector: selector}
	s.pool, err = events.NewConsumerPool(cfg, adapter, s)
	if err != nil {
		return nil, err
	}
	s.subscribers = events.NewSubscriberManager(s.pool)
	s.pool.Start(ctx)
	go func() {
		<-ctx.Done()
		s.subscribers.Shutdown(ctx)
		s.pool.Shutdown(ctx)
	}()
	return s, nil
}

func (s *Source) TypedName() plugin.TypedName { return s.typedName }

// RegisterDependencies subscribes the source to endpoint lifecycle events.
func (s *Source) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      s.typedName,
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  s,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType, sourcenotifications.EndpointNotificationSourceType),
	})
}

// AddConsumer delivers every endpoint's decoded stream to c. Consumers bind
// before endpoint discovery starts; owner names the binding plugin in errors
// and in the state dump.
func (s *Source) AddConsumer(owner plugin.TypedName, c events.EventConsumer) error {
	if c == nil {
		return fmt.Errorf("%s bound a nil KV event consumer", owner)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started {
		return fmt.Errorf("%s bound a KV event consumer after endpoint discovery started", owner)
	}
	s.consumers = append(s.consumers, consumer{owner: owner, EventConsumer: c})
	return nil
}

// maxDumpSubscribers bounds the DumpState payload on large deployments.
const maxDumpSubscribers = 100

type sourceState struct {
	Subscribers      []string `json:"subscribers"`
	TotalSubscribers int      `json:"totalSubscribers"`
	MaxSubscribers   int      `json:"maxSubscribers"`
	Consumers        []string `json:"consumers"`
}

// DumpState reports the endpoints with an active subscriber, sorted and
// capped, and the plugins with a bound consumer.
func (s *Source) DumpState() (json.RawMessage, error) {
	ids, _ := s.subscribers.GetActiveSubscribers()
	sort.Strings(ids)
	total := len(ids)
	if total > maxDumpSubscribers {
		ids = ids[:maxDumpSubscribers]
	}
	s.mu.RLock()
	consumers := make([]string, 0, len(s.consumers))
	for _, c := range s.consumers {
		consumers = append(consumers, c.owner.String())
	}
	s.mu.RUnlock()
	return json.Marshal(sourceState{
		Subscribers: ids, TotalSubscribers: total, MaxSubscribers: maxDumpSubscribers, Consumers: consumers,
	})
}

// Extract opens a subscriber for an endpoint that matches the selector and
// removes it when the endpoint is deleted or stops matching.
func (s *Source) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	// Consumers run under the read lock, so only the first event takes the
	// write lock.
	s.mu.RLock()
	started := s.started
	s.mu.RUnlock()
	if !started {
		s.mu.Lock()
		s.started = true
		s.mu.Unlock()
	}
	meta := event.Endpoint.GetMetadata()
	if meta == nil || meta.ID.Name == "" {
		return nil
	}
	key := meta.ID.String()
	if event.Type == fwkdl.EventDelete || !s.selector.Matches(labels.Set(meta.Labels)) {
		s.subscribers.RemoveSubscriber(ctx, key)
		return nil
	}
	if meta.Address == "" {
		return nil
	}
	zmqEndpoint, replay, snapshot := SubscriberEndpoints(meta, s.config.PodDiscoveryConfig)
	return s.subscribers.EnsureSubscriber(s.ctx, key, PodIdentifier(meta),
		zmqEndpoint, replay, snapshot, s.config.TopicFilter, true)
}

// PodIdentifier returns the per-endpoint KV-block index key for pod-discovery
// subscriptions. Each engine publishes KV events on its own socket; when a pod
// serves several ranks behind one port, the rank suffix keeps each rank's
// blocks distinct in the index.
func PodIdentifier(meta *fwkdl.EndpointMetadata) string {
	id := fmt.Sprintf("%s:%s", meta.Address, meta.Port)
	if meta.RankByHeader {
		id += "/rank-" + strconv.Itoa(meta.RankIndex)
	}
	return id
}

// SubscriberEndpoints returns the endpoint's KV-events, replay, and snapshot
// socket addresses; an empty replay or snapshot address disables that path.
// An endpoint with ZMQPort gets no replay or snapshot address: those ports
// follow the configured port plus rank offset and can belong to another engine
// on the same address.
func SubscriberEndpoints(meta *fwkdl.EndpointMetadata, cfg *events.PodDiscoveryConfig) (zmqEndpoint, replayEndpoint, snapshotEndpoint string) {
	if meta.ZMQPort > 0 {
		return "tcp://" + net.JoinHostPort(meta.Address, strconv.Itoa(meta.ZMQPort)), "", ""
	}
	// vLLM offsets each engine's publisher ports by its data-parallel rank,
	// which RankIndex holds (global when a data-parallel group spans pods).
	offset := meta.GetRankIndex()
	zmqEndpoint = "tcp://" + net.JoinHostPort(meta.Address, strconv.Itoa(cfg.SocketPort+offset))
	if replayPort := cfg.EffectiveReplayPort(); replayPort > 0 {
		replayEndpoint = "tcp://" + net.JoinHostPort(meta.Address, strconv.Itoa(replayPort+offset))
	}
	if snapshotPort := cfg.SnapshotSocketPort; snapshotPort > 0 {
		snapshotEndpoint = "tcp://" + net.JoinHostPort(meta.Address, strconv.Itoa(snapshotPort+offset))
	}
	return zmqEndpoint, replayEndpoint, snapshotEndpoint
}

// ProcessEvents delivers a batch to every consumer. A resync batch reaches
// only the consumers being resynced, which skip shared batches the rebuild
// covers.
func (s *Source) ProcessEvents(ctx context.Context, source events.EventSource, batch events.EventBatch) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	state := s.resyncState(source.Endpoint, false)
	for i, c := range s.consumers {
		if !state.delivers(i, source) {
			continue
		}
		if err := invoke(func() error { return c.ProcessEvents(ctx, source, batch) }); err != nil {
			log.FromContext(ctx).Error(err, "KV event consumer failed", "consumer", c.owner, "endpoint", source.Endpoint)
			if resetErr := invoke(func() error { return c.Reset(ctx, source) }); resetErr != nil {
				log.FromContext(ctx).Error(resetErr, "KV event consumer reset failed", "consumer", c.owner, "endpoint", source.Endpoint)
			}
			// The stream is continuous, so the subscriber sees no gap; the
			// reset consumer gets its state back from a rebuild of the endpoint.
			// It leaves a running resync, whose earlier batches it dropped.
			state = s.resyncState(source.Endpoint, true)
			delete(state.active, i)
			delete(state.skipThrough, i)
			state.pending[i] = true
			s.subscribers.RequestResync(source.Endpoint)
		}
	}
	return nil
}

// Reset resets every consumer of the endpoint. A resync reset resets the
// consumers that requested the resync and starts delivering the rebuild to
// them. An aborted resync resets the consumers that hold its partial rebuild
// and returns them to pending.
func (s *Source) Reset(ctx context.Context, source events.EventSource) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var targets map[int]bool
	if source.Resync {
		state := s.resyncState(source.Endpoint, true)
		if source.ResyncAborted {
			targets, state.active = state.active, map[int]bool{}
		} else {
			targets, state.active, state.pending = state.pending, state.pending, map[int]bool{}
		}
		for i := range targets {
			if source.ResyncAborted {
				state.pending[i] = true
				delete(state.skipThrough, i)
			} else {
				state.skipThrough[i] = source.Sequence
			}
		}
	} else {
		s.resyncMu.Lock()
		delete(s.resyncs, source.Endpoint)
		s.resyncMu.Unlock()
	}
	var errs []error
	for i, c := range s.consumers {
		if source.Resync && !targets[i] {
			continue
		}
		if err := invoke(func() error { return c.Reset(ctx, source) }); err != nil {
			errs = append(errs, fmt.Errorf("reset %s: %w", c.owner, err))
		}
	}
	return errors.Join(errs...)
}

// resyncState returns the endpoint's resync state, creating it when create
// is set.
func (s *Source) resyncState(endpoint string, create bool) *resyncState {
	s.resyncMu.Lock()
	defer s.resyncMu.Unlock()
	state := s.resyncs[endpoint]
	if state == nil && create {
		if s.resyncs == nil {
			s.resyncs = map[string]*resyncState{}
		}
		state = &resyncState{pending: map[int]bool{}, active: map[int]bool{}, skipThrough: map[int]uint64{}}
		s.resyncs[endpoint] = state
	}
	return state
}

func invoke(fn func() error) (err error) {
	defer func() {
		if value := recover(); value != nil {
			err = fmt.Errorf("KV event consumer panicked: %v", value)
		}
	}()
	return fn()
}
