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
	"strconv"
	"sync"

	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	events "github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

const PluginType = "kv-events-source"

type extractor interface {
	plugin.Plugin
	events.EventConsumer
}

// Source shares one ordered, decoded event stream among its bound consumers.
type Source struct {
	typedName   plugin.TypedName
	ctx         context.Context
	config      *events.Config
	selector    labels.Selector
	pool        *events.Pool
	subscribers *events.SubscriberManager
	mu          sync.RWMutex
	started     bool
	extractors  []extractor
}

var (
	_ fwkdl.EndpointSource  = &Source{}
	_ fwkdl.ExtractorBinder = &Source{}
	_ events.EventConsumer  = &Source{}
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

func (s *Source) AppendExtractor(p plugin.Plugin) error {
	ext, ok := p.(extractor)
	if !ok {
		return fmt.Errorf("extractor %s must implement kvevents.EventConsumer", p.TypedName())
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started {
		return errors.New("KV event extractors must be bound before endpoint discovery")
	}
	s.extractors = append(s.extractors, ext)
	return nil
}

func (s *Source) NotifyEndpoint(ctx context.Context, event fwkdl.EndpointEvent) (*fwkdl.EndpointEvent, error) {
	s.mu.Lock()
	s.started = true
	s.mu.Unlock()
	meta := event.Endpoint.GetMetadata()
	if meta == nil || meta.ID.Name == "" {
		return nil, nil //nolint:nilnil // EndpointSource uses nil to suppress lifecycle dispatch.
	}
	key := meta.ID.String()
	if event.Type == fwkdl.EventDelete || !s.selector.Matches(labels.Set(meta.Labels)) {
		s.subscribers.RemoveSubscriber(ctx, key)
		return nil, nil //nolint:nilnil // EndpointSource uses nil to suppress lifecycle dispatch.
	}
	if meta.Address == "" {
		return nil, nil //nolint:nilnil // EndpointSource uses nil to suppress lifecycle dispatch.
	}
	cfg := s.config.PodDiscoveryConfig
	socket := func(port int) string {
		return "tcp://" + net.JoinHostPort(meta.Address, strconv.Itoa(port+meta.GetRankIndex()))
	}
	replay, snapshot := "", ""
	if port := cfg.EffectiveReplayPort(); port > 0 {
		replay = socket(port)
	}
	if cfg.SnapshotSocketPort > 0 {
		snapshot = socket(cfg.SnapshotSocketPort)
	}
	err := s.subscribers.EnsureSubscriber(s.ctx, key, fmt.Sprintf("%s:%s", meta.Address, meta.Port),
		socket(cfg.SocketPort), replay, snapshot, s.config.TopicFilter, true)
	return nil, err
}

func (s *Source) ProcessEvents(ctx context.Context, source events.EventSource, batch events.EventBatch) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, ext := range s.extractors {
		if err := invoke(func() error { return ext.ProcessEvents(ctx, source, batch) }); err != nil {
			metrics.RecordDataLayerExtractError(PluginType, ext.TypedName().Type)
			log.FromContext(ctx).Error(err, "KV event consumer failed", "extractor", ext.TypedName(), "endpoint", source.Endpoint)
			if resetErr := invoke(func() error { return ext.Reset(ctx, source) }); resetErr != nil {
				log.FromContext(ctx).Error(resetErr, "KV event consumer reset failed", "extractor", ext.TypedName(), "endpoint", source.Endpoint)
			}
		}
	}
	return nil
}

func (s *Source) Reset(ctx context.Context, source events.EventSource) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var errs []error
	for _, ext := range s.extractors {
		if err := invoke(func() error { return ext.Reset(ctx, source) }); err != nil {
			errs = append(errs, fmt.Errorf("reset %s: %w", ext.TypedName(), err))
		}
	}
	return errors.Join(errs...)
}

func invoke(fn func() error) (err error) {
	defer func() {
		if value := recover(); value != nil {
			err = fmt.Errorf("KV event consumer panicked: %v", value)
		}
	}()
	return fn()
}
