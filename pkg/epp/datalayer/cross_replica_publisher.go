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

package datalayer

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	// defaultCrossReplicaSyncInterval is the fallback cadence at which local
	// per-endpoint state is pushed to the syncer when none is configured.
	defaultCrossReplicaSyncInterval   = 200 * time.Millisecond
	defaultCrossReplicaPublishTimeout = time.Second
)

// crossReplicaPublisher owns cross-replica publishing and endpoint lifecycle
// coordination. One shared ticker publishes every registered endpoint.
type crossReplicaPublisher struct {
	syncer         fwkdl.CrossReplicaSyncer
	contributors   []fwkdl.CrossReplicaContributor
	interval       time.Duration
	publishTimeout time.Duration

	// mu guards endpoints and orders syncer operations with endpoint removal.
	mu        sync.RWMutex
	endpoints sets.Set[types.NamespacedName]
}

// newCrossReplicaPublisher collects the opted-in CrossReplicaContributors, or
// returns nil if there is no syncer or none opt in. Non-positive durations
// fall back to their defaults.
func newCrossReplicaPublisher(syncer fwkdl.CrossReplicaSyncer, extractors *extractorMap, interval, publishTimeout time.Duration) *crossReplicaPublisher {
	if syncer == nil {
		return nil
	}
	var contributors []fwkdl.CrossReplicaContributor
	extractors.Range(func(_ string, exts []fwkplugin.Plugin) bool {
		for _, ext := range exts {
			if c, ok := ext.(fwkdl.CrossReplicaContributor); ok && !c.CrossReplicaState().SyncDisabled {
				contributors = append(contributors, c)
			}
		}
		return true
	})
	if len(contributors) == 0 {
		return nil
	}
	if interval <= 0 {
		interval = defaultCrossReplicaSyncInterval
	}
	if publishTimeout <= 0 {
		publishTimeout = defaultCrossReplicaPublishTimeout
	}
	return &crossReplicaPublisher{
		syncer:         syncer,
		contributors:   contributors,
		interval:       interval,
		publishTimeout: publishTimeout,
	}
}

func (p *crossReplicaPublisher) start(ctx context.Context) {
	go p.run(ctx)
}

func (p *crossReplicaPublisher) registerEndpoint(key types.NamespacedName) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.endpoints == nil {
		p.endpoints = sets.New[types.NamespacedName]()
	}
	if p.endpoints.Has(key) {
		return false
	}
	p.endpoints.Insert(key)
	return true
}

func (p *crossReplicaPublisher) run(ctx context.Context) {
	ticker := time.NewTicker(p.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.publishAll(ctx)
		}
	}
}

func (p *crossReplicaPublisher) publishAll(ctx context.Context) {
	for _, endpointID := range p.endpointSnapshot() {
		dispatchCtx, cancel := context.WithTimeout(ctx, p.publishTimeout)
		p.publish(dispatchCtx, endpointID)
		cancel()
	}
}

func (p *crossReplicaPublisher) endpointSnapshot() []types.NamespacedName {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.endpoints.UnsortedList()
}

func (p *crossReplicaPublisher) handleEndpointEvent(ctx context.Context, event fwkdl.EndpointEvent, plugin fwkplugin.Plugin) {
	contributor, ok := plugin.(fwkdl.CrossReplicaContributor)
	if !ok {
		return
	}
	spec := contributor.CrossReplicaState()
	if spec.SyncDisabled {
		return
	}
	if event.Type != fwkdl.EventAddOrUpdate {
		return
	}
	endpointID := event.Endpoint.GetMetadata().GetNamespacedName().String()
	event.Endpoint.GetAttributes().Put(spec.AttributeKey, &fwkdl.DynamicAttribute{
		Get: func() fwkdl.Cloneable {
			if value, ok, _ := p.get(ctx, spec, endpointID); ok {
				if cloneable, ok := value.(fwkdl.Cloneable); ok {
					return cloneable
				}
			}
			return nil
		},
	})
}

func (p *crossReplicaPublisher) set(ctx context.Context, spec fwkdl.CrossReplicaSpec, key types.NamespacedName) error {
	p.mu.RLock()
	defer p.mu.RUnlock()
	// The endpoint may have been deleted after publishAll took its snapshot.
	if !p.endpoints.Has(key) {
		return nil
	}
	endpointID := key.String()
	return p.syncer.Set(ctx, spec.StateKey, endpointID, spec.Supply(endpointID)(), spec.Aggregate)
}

func (p *crossReplicaPublisher) get(ctx context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) (any, bool, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.syncer.Get(ctx, spec.StateKey, endpointID)
}

func (p *crossReplicaPublisher) delete(ctx context.Context, key types.NamespacedName) (bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.endpoints.Has(key) {
		return false, nil
	}
	p.endpoints.Delete(key)

	endpointID := key.String()
	var errs []error
	for _, c := range p.contributors {
		spec := c.CrossReplicaState()
		if err := p.syncer.Delete(ctx, spec.StateKey, endpointID); err != nil {
			errs = append(errs, fmt.Errorf("delete shared state for key %s: %w", spec.StateKey, err))
		}
	}
	return true, errors.Join(errs...)
}

func (p *crossReplicaPublisher) publish(ctx context.Context, key types.NamespacedName) {
	endpointID := key.String()
	logger := log.FromContext(ctx).WithValues("endpoint", endpointID)
	var wg sync.WaitGroup
	for _, c := range p.contributors {
		spec := c.CrossReplicaState()
		wg.Go(func() {
			if err := p.set(ctx, spec, key); err != nil {
				logger.V(logging.DEBUG).Info("cross-replica publish failed", "key", spec.StateKey, "err", err)
			}
		})
	}
	wg.Wait()
}
