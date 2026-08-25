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
	syncer       fwkdl.CrossReplicaSyncer
	contributors []fwkdl.CrossReplicaContributor
	interval     time.Duration

	// mu guards registered and serializes publishing with endpoint removal.
	mu         sync.Mutex
	registered sets.Set[types.NamespacedName]
}

// newCrossReplicaPublisher collects the opted-in CrossReplicaContributors, or
// returns nil if there is no syncer or none opt in. A non-positive interval
// falls back to defaultCrossReplicaSyncInterval.
func newCrossReplicaPublisher(syncer fwkdl.CrossReplicaSyncer, extractors *extractorMap, interval time.Duration) *crossReplicaPublisher {
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
	return &crossReplicaPublisher{syncer: syncer, contributors: contributors, interval: interval}
}

// Interval returns the configured sync cadence.
func (p *crossReplicaPublisher) Interval() time.Duration {
	return p.interval
}

// Start starts the shared publishing loop.
func (p *crossReplicaPublisher) Start(ctx context.Context) {
	go p.run(ctx)
}

// RegisterEndpoint adds key to the publishing set if it is absent.
func (p *crossReplicaPublisher) RegisterEndpoint(key types.NamespacedName) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.registered == nil {
		p.registered = sets.New[types.NamespacedName]()
	}
	if p.registered.Has(key) {
		return false
	}
	p.registered.Insert(key)
	return true
}

// UnregisterEndpoint waits for an in-flight publish, runs finalize, and removes
// key before a replacement with the same key can publish.
func (p *crossReplicaPublisher) UnregisterEndpoint(key types.NamespacedName, finalize func()) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.registered.Has(key) {
		return false
	}
	p.registered.Delete(key)
	if finalize != nil {
		finalize()
	}
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
	for _, endpointID := range p.registeredEndpoints() {
		p.mu.Lock()
		if !p.registered.Has(endpointID) {
			p.mu.Unlock()
			continue
		}
		dispatchCtx, cancel := context.WithTimeout(ctx, defaultCrossReplicaPublishTimeout)
		p.publish(dispatchCtx, endpointID.String())
		cancel()
		p.mu.Unlock()
	}
}

func (p *crossReplicaPublisher) registeredEndpoints() []types.NamespacedName {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.registered.UnsortedList()
}

// HandleEndpointEvent installs aggregate state readers.
func (p *crossReplicaPublisher) HandleEndpointEvent(ctx context.Context, event fwkdl.EndpointEvent, plugin fwkplugin.Plugin) error {
	contributor, ok := plugin.(fwkdl.CrossReplicaContributor)
	if !ok {
		return nil
	}
	spec := contributor.CrossReplicaState()
	if spec.SyncDisabled {
		return nil
	}
	endpointID := event.Endpoint.GetMetadata().GetNamespacedName().String()
	switch event.Type {
	case fwkdl.EventAddOrUpdate:
		event.Endpoint.GetAttributes().Put(spec.AttributeKey, &fwkdl.DynamicAttribute{
			Get: func() fwkdl.Cloneable {
				if value, ok, _ := p.syncer.Get(ctx, spec.StateKey, endpointID, spec.Aggregate); ok {
					if cloneable, ok := value.(fwkdl.Cloneable); ok {
						return cloneable
					}
				}
				return nil
			},
		})
	case fwkdl.EventDelete:
		if err := p.syncer.Delete(ctx, spec.StateKey, endpointID); err != nil {
			return fmt.Errorf("delete shared state for key %s: %w", spec.StateKey, err)
		}
	}
	return nil
}

func (p *crossReplicaPublisher) publish(ctx context.Context, endpointID string) {
	logger := log.FromContext(ctx).WithValues("endpoint", endpointID)
	for _, c := range p.contributors {
		spec := c.CrossReplicaState()
		if err := p.syncer.Set(ctx, spec.StateKey, endpointID, spec.Supply(endpointID)()); err != nil {
			logger.V(logging.DEBUG).Info("cross-replica publish failed", "key", spec.StateKey, "err", err)
		}
	}
}
