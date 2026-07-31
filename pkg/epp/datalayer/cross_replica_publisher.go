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
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	crossReplicaPublisherType = "cross-replica-publisher"
	// defaultCrossReplicaSyncInterval is the fallback cadence at which local
	// per-endpoint state is pushed to the syncer when none is configured.
	defaultCrossReplicaSyncInterval = 200 * time.Millisecond
)

// crossReplicaPublisher is a PollingDispatcher that publishes each endpoint's
// local state to the syncer. The datalayer drives it per endpoint at interval,
// like any other polling source.
type crossReplicaPublisher struct {
	syncer       fwkdl.CrossReplicaSyncer
	contributors []fwkdl.CrossReplicaContributor
	interval     time.Duration
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

func (p *crossReplicaPublisher) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: crossReplicaPublisherType, Name: crossReplicaPublisherType}
}

// Interval is the cadence at which the datalayer calls Dispatch.
func (p *crossReplicaPublisher) Interval() time.Duration {
	return p.interval
}

// Dispatch publishes each contributor's local value for ep. Set failures are
// logged, not returned, so the Collector doesn't count them as poll errors.
func (p *crossReplicaPublisher) Dispatch(ctx context.Context, ep fwkdl.Endpoint) error {
	endpointID := ep.GetMetadata().GetNamespacedName().String()
	logger := log.FromContext(ctx).WithValues("endpoint", endpointID)
	for _, c := range p.contributors {
		spec := c.CrossReplicaState()
		if err := p.syncer.Set(ctx, spec.StateKey, endpointID, spec.Supply(endpointID)()); err != nil {
			logger.V(logging.DEBUG).Info("cross-replica publish failed", "key", spec.StateKey, "err", err)
		}
	}
	return nil
}

// AppendExtractor is unused: the publisher pushes state, it has no extractors.
func (p *crossReplicaPublisher) AppendExtractor(fwkplugin.Plugin) error {
	return errors.New("cross-replica publisher does not accept extractors")
}
