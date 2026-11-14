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

package crossreplica

import (
	"context"
	"time"

	"github.com/go-logr/logr"
	"golang.org/x/time/rate"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/utils/clock"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

const (
	defaultCrossReplicaSyncInterval   = 200 * time.Millisecond
	defaultCrossReplicaPublishTimeout = time.Second
	publishFailureLogInterval         = 30 * time.Second
)

// crossReplicaSource is one kind of local state the publisher shares.
type crossReplicaSource struct {
	spec fwkdl.CrossReplicaSpec
	// ids lists the IDs to publish in the current cycle.
	// For a priority band demand source, this is the list of priority values.
	ids func() []string
}

// Publisher writes this replica's flow control state to the cross-replica syncer on a
// fixed interval.
type Publisher struct {
	syncer         fwkdl.CrossReplicaSyncer
	sources        []*crossReplicaSource
	interval       time.Duration
	publishTimeout time.Duration
	clock          clock.WithTicker
	failureLog     rate.Sometimes
}

// NewPublisher returns a publisher of the queued work per priority band. Values are
// shared only with replicas using the same scope. Non-positive durations fall back to their defaults.
func NewPublisher(
	syncer fwkdl.CrossReplicaSyncer,
	registry bandOccupancy,
	scope string,
	interval, publishTimeout time.Duration,
	clk clock.WithTicker,
) *Publisher {
	return newPublisher(syncer, []*crossReplicaSource{newPriorityBandDemandSource(registry, scope)},
		interval, publishTimeout, clk)
}

func newPublisher(
	syncer fwkdl.CrossReplicaSyncer,
	sources []*crossReplicaSource,
	interval, publishTimeout time.Duration,
	clk clock.WithTicker,
) *Publisher {
	if interval <= 0 {
		interval = defaultCrossReplicaSyncInterval
	}
	if publishTimeout <= 0 {
		publishTimeout = defaultCrossReplicaPublishTimeout
	}
	return &Publisher{
		syncer:         syncer,
		sources:        sources,
		interval:       interval,
		publishTimeout: publishTimeout,
		clock:          clk,
		failureLog:     rate.Sometimes{Interval: publishFailureLogInterval},
	}
}

// Run publishes every interval until ctx ends. Published values are never deleted: they expire after
// the syncer's state TTL once this replica stops writing them, and the registry removes a band only
// when it is idle, so its last published value is zero.
func (p *Publisher) Run(ctx context.Context) {
	logger := log.FromContext(ctx).WithName("cross-replica-publisher")
	defer utilruntime.HandleCrashWithLogger(logger)
	ctx = log.IntoContext(ctx, logger)

	ticker := p.clock.NewTicker(p.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C():
			p.publishAll(ctx)
		}
	}
}

func (p *Publisher) publishAll(ctx context.Context) {
	logger := log.FromContext(ctx)
	for _, src := range p.sources {
		p.publish(ctx, logger, src)
	}
}

func (p *Publisher) publish(ctx context.Context, logger logr.Logger, src *crossReplicaSource) {
	ids := src.ids()
	if batch, ok := p.syncer.(fwkdl.CrossReplicaBatchSyncer); ok {
		for id, err := range batch.SetBatch(ctx, src.spec, ids, p.publishTimeout) {
			p.logFailure(logger, err, src.spec.StateKey, id)
		}
	} else {
		for _, id := range ids {
			setCtx, cancel := context.WithTimeout(ctx, p.publishTimeout)
			err := p.syncer.Set(setCtx, src.spec, id)
			cancel()
			if err != nil {
				p.logFailure(logger, err, src.spec.StateKey, id)
			}
		}
	}
}

func (p *Publisher) logFailure(logger logr.Logger, err error, key fwkdl.StateKey, id string) {
	p.failureLog.Do(func() {
		logger.Error(err, "cross-replica publish failed", "key", key, "id", id)
	})
}
