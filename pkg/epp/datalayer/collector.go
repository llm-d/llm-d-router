/*
Copyright 2025 The Kubernetes Authors.
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
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	defaultCollectionTimeout = time.Second
)

// Ticker implements a time source for periodic invocation.
// The Ticker is passed in as parameter a Collector to allow control over time
// progress in tests, ensuring tests are deterministic and fast.
type Ticker interface {
	Channel() <-chan time.Time
	Stop()
}

// TimeTicker implements a Ticker based on time.Ticker.
type TimeTicker struct {
	*time.Ticker
}

// NewTimeTicker returns a new time.Ticker with the configured duration.
func NewTimeTicker(d time.Duration) Ticker {
	return &TimeTicker{
		Ticker: time.NewTicker(d),
	}
}

// Channel exposes the ticker's channel.
func (t *TimeTicker) Channel() <-chan time.Time {
	return t.C
}

// Collector runs data collection for a single endpoint.
//
// Lifecycle contract: any in-flight write the collection goroutine performs
// against the endpoint completes before Stop returns. Callers may therefore
// mutate or release endpoint state immediately after Stop returns without
// racing the collection goroutine.
type Collector struct {
	mu     sync.Mutex
	cancel context.CancelFunc
	done   chan struct{}
}

// NewCollector returns a new collector.
func NewCollector() *Collector {
	return &Collector{done: make(chan struct{})}
}

// Start launches the collection goroutine.
// Each PollingDispatcher owns its extractors; the Collector calls Dispatch per tick.
func (c *Collector) Start(ctx context.Context, ticker Ticker, ep fwkdl.Endpoint, dispatchers []fwkdl.PollingDispatcher) error {
	if len(dispatchers) == 0 {
		return errors.New("cannot start collector with empty dispatchers")
	}
	for _, d := range dispatchers {
		if d == nil {
			return errors.New("cannot add nil dispatcher")
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cancel != nil {
		return errors.New("collector start called multiple times")
	}
	ctx, cancel := context.WithCancel(ctx)
	c.cancel = cancel
	go c.run(ctx, ticker, ep, dispatchers)
	return nil
}

// Stop cancels the collection goroutine and blocks until it has exited. Idempotent.
func (c *Collector) Stop() {
	c.mu.Lock()
	cancel := c.cancel
	c.mu.Unlock()
	if cancel != nil {
		cancel()
		<-c.done
	}
}

func (c *Collector) run(ctx context.Context, ticker Ticker, ep fwkdl.Endpoint, dispatchers []fwkdl.PollingDispatcher) {
	defer func() {
		close(c.done)
		ticker.Stop()
	}()
	logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().GetIPAddress())

	statuses := make([]pollStatusRecorder, len(dispatchers))
	for i, disp := range dispatchers {
		statuses[i].key = fwkdl.PollStatusDataKey(disp.TypedName().Name)
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.Channel():
			for i, disp := range dispatchers {
				if ctx.Err() != nil {
					return
				}
				// Only polls that run are recorded, so a source with a longer
				// interval does not report a success on the ticks it skips.
				if interval, ok := disp.(*intervalDispatcher); ok {
					if !interval.due() {
						continue
					}
					// due has advanced the countdown for this tick, and the
					// wrapper's Dispatch would advance it again.
					disp = interval.PollingDispatcher
				}
				dispCtx, cancel := context.WithTimeout(ctx, defaultCollectionTimeout)
				err := disp.Dispatch(dispCtx, ep)
				cancel()
				// A poll cut short by Stop says nothing about the endpoint.
				if ctx.Err() != nil {
					return
				}
				if err != nil {
					tn := disp.TypedName()
					metrics.RecordDataLayerPollError(tn.Type)
					logger.V(logging.DEBUG).Info("dispatch failed", "source", tn, "err", err)
				}
				statuses[i].record(ep, err, time.Now())
			}
		}
	}
}

// pollStatusRecorder tracks one source's failure run for one endpoint and
// writes it to the endpoint as a PollStatus attribute. It writes only when the
// status changes, so a source that keeps succeeding costs nothing.
type pollStatusRecorder struct {
	key    fwkplugin.DataKey
	status fwkdl.PollStatus
}

func (r *pollStatusRecorder) record(ep fwkdl.Endpoint, err error, now time.Time) {
	if err == nil {
		if r.status.ConsecutiveFailures == 0 {
			return
		}
		r.status = fwkdl.PollStatus{}
	} else {
		if r.status.ConsecutiveFailures == 0 {
			r.status.FailingSince = now
		}
		r.status.ConsecutiveFailures++
	}
	// AttributeMap does not require Put to clone, so pass a copy the next
	// record cannot modify.
	status := r.status
	ep.GetAttributes().Put(r.key, &status)
}

const pollStatusProducerName = "datalayer-poll-status"

// pollStatusProducer declares the PollStatus the collector records for each
// polling source in the data layer configuration.
type pollStatusProducer struct {
	keys map[fwkplugin.DataKey]any
}

// NewPollStatusProducer returns a plugin that declares the PollStatus of each
// polling source in cfg. Added to the plugin handle before data dependencies
// are validated, it makes a consumer of a source's PollStatus fail startup
// unless the runtime polls that source.
func NewPollStatusProducer(cfg *Config) fwkplugin.ProducerPlugin {
	keys := map[fwkplugin.DataKey]any{}
	if cfg != nil {
		for _, src := range cfg.Sources {
			if _, ok := src.Plugin.(fwkdl.PollingDispatcher); ok {
				keys[fwkdl.PollStatusDataKey(src.Plugin.TypedName().Name)] = &fwkdl.PollStatus{}
			}
		}
	}
	return &pollStatusProducer{keys: keys}
}

func (p *pollStatusProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: pollStatusProducerName, Name: pollStatusProducerName}
}

func (p *pollStatusProducer) Produces() map[fwkplugin.DataKey]any { return p.keys }
