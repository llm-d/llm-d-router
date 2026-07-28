/*
Copyright 2025 The Kubernetes Authors.

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
// Each ScheduledDispatcher owns its extractors; the Collector invokes
// Dispatch on a source only when its period elapses (measured in base ticks).
func (c *Collector) Start(ctx context.Context, ticker Ticker, ep fwkdl.Endpoint, scheduled []ScheduledDispatcher) error {
	if len(scheduled) == 0 {
		return errors.New("cannot start collector with empty dispatchers")
	}
	for _, s := range scheduled {
		if s.Dispatcher == nil {
			return errors.New("cannot add nil dispatcher")
		}
		if s.PeriodTicks < 1 {
			return errors.New("periodTicks must be >= 1")
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
	go c.run(ctx, ticker, ep, scheduled)
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

func (c *Collector) run(ctx context.Context, ticker Ticker, ep fwkdl.Endpoint, scheduled []ScheduledDispatcher) {
	defer func() {
		close(c.done)
		ticker.Stop()
	}()
	logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().GetIPAddress())

	// nextDue[i] is the tick index when scheduled[i] should next Dispatch.
	// Zero means fire on the first tick (tick == 0).
	nextDue := make([]int, len(scheduled))
	tick := 0

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.Channel():
			for i, s := range scheduled {
				if ctx.Err() != nil {
					return
				}
				if tick != nextDue[i] {
					continue
				}
				dispCtx, cancel := context.WithTimeout(ctx, defaultCollectionTimeout)
				if err := s.Dispatcher.Dispatch(dispCtx, ep); err != nil {
					tn := s.Dispatcher.TypedName()
					metrics.RecordDataLayerPollError(tn.Type)
					logger.V(logging.DEBUG).Info("dispatch failed", "source", tn, "err", err)
				}
				cancel()
				nextDue[i] = tick + s.PeriodTicks
			}
			tick++
		}
	}
}
