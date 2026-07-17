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

// Start launches the collection goroutines for polling and streaming dispatchers.
func (c *Collector) Start(
	ctx context.Context,
	pollingTicker Ticker,
	ep fwkdl.Endpoint,
	pollers []fwkdl.PollingDispatcher,
	streamers []fwkdl.StreamingDispatcher,
) error {
	if len(pollers) == 0 && len(streamers) == 0 {
		return errors.New("cannot start collector with empty dispatchers")
	}
	for _, p := range pollers {
		if p == nil {
			return errors.New("cannot add nil dispatcher")
		}
	}
	for _, s := range streamers {
		if s == nil {
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
	go c.run(ctx, pollingTicker, ep, pollers, streamers)
	return nil
}

// Stop cancels the collection goroutines and blocks until all have exited. Idempotent.
func (c *Collector) Stop() {
	c.mu.Lock()
	cancel := c.cancel
	c.mu.Unlock()
	if cancel != nil {
		cancel()
		<-c.done
	}
}

func (c *Collector) run(
	ctx context.Context,
	pollingTicker Ticker,
	ep fwkdl.Endpoint,
	pollers []fwkdl.PollingDispatcher,
	streamers []fwkdl.StreamingDispatcher,
) {
	var wg sync.WaitGroup

	if len(pollers) > 0 && pollingTicker != nil {
		wg.Add(1)
		go c.runPolling(ctx, pollingTicker, ep, pollers, &wg)
	}

	for _, s := range streamers {
		wg.Add(1)
		go c.runStreaming(ctx, ep, s, &wg)
	}

	go func() {
		wg.Wait()
		if pollingTicker != nil {
			pollingTicker.Stop()
		}
		close(c.done)
	}()
}

func (c *Collector) runPolling(
	ctx context.Context,
	pollingTicker Ticker,
	ep fwkdl.Endpoint,
	dispatchers []fwkdl.PollingDispatcher,
	wg *sync.WaitGroup,
) {
	defer wg.Done()
	logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().GetIPAddress())

	for {
		select {
		case <-ctx.Done():
			return
		case <-pollingTicker.Channel():
			for _, disp := range dispatchers {
				if ctx.Err() != nil {
					return
				}
				dispCtx, cancel := context.WithTimeout(ctx, defaultCollectionTimeout)
				if err := disp.Dispatch(dispCtx, ep); err != nil {
					tn := disp.TypedName()
					metrics.RecordDataLayerPollError(tn.Type)
					logger.V(logging.DEBUG).Info("dispatch failed", "source", tn, "err", err)
				}
				cancel()
			}
		}
	}
}

func (c *Collector) runStreaming(
	ctx context.Context,
	ep fwkdl.Endpoint,
	disp fwkdl.StreamingDispatcher,
	wg *sync.WaitGroup,
) {
	defer wg.Done()
	logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().GetIPAddress())

	if err := disp.Start(ctx, ep); err != nil && ctx.Err() == nil {
		tn := disp.TypedName()
		metrics.RecordDataLayerPollError(tn.Type)
		logger.V(logging.DEBUG).Info("streaming failed", "source", tn, "err", err)
	}
}
