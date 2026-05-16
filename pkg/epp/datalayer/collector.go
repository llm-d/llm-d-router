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

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/metrics"
)

// TODO:
// currently the data store is expected to manage the state of multiple
// Collectors (e.g., using sync.Map mapping pod to its Collector). Alternatively,
// this can be encapsulated in this file, providing the data store with an interface
// to only update on endpoint addition/change and deletion. This can also be used
// to centrally track statistics such errors, active routines, etc.

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
func (c *Collector) Start(ctx context.Context, ticker Ticker, ep fwkdl.Endpoint, pollers []fwkdl.PollingDataSource, extractors map[string][]fwkdl.ExtractorBase) error {
	if len(pollers) == 0 {
		return errors.New("cannot start collector with empty sources")
	}
	for _, src := range pollers {
		if src == nil {
			return errors.New("cannot add nil data source")
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// Filter to poll-capable extractors up front so the hot loop avoids per-tick type assertions.
	pollingExtractors := make(map[string][]fwkdl.Extractor, len(extractors))
	for name, exts := range extractors {
		for _, ext := range exts {
			if e, ok := ext.(fwkdl.Extractor); ok {
				pollingExtractors[name] = append(pollingExtractors[name], e)
			}
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cancel != nil {
		return errors.New("collector start called multiple times")
	}
	ctx, cancel := context.WithCancel(ctx)
	c.cancel = cancel
	go c.run(ctx, ticker, ep, pollers, pollingExtractors)
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

func (c *Collector) run(ctx context.Context, ticker Ticker, ep fwkdl.Endpoint, pollers []fwkdl.PollingDataSource, extractors map[string][]fwkdl.Extractor) {
	defer func() {
		close(c.done)
		ticker.Stop()
	}()
	logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().GetIPAddress())

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.Channel():
			for _, src := range pollers {
				if ctx.Err() != nil {
					return
				}
				c.pollOne(ctx, src, ep, extractors, logger)
			}
		}
	}
}

func (c *Collector) pollOne(ctx context.Context, src fwkdl.PollingDataSource, ep fwkdl.Endpoint, extractors map[string][]fwkdl.Extractor, logger logr.Logger) {
	tn := src.TypedName()

	pollCtx, cancel := context.WithTimeout(ctx, defaultCollectionTimeout)
	defer cancel()
	data, err := src.Poll(pollCtx, ep)
	if err != nil {
		metrics.DataLayerPollErrorsTotal.WithLabelValues(tn.Type).Inc()
		metrics.LlmdDataLayerPollErrorsTotal.WithLabelValues(tn.Type).Inc()
		logger.V(logging.DEBUG).Info("poll failed", "source", tn, "err", err)
		return
	}
	if data == nil {
		return
	}

	for _, ext := range extractors[tn.Name] {
		if ctx.Err() != nil {
			return
		}
		extCtx, cancel := context.WithTimeout(ctx, defaultCollectionTimeout)
		err := ext.Extract(extCtx, data, ep)
		cancel()
		if err != nil {
			extName := ext.TypedName()
			metrics.DataLayerExtractErrorsTotal.WithLabelValues(tn.Type, extName.Type).Inc()
			metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(tn.Type, extName.Type).Inc()
			logger.V(logging.DEBUG).Info("extract failed", "source", tn, "extractor", extName, "err", err)
		}
	}
}
