/*
Copyright 2025 The llm-d Authors.

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
	"sync/atomic"
	"testing"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	// collectorBenchWarmupPolls moves goroutine start-up and first-poll
	// costs, such as per-label metric children, out of the timed window.
	collectorBenchWarmupPolls = 10
	// collectorBenchTickWait bounds every wait on the collection goroutine.
	collectorBenchTickWait = 10 * time.Second
)

// collectorBenchTicker hands ticks to the collector over an unbuffered
// channel. A send completes only when the collection goroutine is back in
// its select, so an accepted tick proves the previous poll finished,
// including the bookkeeping the collector does after Dispatch returns. The
// buffered mock ticker cannot give that guarantee and drops ticks when full.
type collectorBenchTicker struct{ ch chan time.Time }

func (t *collectorBenchTicker) Channel() <-chan time.Time { return t.ch }
func (t *collectorBenchTicker) Stop()                     {}

// collectorBenchSource is a PollingDispatcher whose Dispatch returns nil at
// once, so the measurement covers the collector's per-poll path and not I/O.
type collectorBenchSource struct{ calls atomic.Int64 }

func (s *collectorBenchSource) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "bench-type", Name: "bench"}
}
func (s *collectorBenchSource) Dispatch(context.Context, fwkdl.Endpoint) error {
	s.calls.Add(1)
	return nil
}
func (s *collectorBenchSource) AppendExtractor(fwkplugin.Plugin) error { return nil }
func (s *collectorBenchSource) Interval() time.Duration                { return 0 }

// BenchmarkCollectorPoll measures one completed poll through Collector.run:
// tick receipt, context set-up, Dispatch, and the bookkeeping that follows.
//
// Accounting: one operation is one accepted tick with one dispatcher. Ticks
// are delivered over an unbuffered channel, so tick i+1 is accepted only after
// poll i has completed. After the b.N measured ticks one more tick is sent as
// a barrier for poll b.N, the timer is stopped, and the benchmark waits for
// that extra poll to be entered before Stop, so shutdown never cancels a
// measured poll. The extra poll is not measured. The poll in flight when the
// timer starts and the one entered as it stops each contribute a fraction of
// a poll; both are constant per run and shrink relative to b.N.
//
// Allocation counts are process-wide for the timed window, so they include
// the collection goroutine. The sender does no per-tick allocation: the tick
// value is the zero time and the wait reuses one timer.
//
// Limitation: a channel hand-off and goroutine switch per tick are part of
// every operation and are also part of the production path; a real source
// adds network and extraction time that dwarfs the bookkeeping measured here.
func BenchmarkCollectorPoll(b *testing.B) {
	ticker := &collectorBenchTicker{ch: make(chan time.Time)}
	src := &collectorBenchSource{}
	c := NewCollector()
	if err := c.Start(context.Background(), ticker, defaultEndpoint(), []fwkdl.PollingDispatcher{src}); err != nil {
		b.Fatal(err)
	}
	// The collector is never left holding a tick, so Stop cannot block here.
	defer c.Stop()

	wait := time.NewTimer(collectorBenchTickWait)
	defer wait.Stop()
	tick := func() {
		wait.Reset(collectorBenchTickWait)
		select {
		case ticker.ch <- time.Time{}:
		case <-wait.C:
			b.Fatal("collector did not accept a tick")
		}
	}

	for range collectorBenchWarmupPolls {
		tick()
	}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		tick()
	}
	tick() // barrier: accepted only after poll b.N completed
	b.StopTimer()

	want := int64(collectorBenchWarmupPolls + b.N + 1)
	wait.Reset(collectorBenchTickWait)
	for src.calls.Load() != want {
		select {
		case <-wait.C:
			b.Fatalf("expected %d dispatches, saw %d", want, src.calls.Load())
		default:
			time.Sleep(10 * time.Microsecond)
		}
	}
	c.Stop()
	if got := src.calls.Load(); got != want {
		b.Fatalf("expected %d dispatches after Stop, saw %d", want, got)
	}
}
