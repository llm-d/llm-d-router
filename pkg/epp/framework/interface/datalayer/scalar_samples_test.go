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
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestScalarSampleFreshWithin(t *testing.T) {
	assert.False(t, ScalarSample{}.FreshWithin(time.Hour), "a never-observed sample is not fresh")
	assert.True(t, ScalarSample{ScrapeTime: time.Now()}.FreshWithin(time.Minute))
	assert.False(t, ScalarSample{ScrapeTime: time.Now().Add(-2 * time.Minute)}.FreshWithin(time.Minute))
}

func TestScalarSampleUpdate(t *testing.T) {
	origin := "mapping"
	t1 := time.Unix(100, 0)
	first := NewScalarSampleUpdate(nil, origin, t1, 1)
	first.Observe(ScalarFieldWaitingQueueSize, 4)
	first.ObserveCustom("custom", 7)
	m := &Metrics{ScalarSamples: first.Result()}

	queue, ok := m.ScalarSample(ScalarFieldWaitingQueueSize)
	require.True(t, ok)
	assert.Equal(t, ScalarSample{Value: 4, ScrapeTime: t1}, queue)
	_, ok = m.ScalarSample(ScalarFieldRunningRequestsSize)
	assert.False(t, ok, "an unobserved standard metric has no sample")
	custom, ok := m.CustomScalarSample("custom")
	require.True(t, ok)
	assert.Equal(t, ScalarSample{Value: 7, ScrapeTime: t1}, custom)
	_, ok = m.CustomScalarSample(ScalarFieldWaitingQueueSize.String())
	assert.False(t, ok, "standard and custom samples use separate keys")

	t.Run("observation copies the previous snapshot", func(t *testing.T) {
		t2 := t1.Add(time.Second)
		next := NewScalarSampleUpdate(m.ScalarSamples, origin, t2, 1)
		next.Observe(ScalarFieldRunningRequestsSize, 2)
		next.ObserveCustom("other", 1)
		later := &Metrics{ScalarSamples: next.Result()}

		queue, _ := later.ScalarSample(ScalarFieldWaitingQueueSize)
		assert.Equal(t, t1, queue.ScrapeTime, "an unobserved metric keeps its sample")
		running, _ := later.ScalarSample(ScalarFieldRunningRequestsSize)
		assert.Equal(t, ScalarSample{Value: 2, ScrapeTime: t2}, running)
		custom, _ := later.CustomScalarSample("custom")
		assert.Equal(t, t1, custom.ScrapeTime)

		_, ok := m.ScalarSample(ScalarFieldRunningRequestsSize)
		assert.False(t, ok, "the previous snapshot is not modified")
		_, ok = m.CustomScalarSample("other")
		assert.False(t, ok, "the previous custom map is not modified")
	})

	t.Run("no observation keeps the previous snapshot", func(t *testing.T) {
		next := NewScalarSampleUpdate(m.ScalarSamples, origin, t1.Add(time.Second), 1)
		assert.Same(t, m.ScalarSamples, next.Result())
	})

	t.Run("mapping change discards samples", func(t *testing.T) {
		next := NewScalarSampleUpdate(m.ScalarSamples, "other-mapping", t1.Add(time.Second), 1)
		assert.Nil(t, next.Result())

		next.Observe(ScalarFieldRunningRequestsSize, 1)
		moved := &Metrics{ScalarSamples: next.Result()}
		_, ok := moved.ScalarSample(ScalarFieldWaitingQueueSize)
		assert.False(t, ok)
		_, ok = moved.CustomScalarSample("custom")
		assert.False(t, ok)
	})
}

func TestScalarSampleReadsOfNilAndUnknown(t *testing.T) {
	var none *Metrics
	_, ok := none.ScalarSample(ScalarFieldWaitingQueueSize)
	assert.False(t, ok)
	_, ok = NewMetrics().CustomScalarSample("custom")
	assert.False(t, ok)

	update := NewScalarSampleUpdate(nil, "mapping", time.Now(), 0)
	update.Observe(scalarFieldCount, 1)
	_, ok = (&Metrics{ScalarSamples: update.Result()}).ScalarSample(scalarFieldCount)
	assert.False(t, ok)
}

func TestMetricsCloneSharesScalarSamples(t *testing.T) {
	update := NewScalarSampleUpdate(nil, "mapping", time.Now(), 0)
	update.Observe(ScalarFieldWaitingQueueSize, 3)
	m := NewMetrics()
	m.ScalarSamples = update.Result()

	clone := m.Clone()
	assert.Same(t, m.ScalarSamples, clone.ScalarSamples)
	if diff := cmp.Diff(m, clone); diff != "" {
		t.Errorf("Unexpected output (-want +got): %v", diff)
	}

	other := NewScalarSampleUpdate(nil, "mapping", time.Now(), 0)
	other.Observe(ScalarFieldWaitingQueueSize, 4)
	clone.ScalarSamples = other.Result()
	assert.NotEmpty(t, cmp.Diff(m, clone), "go-cmp reports differing samples")
}

func TestScalarSampleAllocations(t *testing.T) {
	update := NewScalarSampleUpdate(nil, "mapping", time.Now(), 1)
	update.Observe(ScalarFieldWaitingQueueSize, 3)
	update.ObserveCustom("custom", 1)
	m := &Metrics{
		ActiveModels:  map[string]int{"a": 1},
		WaitingModels: map[string]int{},
		ScalarSamples: update.Result(),
	}

	reads := testing.AllocsPerRun(100, func() {
		_, _ = m.ScalarSample(ScalarFieldWaitingQueueSize)
		_, _ = m.CustomScalarSample("custom")
		_, _ = m.CustomScalarSample("missing")
	})
	assert.Zero(t, reads, "reading a sample must not allocate")

	without := *m
	without.ScalarSamples = nil
	withSamples := testing.AllocsPerRun(100, func() { _ = m.Clone() })
	withoutSamples := testing.AllocsPerRun(100, func() { _ = without.Clone() })
	assert.Equal(t, withoutSamples, withSamples, "samples must add no allocations to Clone")

	// One snapshot and one custom slice per scrape that observes metrics.
	scrape := testing.AllocsPerRun(100, func() {
		next := NewScalarSampleUpdate(m.ScalarSamples, "mapping", time.Now(), 1)
		next.Observe(ScalarFieldWaitingQueueSize, 4)
		next.ObserveCustom("custom", 2)
		_ = next.Result()
	})
	assert.LessOrEqual(t, scrape, 2.0)
}
