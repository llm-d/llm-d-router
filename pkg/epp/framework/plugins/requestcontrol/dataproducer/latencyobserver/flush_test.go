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

package latencyobserver

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testEndpointID is the namespaced name of the endpoint the tests observe.
const testEndpointID = "default/a"

// flushConfig is tuned for tests: a short bucket so the floor path runs, and a
// low minRequests so a handful of observations is enough to be trusted.
func flushConfig() Config {
	cfg := DefaultConfig
	cfg.WindowSize, cfg.MaxRequests, cfg.MinRequests = 64, 16, 4
	cfg.BucketDuration, cfg.BucketHistorySize = "1s", 4
	return cfg
}

// flushAt recomputes the endpoint's snapshot at a controlled time. Dispatch
// reads time.Now(), so tests drive publish directly to steer the bucket clock.
func flushAt(p *Observer, now time.Time) {
	p.publish(context.Background(), testEndpointID, p.stateFor(testEndpointID), now)
}

func TestResolveConfig(t *testing.T) {
	resolved, err := DefaultConfig.resolve()
	require.NoError(t, err)
	assert.Equal(t, time.Second, resolved.interval)
	assert.Equal(t, 100, resolved.maxRequests)
	assert.InDelta(t, 0.25, resolved.lowPercentile, 1e-9, "percentiles resolve to fractions")

	invalid := map[string]func(*Config){
		"zero window":              func(c *Config) { c.WindowSize = 0 },
		"maxRequests above window": func(c *Config) { c.MaxRequests = c.WindowSize + 1 },
		"zero minRequests":         func(c *Config) { c.MinRequests = 0 },
		"bucket history below two": func(c *Config) { c.BucketHistorySize = 1 },
		"percentiles inverted":     func(c *Config) { c.LowPercentile, c.TypicalPercentile = 50, 25 },
		"unparsable duration":      func(c *Config) { c.IntervalDuration = "soon" },
		"non-positive duration":    func(c *Config) { c.BucketDuration = "-1m" },
	}
	for name, mutate := range invalid {
		t.Run(name, func(t *testing.T) {
			cfg := DefaultConfig
			mutate(&cfg)
			_, err := cfg.resolve()
			require.Error(t, err)
		})
	}
}

func TestFlush(t *testing.T) {
	now := time.Now()

	t.Run("publishes the load anchors from the short window", func(t *testing.T) {
		p := newObserverWithConfig(t, flushConfig())
		// TTFT rises with the in-flight count it was dispatched at, which is the
		// relationship the scorer's curve captures.
		for i := range 10 {
			p.record(testEndpointID, 0.1+float64(i)*0.1, int64(i), now.Add(time.Duration(i)*time.Millisecond))
		}

		flushAt(p, now.Add(time.Second))

		s := p.stateFor(testEndpointID).published.Load()
		require.NotNil(t, s)
		assert.Equal(t, 10, s.RecentRequestCount)
		assert.Equal(t, 4, s.CalibrationThreshold)
		assert.Less(t, s.LowLoadTTFT, s.TypicalLoadTTFT, "P25 must sit below P50")
		assert.Less(t, s.InflightAtLowLoad, s.InflightAtTypicalLoad, "the faster band must be less loaded")
	})

	t.Run("the floor arrives only once a bucket closes", func(t *testing.T) {
		p := newObserverWithConfig(t, flushConfig())
		for i := range 10 {
			ttft := 1.0
			if i < 2 {
				ttft = 0.1 // a fast tail the floor must follow
			}
			p.record(testEndpointID, ttft, 1, now.Add(time.Duration(i)*time.Millisecond))
		}

		flushAt(p, now) // starts the bucket clock
		assert.Zero(t, p.stateFor(testEndpointID).published.Load().FloorTTFT)

		// One bucket later exactly: the bucket window looks back bucketDuration,
		// so flushing further out would close an empty bucket.
		flushAt(p, now.Add(time.Second))
		s := p.stateFor(testEndpointID).published.Load()
		assert.Greater(t, s.FloorTTFT, 0.0)
		assert.Less(t, s.FloorTTFT, 1.0, "tracks the fast requests, not the bulk")
	})

	t.Run("Floor withholds the value below minRequests", func(t *testing.T) {
		p := newObserverWithConfig(t, flushConfig()) // minRequests 4
		for i := range 2 {
			p.record(testEndpointID, 0.1, 1, now.Add(time.Duration(i)*time.Millisecond))
		}
		flushAt(p, now)
		flushAt(p, now.Add(time.Second))

		s := p.stateFor(testEndpointID).published.Load()
		assert.Greater(t, s.FloorTTFT, 0.0, "the raw floor is computed")
		assert.Zero(t, s.Floor(), "but Floor() withholds it")
	})
}

// The datalayer drives the recompute: one Dispatch per endpoint publishes that
// endpoint's snapshot, and the plugin accepts no extractors.
func TestDispatch(t *testing.T) {
	ctx := context.Background()
	p := newObserverWithConfig(t, flushConfig())
	p.record(testEndpointID, 0.3, 1, time.Now())

	require.NoError(t, p.Dispatch(ctx, newDataEndpoint()))
	assert.NotNil(t, p.stateFor(testEndpointID).published.Load())
	assert.Equal(t, time.Second, p.Interval(), "the configured interval is the dispatch cadence")
	require.NoError(t, p.Dispatch(ctx, nil), "a nil endpoint is a no-op")
	assert.Error(t, p.AppendExtractor(p), "this dispatcher sources nothing")
}
