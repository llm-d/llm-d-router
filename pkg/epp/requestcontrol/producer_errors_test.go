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

package requestcontrol

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type chainProducer struct {
	name     string
	produces fwkplugin.DataKey
	requires []fwkplugin.DataKey
	optional []fwkplugin.DataKey
	err      error
	// hang makes Produce wait until its context is done.
	hang bool
	// timeout, when positive, is returned by ProduceTimeout.
	timeout time.Duration
	// onProduce, when set, runs at the start of Produce.
	onProduce func()
	ran       atomic.Bool
}

func (p *chainProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "chain", Name: p.name}
}

func (p *chainProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.produces: 0}
}

func (p *chainProducer) Consumes() fwkplugin.DataDependencies {
	deps := fwkplugin.DataDependencies{Required: map[fwkplugin.DataKey]any{}, Optional: map[fwkplugin.DataKey]any{}}
	for _, key := range p.requires {
		deps.Required[key] = 0
	}
	for _, key := range p.optional {
		deps.Optional[key] = 0
	}
	return deps
}

func (p *chainProducer) ProduceTimeout() time.Duration {
	return p.timeout
}

func (p *chainProducer) Produce(ctx context.Context, _ *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	p.ran.Store(true)
	if p.onProduce != nil {
		p.onProduce()
	}
	if p.hang {
		<-ctx.Done()
		return ctx.Err()
	}
	return p.err
}

func newDirectorWithProducers(producers ...*chainProducer) *Director {
	plugins := make([]fwkrc.DataProducer, len(producers))
	for i, p := range producers {
		plugins[i] = p
	}
	return &Director{requestControlPlugins: Config{dataProducerPlugins: plugins}}
}

// A failing producer skips only the producers that require its output, directly
// or through another skipped producer; independent producers and producers that
// only optionally consume its output still run.
func TestRunDataProducerPluginsContinuesPastFailure(t *testing.T) {
	keyA, keyB, keyC := fwkplugin.NewDataKey("a", "a"), fwkplugin.NewDataKey("b", "b"), fwkplugin.NewDataKey("c", "c")
	a := &chainProducer{name: "a", produces: keyA, err: errors.New("unsupported body")}
	b := &chainProducer{name: "b", produces: keyB}
	c := &chainProducer{name: "c", produces: keyC, requires: []fwkplugin.DataKey{keyA}}
	d := &chainProducer{name: "d", produces: fwkplugin.NewDataKey("d", "d"), requires: []fwkplugin.DataKey{keyC}}
	e := &chainProducer{name: "e", produces: fwkplugin.NewDataKey("e", "e"), requires: []fwkplugin.DataKey{keyB}}
	f := &chainProducer{name: "f", produces: fwkplugin.NewDataKey("f", "f"), optional: []fwkplugin.DataKey{keyA}}

	err := newDirectorWithProducers(a, b, c, d, e, f).runDataProducerPlugins(context.Background(), &fwksched.InferenceRequest{}, nil)

	require.Error(t, err)
	assert.ErrorContains(t, err, `DataProducer "a/chain" failed: unsupported body`)
	assert.ErrorContains(t, err, `DataProducer "c/chain" skipped`)
	assert.ErrorContains(t, err, `DataProducer "d/chain" skipped`)
	assert.True(t, a.ran.Load())
	assert.True(t, b.ran.Load(), "b does not depend on a")
	assert.False(t, c.ran.Load(), "c requires a's output")
	assert.False(t, d.ran.Load(), "d requires the output of c, which was skipped")
	assert.True(t, e.ran.Load(), "e requires only b's output")
	assert.True(t, f.ran.Load(), "f only optionally consumes a's output")
}

// A producer that times out counts as failed: the producers that require its
// output are skipped and the others still run.
func TestRunDataProducerPluginsContinuesPastTimeout(t *testing.T) {
	keyA := fwkplugin.NewDataKey("a", "a")
	a := &chainProducer{name: "a", produces: keyA, hang: true, timeout: 20 * time.Millisecond}
	b := &chainProducer{name: "b", produces: fwkplugin.NewDataKey("b", "b")}
	c := &chainProducer{name: "c", produces: fwkplugin.NewDataKey("c", "c"), requires: []fwkplugin.DataKey{keyA}}

	err := newDirectorWithProducers(a, b, c).runDataProducerPlugins(context.Background(), &fwksched.InferenceRequest{}, nil)

	require.ErrorIs(t, err, context.DeadlineExceeded)
	assert.ErrorContains(t, err, `DataProducer "c/chain" skipped: required input a/a was not produced`)
	assert.True(t, b.ran.Load(), "b does not depend on a")
	assert.False(t, c.ran.Load(), "c requires a's output")
}

// Once the request's context is done, no further producer is started, and the
// cancellation is reported once.
func TestRunDataProducerPluginsStopsWhenRequestIsDone(t *testing.T) {
	t.Run("done before the first producer", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		a := &chainProducer{name: "a", produces: fwkplugin.NewDataKey("a", "a")}

		err := newDirectorWithProducers(a).runDataProducerPlugins(ctx, &fwksched.InferenceRequest{}, nil)

		require.ErrorIs(t, err, context.Canceled)
		assert.Never(t, a.ran.Load, 50*time.Millisecond, 5*time.Millisecond, "no producer starts for a request that is done")
	})
	t.Run("cancelled while a producer runs", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		a := &chainProducer{name: "a", produces: fwkplugin.NewDataKey("a", "a"), hang: true, onProduce: cancel}
		b := &chainProducer{name: "b", produces: fwkplugin.NewDataKey("b", "b")}

		err := newDirectorWithProducers(a, b).runDataProducerPlugins(ctx, &fwksched.InferenceRequest{}, nil)

		require.ErrorIs(t, err, context.Canceled)
		assert.Equal(t, 1, strings.Count(err.Error(), context.Canceled.Error()))
		assert.False(t, b.ran.Load(), "no producer starts after the request is done")
	})
}

func TestRunDataProducerPluginsWithoutFailures(t *testing.T) {
	keyA := fwkplugin.NewDataKey("a", "a")
	a := &chainProducer{name: "a", produces: keyA}
	b := &chainProducer{name: "b", produces: fwkplugin.NewDataKey("b", "b"), requires: []fwkplugin.DataKey{keyA}}

	require.NoError(t, newDirectorWithProducers(a, b).runDataProducerPlugins(context.Background(), &fwksched.InferenceRequest{}, nil))
	assert.True(t, a.ran.Load())
	assert.True(t, b.ran.Load())
}
