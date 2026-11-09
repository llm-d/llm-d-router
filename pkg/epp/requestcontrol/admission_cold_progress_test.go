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
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

type blockedInitialAdmissionPrefix struct {
	entered      chan struct{}
	release      chan struct{}
	prepareCalls atomic.Int64
	produceCalls atomic.Int64
	active       atomic.Bool
	overlapped   atomic.Bool
}

func (p *blockedInitialAdmissionPrefix) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "blocked-initial-prefix", Name: "blocked-initial-prefix"}
}

func (p *blockedInitialAdmissionPrefix) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{attrprefix.PrefixCacheMatchInfoDataKey: attrprefix.PrefixCacheMatchInfo{}}
}

func (p *blockedInitialAdmissionPrefix) ProduceTimeout() time.Duration { return time.Minute }

func (p *blockedInitialAdmissionPrefix) PrepareForAdmission(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	p.active.Store(true)
	defer p.active.Store(false)
	if p.prepareCalls.Add(1) == 1 {
		close(p.entered)
		select {
		case <-p.release:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	p.publish(request, endpoints)
	return nil
}

func (p *blockedInitialAdmissionPrefix) Produce(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	if p.active.Load() {
		p.overlapped.Store(true)
	}
	p.produceCalls.Add(1)
	p.publish(request, endpoints)
	return nil
}

func (*blockedInitialAdmissionPrefix) publish(request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) {
	for _, endpoint := range endpoints {
		endpoint.Put(attrprefix.PrefixCacheMatchInfoDataKey,
			attrprefix.NewPrefixCacheMatchInfo(0, request.Body.TokenizedRequest.TokenCount(), 1))
	}
}

func TestDirectorAdmissionColdProgressDispatchesDuringInitialPrefixLookup(t *testing.T) {
	endpoint := projectedEndpoint("cold-progress", "")
	h := newProjectedAdmissionHarness(t, time.Minute, endpoint)
	incumbent := h.seed(t, endpoint, 60)
	prefix := &blockedInitialAdmissionPrefix{entered: make(chan struct{}), release: make(chan struct{})}
	t.Cleanup(func() { close(prefix.release) })
	h.director.requestControlPlugins.WithDataProducerPlugins(prefix, h.producer)
	request, result := h.start(h.ctx, "cold-progress-incoming", 50)
	require.Equal(t, map[string]int64{"cold-progress": 50}, awaitProjectedCosts(t, h))
	select {
	case <-prefix.entered:
	case <-time.After(time.Second):
		t.Fatal("initial prefix lookup did not start")
	}
	h.requireQueued(t, result)
	h.finish(incumbent)
	require.NoError(t, waitAdmit(t, result), "available undiscounted capacity must not wait for the pending prefix lookup")
	select {
	case <-prefix.release:
		t.Fatal("request completed only after releasing the prefix lookup")
	default:
	}
	require.Equal(t, int64(1), prefix.prepareCalls.Load())
	require.LessOrEqual(t, prefix.produceCalls.Load(), int64(1))
	require.False(t, prefix.overlapped.Load(), "cancellation does not prove the preparation goroutine has exited")
	require.Equal(t, int64(1), h.scheduler.calls.Load())
	require.Equal(t, int64(1), h.producer.preCalls.Load())
	require.Equal(t, int64(50), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	h.finish(request.SchedulingRequest)
}

type uncooperativeAdmissionPrefix struct {
	*blockedInitialAdmissionPrefix
}

func (p *uncooperativeAdmissionPrefix) PrepareForAdmission(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	p.active.Store(true)
	defer p.active.Store(false)
	p.prepareCalls.Add(1)
	close(p.entered)
	<-p.release
	p.publish(request, endpoints)
	return nil
}

func TestDirectorColdDispatchDoesNotOverlapUnfinishedPreparation(t *testing.T) {
	endpoint := projectedEndpoint("cold-unfinished", "")
	h := newProjectedAdmissionHarness(t, time.Minute, endpoint)
	incumbent := h.seed(t, endpoint, 60)
	prefix := &uncooperativeAdmissionPrefix{&blockedInitialAdmissionPrefix{
		entered: make(chan struct{}), release: make(chan struct{}),
	}}
	t.Cleanup(func() { close(prefix.release) })
	h.director.requestControlPlugins.WithDataProducerPlugins(prefix, h.producer)
	request, result := h.start(h.ctx, "cold-unfinished-incoming", 50)
	require.Equal(t, map[string]int64{"cold-unfinished": 50}, awaitProjectedCosts(t, h))
	select {
	case <-prefix.entered:
	case <-time.After(time.Second):
		t.Fatal("prefix preparation did not start")
	}
	h.finish(incumbent)
	require.NoError(t, waitAdmit(t, result))
	require.Zero(t, prefix.produceCalls.Load(), "Produce must not overlap the still-running preparation")
	require.False(t, prefix.overlapped.Load())
	require.Equal(t, int64(1), h.producer.preCalls.Load(), "independent load accounting must still run")
	require.Equal(t, int64(50), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	h.finish(request.SchedulingRequest)
}
