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
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/random"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

type admissionMetricsObserver struct {
	observed chan map[string]*fwkdl.Metrics
}

type failedAdmissionCost struct {
	*observedInFlightProducer
}

func (*failedAdmissionCost) PrepareWithoutPrefix(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return errors.New("cost unavailable")
}

func TestDirectorAdmissionDoesNotTreatFailedCostAsZero(t *testing.T) {
	endpoint := projectedEndpoint("cost-unavailable", "")
	h := newProjectedAdmissionHarness(t, time.Minute, endpoint)
	h.director.requestControlPlugins.WithDataProducerPlugins(&failedAdmissionCost{h.producer})
	_, result := h.start(h.ctx, "failed-cost", 50)
	err := waitAdmit(t, result)
	var serviceError errcommon.Error
	require.ErrorAs(t, err, &serviceError)
	require.Equal(t, errcommon.ServiceUnavailable, serviceError.Code)
	require.Zero(t, h.scheduler.calls.Load())
	require.Zero(t, h.producer.preCalls.Load())
	require.Zero(t, h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
}

type failedWarmAdmissionCost struct {
	*observedInFlightProducer
}

func (*failedWarmAdmissionCost) PrepareForAdmission(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return errors.New("warm cost unavailable")
}

func TestDirectorAdmissionDoesNotTreatFailedWarmCostAsZero(t *testing.T) {
	endpoint := projectedEndpoint("warm-cost-unavailable", "")
	h := newProjectedAdmissionHarness(t, time.Minute, endpoint)
	incumbent := h.seed(t, endpoint, 60)
	h.director.requestControlPlugins.WithDataProducerPlugins(&failedWarmAdmissionCost{h.producer})
	_, result := h.start(h.ctx, "failed-warm-cost", 50)
	var serviceError errcommon.Error
	require.ErrorAs(t, waitAdmit(t, result), &serviceError)
	require.Equal(t, errcommon.ServiceUnavailable, serviceError.Code)
	require.Zero(t, h.scheduler.calls.Load())
	require.Zero(t, h.producer.preCalls.Load())
	require.Equal(t, int64(60), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	h.finish(incumbent)
}

type admissionOrdinaryBarrier struct {
	entered chan struct{}
	release chan struct{}
}

func (*admissionOrdinaryBarrier) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "ordinary-barrier", Name: "ordinary-barrier"}
}
func (*admissionOrdinaryBarrier) Produces() map[fwkplugin.DataKey]any { return nil }
func (p *admissionOrdinaryBarrier) Produce(ctx context.Context, _ *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	close(p.entered)
	select {
	case <-p.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestDirectorAdmissionKeepsLoadLiveAfterRelease(t *testing.T) {
	a, b := projectedEndpoint("racing-a", ""), projectedEndpoint("full-b", "")
	h := newProjectedAdmissionHarness(t, time.Minute, a, b)
	incumbentA, incumbentB := h.seed(t, a, 60), h.seed(t, b, 100)
	barrier := &admissionOrdinaryBarrier{entered: make(chan struct{}), release: make(chan struct{})}
	t.Cleanup(func() {
		select {
		case <-barrier.release:
		default:
			close(barrier.release)
		}
	})
	h.director.requestControlPlugins.WithDataProducerPlugins(barrier, h.producer)
	_, result := h.start(h.ctx, "racing-incoming", 30)
	select {
	case <-barrier.entered:
	case <-time.After(time.Second):
		t.Fatal("ordinary producer did not start after admission")
	}
	extra := &fwksched.InferenceRequest{RequestID: "racing-extra", Body: projectedBody(20), SchedulingResult: incumbentA.SchedulingResult}
	require.NoError(t, h.producer.InFlightLoadProducer.PreRequest(h.ctx, extra, extra.SchedulingResult))
	close(barrier.release)
	require.ErrorContains(t, waitAdmit(t, result), "no endpoint fits")
	require.Zero(t, h.producer.preCalls.Load(), "a concurrent load change must still be visible to the final filter")
	require.Equal(t, int64(80), h.producer.GetTokens(a.GetMetadata().ID.String()))
	h.finish(extra)
	h.finish(incumbentA)
	h.finish(incumbentB)
}

func (*admissionMetricsObserver) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "admission-metrics-observer", Name: "admission-metrics-observer"}
}

func (*admissionMetricsObserver) Produces() map[fwkplugin.DataKey]any { return nil }

func (p *admissionMetricsObserver) Produce(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	metrics := make(map[string]*fwkdl.Metrics, len(endpoints))
	for _, endpoint := range endpoints {
		metrics[endpoint.GetMetadata().ID.Name] = endpoint.GetMetrics().Clone()
	}
	select {
	case p.observed <- metrics:
	default:
	}
	return nil
}

func TestDirectorAdmissionUsesMetricsUpdatedWhileQueued(t *testing.T) {
	endpoint := projectedEndpoint("metrics-updated", "")
	endpoint.UpdateMetrics(&fwkdl.Metrics{WaitingQueueSize: 1, UpdateTime: time.Now()})
	h := newProjectedAdmissionHarness(t, time.Minute, endpoint)
	incumbent := h.seed(t, endpoint, 100)
	observer := &admissionMetricsObserver{observed: make(chan map[string]*fwkdl.Metrics, 1)}
	h.director.requestControlPlugins.WithDataProducerPlugins(observer, h.producer)
	request, result := h.start(h.ctx, "fresh-metrics", 50)
	require.Equal(t, map[string]int64{"metrics-updated": 50}, awaitProjectedCosts(t, h))
	h.requireQueued(t, result)

	latest := &fwkdl.Metrics{
		WaitingQueueSize: 15, RunningRequestsSize: 23,
		KVCacheUsagePercent: 0.8, UpdateTime: time.Now(),
	}
	endpoint.UpdateMetrics(latest)
	h.finish(incumbent)
	require.NoError(t, waitAdmit(t, result))
	select {
	case observed := <-observer.observed:
		require.Contains(t, observed, "metrics-updated")
		got := observed["metrics-updated"]
		require.Equal(t, latest.WaitingQueueSize, got.WaitingQueueSize)
		require.Equal(t, latest.RunningRequestsSize, got.RunningRequestsSize)
		require.Equal(t, latest.KVCacheUsagePercent, got.KVCacheUsagePercent)
		require.Equal(t, latest.UpdateTime, got.UpdateTime)
	case <-time.After(time.Second):
		t.Fatal("ordinary producer did not observe the dispatched endpoints")
	}
	require.Equal(t, int64(1), h.producer.preCalls.Load())
	h.finish(request.SchedulingRequest)
	require.Zero(t, h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	require.Zero(t, h.producer.GetRequests(endpoint.GetMetadata().ID.String()))
}

type admissionCandidateObserver struct {
	delegate contracts.EndpointCandidates
	empty    chan struct{}
}

func (o *admissionCandidateObserver) Locate(ctx context.Context, metadata map[string]any) []fwkdl.Endpoint {
	endpoints := o.delegate.Locate(ctx, metadata)
	if len(endpoints) == 0 {
		select {
		case o.empty <- struct{}{}:
		default:
		}
	}
	return endpoints
}

func TestDirectorAdmissionDiscardsEmptySnapshotWhenEndpointAppears(t *testing.T) {
	h := newProjectedAdmissionHarness(t, time.Minute)
	observer := &admissionCandidateObserver{delegate: h.candidates, empty: make(chan struct{}, 1)}
	h.director.endpointCandidates = observer
	request, result := h.start(h.ctx, "endpoint-appears", 50)
	select {
	case <-observer.empty:
	case <-time.After(time.Second):
		t.Fatal("request preparation did not observe the empty pool")
	}
	h.requireQueued(t, result)

	endpoint := projectedEndpoint("new-endpoint", "")
	require.NoError(t, h.producer.Extract(h.ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint}))
	h.candidates.mu.Lock()
	h.candidates.result = []fwkdl.Endpoint{endpoint}
	h.candidates.mu.Unlock()
	require.NoError(t, waitAdmit(t, result))
	require.True(t, request.FlowControlAdmitted)
	require.Equal(t, "new-endpoint", request.TargetPod.ID.Name)
	require.Equal(t, int64(1), h.scheduler.calls.Load())
	require.Equal(t, int64(1), h.producer.preCalls.Load())
	require.Equal(t, int64(50), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	h.finish(request.SchedulingRequest)
	require.Zero(t, h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	require.Zero(t, h.producer.GetRequests(endpoint.GetMetadata().ID.String()))
}

type failedAdmissionTokenizer struct {
	admissionCalls atomic.Int64
	ordinaryCalls  atomic.Int64
}

func (*failedAdmissionTokenizer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "failed-admission-tokenizer", Name: "failed-admission-tokenizer"}
}

func (*failedAdmissionTokenizer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{tokenizer.TokenizedPromptDataKey: fwksched.TokenizedRequest{}}
}

func (p *failedAdmissionTokenizer) PrepareForAdmission(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	p.admissionCalls.Add(1)
	return errors.New("tokenizer preparation failed")
}

func (p *failedAdmissionTokenizer) Produce(_ context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	p.ordinaryCalls.Add(1)
	request.Body.TokenizedRequest = projectedBody(50).TokenizedRequest
	return nil
}

func TestDirectorAdmissionDoesNotRetryFailedTokenizer(t *testing.T) {
	endpoint := projectedEndpoint("tokenizer-failed", "")
	h := newProjectedAdmissionHarness(t, time.Minute, endpoint)
	incumbent := h.seed(t, endpoint, 60)
	producer := &failedAdmissionTokenizer{}
	h.director.requestControlPlugins.WithDataProducerPlugins(producer, h.producer)
	request, result := h.start(h.ctx, "failed-tokenization", 0)
	require.NoError(t, waitAdmit(t, result))
	require.Equal(t, int64(1), producer.admissionCalls.Load())
	require.Zero(t, producer.ordinaryCalls.Load(), "a failed tokenizer must not change the cost through an ordinary retry")
	require.Zero(t, request.SchedulingRequest.Body.TokenizedRequest.TokenCount())
	require.Equal(t, int64(1), h.scheduler.calls.Load())
	require.Equal(t, int64(1), h.producer.preCalls.Load())
	require.Equal(t, int64(60), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	require.Equal(t, int64(2), h.producer.GetRequests(endpoint.GetMetadata().ID.String()))
	h.finish(request.SchedulingRequest)
	require.Equal(t, int64(60), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	require.Equal(t, int64(1), h.producer.GetRequests(endpoint.GetMetadata().ID.String()))
	h.finish(incumbent)
	require.Zero(t, h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	require.Zero(t, h.producer.GetRequests(endpoint.GetMetadata().ID.String()))
}

func TestDirectorAdmissionPreservesProducerCandidatePool(t *testing.T) {
	a, b := projectedEndpoint("busy-cache-donor", ""), projectedEndpoint("idle-target", "")
	for _, endpoint := range []fwkdl.Endpoint{a, b} {
		metadata := endpoint.GetMetadata().Clone()
		metadata.Labels = nil
		endpoint.UpdateMetadata(metadata)
	}
	h := newProjectedAdmissionHarness(t, time.Minute, a, b)
	incumbent := h.seed(t, a, 100)
	a.GetAttributes().Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(50, 50, 1))
	observer := &admissionMetricsObserver{observed: make(chan map[string]*fwkdl.Metrics, 1)}
	h.director.requestControlPlugins.WithDataProducerPlugins(observer, h.producer)
	h.director.scheduler = scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{
			"decode": scheduling.NewSchedulerProfile().WithFilters(h.scheduler.filter).
				WithPicker(random.NewRandomPicker(1)),
		}))
	request, result := h.start(h.ctx, "producer-candidates", 50)
	require.NoError(t, waitAdmit(t, result))
	select {
	case observed := <-observer.observed:
		require.Len(t, observed, 2)
		require.Contains(t, observed, "busy-cache-donor", "producer inputs must retain donors excluded from scheduling by capacity")
		require.Contains(t, observed, "idle-target")
	case <-time.After(time.Second):
		t.Fatal("ordinary producer did not observe endpoint candidates")
	}
	require.Equal(t, "idle-target", request.TargetPod.ID.Name)
	require.Equal(t, int64(1), h.producer.preCalls.Load())
	require.Equal(t, int64(100), h.producer.GetTokens(a.GetMetadata().ID.String()))
	require.Equal(t, int64(50), h.producer.GetTokens(b.GetMetadata().ID.String()))
	h.finish(request.SchedulingRequest)
	require.Zero(t, h.producer.GetTokens(b.GetMetadata().ID.String()))
	h.finish(incumbent)
	require.Zero(t, h.producer.GetTokens(a.GetMetadata().ID.String()))
}
