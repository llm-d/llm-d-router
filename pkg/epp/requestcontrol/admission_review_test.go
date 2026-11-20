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

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type admissionReviewProducer struct {
	name         string
	key          plugin.DataKey
	firstError   error
	prepareCalls atomic.Int32
	produceCalls atomic.Int32
}

func (p *admissionReviewProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: p.name, Name: p.name}
}

func (p *admissionReviewProducer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.key: int64(0)}
}

func (p *admissionReviewProducer) PrepareForAdmission(_ context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	if p.prepareCalls.Add(1) == 1 && p.firstError != nil {
		return p.firstError
	}
	request.PutAttribute(p.key, int64(42))
	return nil
}

func (p *admissionReviewProducer) Produce(_ context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	p.produceCalls.Add(1)
	request.PutAttribute(p.key, int64(42))
	return nil
}

func runSchedulingProducersAfterPreparation(ctx context.Context, d *Director, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint, failures map[plugin.TypedName]error) error {
	return d.runDataProducers(ctx, fwkrh.DataProducerExtensionPoint, d.requestControlPlugins.dataProducerPlugins,
		request, endpoints, map[string]struct{}{}, failures)
}

func TestDataProducerPreparationFailureDoesNotSkipOtherProducers(t *testing.T) {
	first := &admissionReviewProducer{name: "failed-preparation"}
	next := &admissionReviewProducer{name: "following-producer"}
	failure := errors.New("preparation unavailable")
	d := &Director{requestControlPlugins: Config{dataProducerPlugins: []fwkrh.DataProducer{first, next}}}
	err := runSchedulingProducersAfterPreparation(t.Context(), d, &scheduling.InferenceRequest{}, nil,
		map[plugin.TypedName]error{first.TypedName(): failure})
	require.ErrorIs(t, err, failure)
	require.Equal(t, int32(1), next.produceCalls.Load(), "preparation failure must not skip independent producers")
}

func TestQueuedPreparationRetriesCompletedTransientFailure(t *testing.T) {
	p := &admissionReviewProducer{
		name: "retry-completed-preparation", key: plugin.NewDataKey("retry-completed-value", "retry-completed-preparation"),
		firstError: errors.New("temporary lookup failure"),
	}
	d := &Director{
		endpointCandidates:    &mocks.MockEndpointCandidates{Candidates: []datalayer.Endpoint{datalayer.NewEndpoint(nil, nil)}},
		requestControlPlugins: Config{dataProducerPlugins: []fwkrh.DataProducer{p}},
	}
	prepare := d.queuedPreparation(&scheduling.InferenceRequest{}, nil)
	fits := func(*contracts.PreparedRequest) bool { return false }
	publish := func(*contracts.PreparedRequest) {}
	first := prepare(t.Context(), fits, publish)
	_, ok := first.Request.GetAttribute(p.key)
	require.False(t, ok)
	second := prepare(t.Context(), fits, publish)
	require.Equal(t, int32(2), p.prepareCalls.Load(), "completed failures must not disable subsequent refreshes")
	value, ok := second.Request.GetAttribute(p.key)
	require.True(t, ok)
	require.Equal(t, int64(42), value)
	require.Empty(t, second.ProducerErrors, "a successful refresh must not retain the previous error")
}

func TestQueuedPreparationFailureStaysBoundToAdmittedSnapshot(t *testing.T) {
	p := &admissionReviewProducer{
		name: "finished-preparation", key: plugin.NewDataKey("finished-value", "finished-preparation"),
		firstError: errors.New("temporary lookup failure"),
	}
	d := &Director{
		endpointCandidates:    &mocks.MockEndpointCandidates{Candidates: []datalayer.Endpoint{datalayer.NewEndpoint(nil, nil)}},
		requestControlPlugins: Config{dataProducerPlugins: []fwkrh.DataProducer{p}},
	}
	prepare := d.queuedPreparation(&scheduling.InferenceRequest{}, nil)
	fits := func(*contracts.PreparedRequest) bool { return false }
	publish := func(*contracts.PreparedRequest) {}
	prepared := prepare(t.Context(), fits, publish)
	recovered := prepare(t.Context(), fits, publish)
	require.Empty(t, recovered.ProducerErrors)
	require.Error(t, runSchedulingProducersAfterPreparation(t.Context(), d, prepared.Request, prepared.Endpoints, prepared.ProducerErrors))
	require.Zero(t, p.produceCalls.Load(), "a later refresh must not change an earlier snapshot's cost contract")
}

type admissionTimeoutReviewProducer struct {
	admissionReviewProducer
	release  <-chan struct{}
	finished chan struct{}
}

func (*admissionTimeoutReviewProducer) ProduceTimeout() time.Duration { return 10 * time.Millisecond }

func (p *admissionTimeoutReviewProducer) PrepareForAdmission(_ context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	if p.prepareCalls.Add(1) == 1 {
		<-p.release
		request.PutAttribute(p.key, int64(-1))
		close(p.finished)
		return nil
	}
	request.PutAttribute(p.key, int64(42))
	return nil
}

func TestQueuedPreparationRetriesTimeoutOnlyAfterCompletion(t *testing.T) {
	release := make(chan struct{})
	p := &admissionTimeoutReviewProducer{
		admissionReviewProducer: admissionReviewProducer{
			name: "pending-preparation", key: plugin.NewDataKey("pending-value", "pending-preparation"),
		},
		release: release, finished: make(chan struct{}),
	}
	released := false
	t.Cleanup(func() {
		if !released {
			close(release)
		}
	})
	d := &Director{
		endpointCandidates:    &mocks.MockEndpointCandidates{Candidates: []datalayer.Endpoint{datalayer.NewEndpoint(nil, nil)}},
		requestControlPlugins: Config{dataProducerPlugins: []fwkrh.DataProducer{p}},
	}
	original := &scheduling.InferenceRequest{}
	prepare := d.queuedPreparation(original, nil)
	fits := func(*contracts.PreparedRequest) bool { return false }
	publish := func(*contracts.PreparedRequest) {}
	first := prepare(t.Context(), fits, publish)
	second := prepare(t.Context(), fits, publish)
	require.Equal(t, int32(1), p.prepareCalls.Load(), "an unfinished timeout must not overlap another invocation")
	require.Error(t, runSchedulingProducersAfterPreparation(t.Context(), d, second.Request, second.Endpoints, second.ProducerErrors))
	require.Zero(t, p.produceCalls.Load())
	close(release)
	released = true
	select {
	case <-p.finished:
	case <-time.After(time.Second):
		t.Fatal("timed-out preparation did not finish")
	}
	var recovered *contracts.PreparedRequest
	require.Eventually(t, func() bool {
		recovered = prepare(t.Context(), fits, publish)
		return p.prepareCalls.Load() == 2
	}, time.Second, time.Millisecond)
	for _, request := range []*scheduling.InferenceRequest{original, first.Request, second.Request} {
		_, ok := request.GetAttribute(p.key)
		require.False(t, ok, "late writes must stay on the discarded snapshot")
	}
	value, ok := recovered.Request.GetAttribute(p.key)
	require.True(t, ok)
	require.Equal(t, int64(42), value)
	require.Empty(t, recovered.ProducerErrors)
}

func TestQueuedPreparationSealPreventsWarmStart(t *testing.T) {
	p := &admissionReviewProducer{name: "sealed-preparation"}
	d := &Director{
		endpointCandidates:    &mocks.MockEndpointCandidates{Candidates: []datalayer.Endpoint{datalayer.NewEndpoint(nil, nil)}},
		requestControlPlugins: Config{dataProducerPlugins: []fwkrh.DataProducer{p}},
	}
	d.queuedPreparation(&scheduling.InferenceRequest{}, nil)(
		t.Context(), func(*contracts.PreparedRequest) bool { return false },
		func(cold *contracts.PreparedRequest) {
			require.Empty(t, cold.SealPreparation())
		})
	require.Zero(t, p.prepareCalls.Load(), "dispatch handoff must prevent a later warm invocation")
}

func TestQueuedPreparationColdFallbackTracksPendingWork(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	p := &admissionTimeoutReviewProducer{
		admissionReviewProducer: admissionReviewProducer{
			name: "fallback-preparation", key: plugin.NewDataKey("fallback-value", "fallback-preparation"),
		},
		release: release, finished: make(chan struct{}),
	}
	d := &Director{
		endpointCandidates:    &mocks.MockEndpointCandidates{Candidates: []datalayer.Endpoint{datalayer.NewEndpoint(nil, nil)}},
		requestControlPlugins: Config{dataProducerPlugins: []fwkrh.DataProducer{p}},
	}
	prepared := d.queuedPreparation(&scheduling.InferenceRequest{}, nil)(
		t.Context(), func(*contracts.PreparedRequest) bool { return false }, func(*contracts.PreparedRequest) {})
	cold := prepared.WithoutPrefix
	require.NotNil(t, cold)
	require.Empty(t, cold.ProducerErrors, "the cold snapshot precedes the warm lookup")
	pending := cold.SealPreparation()
	require.Contains(t, pending, p.TypedName())
	require.Error(t, runSchedulingProducersAfterPreparation(t.Context(), d, cold.Request, cold.Endpoints, pending))
	require.Zero(t, p.produceCalls.Load())
	require.Empty(t, cold.ProducerErrors, "shared liveness must not mutate the published failure snapshot")
}
