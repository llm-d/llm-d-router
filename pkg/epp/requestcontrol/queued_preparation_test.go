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

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type queuedLateProducer struct {
	release  chan struct{}
	finished chan struct{}
	calls    atomic.Int32
}

func (p *queuedLateProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "late-queued-producer", Name: "late-queued-producer"}
}
func (p *queuedLateProducer) Produces() map[plugin.DataKey]any { return nil }
func (p *queuedLateProducer) ProduceTimeout() time.Duration    { return time.Millisecond }
func (p *queuedLateProducer) PrepareForAdmission(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) error {
	return p.Produce(ctx, request, endpoints)
}
func (p *queuedLateProducer) Produce(_ context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	p.calls.Add(1)
	<-p.release
	request.Body.TokenizedRequest.CacheSalt = "late"
	request.Body.TokenizedRequest.Prompts[0].TokenIDs[0] = 42
	request.Body.TokenizedRequest = requesthandling.NewTokenizedRequest([][]uint32{{9, 10}})
	close(p.finished)
	return nil
}

func TestQueuedPreparationDiscardsTimedOutWrites(t *testing.T) {
	p := &queuedLateProducer{release: make(chan struct{}), finished: make(chan struct{})}
	request := &scheduling.InferenceRequest{Body: &requesthandling.InferenceRequestBody{
		TokenizedRequest: requesthandling.NewTokenizedRequest([][]uint32{{1}}),
	}}
	d := &Director{
		endpointCandidates:    &mocks.MockEndpointCandidates{Candidates: []datalayer.Endpoint{datalayer.NewEndpoint(nil, nil)}},
		requestControlPlugins: Config{dataProducerPlugins: []fwkrc.DataProducer{p}},
	}
	prepare := d.queuedPreparation(request, nil)
	first := prepare(t.Context(), func(*contracts.PreparedRequest) bool { return false }, func(*contracts.PreparedRequest) {})
	require.Equal(t, uint32(1), first.Request.Body.TokenizedRequest.Prompts[0].TokenIDs[0])
	second := prepare(t.Context(), func(*contracts.PreparedRequest) bool { return false }, func(*contracts.PreparedRequest) {})
	require.Equal(t, int32(1), p.calls.Load(), "a timed-out producer must not be invoked again")
	close(p.release)
	select {
	case <-p.finished:
	case <-time.After(time.Second):
		t.Fatal("late producer did not finish")
	}
	for _, r := range []*scheduling.InferenceRequest{request, first.Request, second.Request} {
		require.Empty(t, r.Body.TokenizedRequest.CacheSalt)
		require.Equal(t, []uint32{1}, r.Body.TokenizedRequest.Prompts[0].TokenIDs)
	}
}

type queuedConditionalProducer struct {
	key   plugin.DataKey
	calls int
}

func (p *queuedConditionalProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "conditional-queued-producer", Name: "conditional-queued-producer"}
}
func (p *queuedConditionalProducer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.key: int64(0)}
}
func (p *queuedConditionalProducer) PrepareForAdmission(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) error {
	return p.Produce(ctx, request, endpoints)
}
func (p *queuedConditionalProducer) Produce(_ context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	p.calls++
	if p.calls == 1 {
		request.PutAttribute(p.key, int64(10))
	}
	return nil
}

func TestQueuedPreparationRefreshClearsProducerAttributes(t *testing.T) {
	p := &queuedConditionalProducer{key: plugin.NewDataKey("queued-refresh-test", "conditional-queued-producer")}
	d := &Director{
		endpointCandidates:    &mocks.MockEndpointCandidates{Candidates: []datalayer.Endpoint{datalayer.NewEndpoint(nil, nil)}},
		requestControlPlugins: Config{dataProducerPlugins: []fwkrc.DataProducer{p}},
	}
	prepare := d.queuedPreparation(&scheduling.InferenceRequest{}, nil)
	first := prepare(t.Context(), func(*contracts.PreparedRequest) bool { return false }, func(*contracts.PreparedRequest) {})
	value, ok := first.Request.GetAttribute(p.key)
	require.True(t, ok)
	require.Equal(t, int64(10), value)
	second := prepare(t.Context(), func(*contracts.PreparedRequest) bool { return false }, func(*contracts.PreparedRequest) {})
	_, ok = second.Request.GetAttribute(p.key)
	require.False(t, ok, "an absent cache hit must not retain a prior producer result")
}
