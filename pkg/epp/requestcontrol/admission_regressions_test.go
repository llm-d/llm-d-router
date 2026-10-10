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
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/random"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

type nonrepeatableAdmissionProducer struct {
	seen     sync.Map
	calls    atomic.Int64
	repeated chan string
	unblock  <-chan struct{}
}

func (*nonrepeatableAdmissionProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "nonrepeatable-admission-producer", Name: "nonrepeatable-admission-producer"}
}

func (*nonrepeatableAdmissionProducer) Produces() map[fwkplugin.DataKey]any {
	return nil
}

func (p *nonrepeatableAdmissionProducer) Produce(ctx context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	p.calls.Add(1)
	if _, repeated := p.seen.LoadOrStore(request.RequestID, true); repeated {
		select {
		case p.repeated <- request.RequestID:
		default:
		}
		select {
		case <-p.unblock:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func TestDirectorAdmissionDoesNotRepeatProducersAfterQueueWait(t *testing.T) {
	endpoint := projectedEndpoint("healthy-after-release", "")
	h := newProjectedAdmissionHarness(t, time.Minute, endpoint)
	incumbent := h.seed(t, endpoint, 100)
	unblock := make(chan struct{})
	t.Cleanup(func() { close(unblock) })
	producer := &nonrepeatableAdmissionProducer{repeated: make(chan string, 8), unblock: unblock}
	h.director.requestControlPlugins.WithDataProducerPlugins(producer, h.producer)
	requests := make([]*handlers.RequestContext, 8)
	results := make([]<-chan error, len(requests))
	for i := range requests {
		requests[i], results[i] = h.start(h.ctx, fmt.Sprintf("aged-%d", i), 10)
	}
	for range requests {
		require.Equal(t, map[string]int64{"healthy-after-release": 10}, awaitProjectedCosts(t, h))
	}
	require.Equal(t, uint64(len(requests)), h.registry.Stats().Global.Len)
	require.Zero(t, h.scheduler.calls.Load())
	require.Zero(t, h.producer.preCalls.Load())

	// The incumbent keeps the queue saturated beyond the candidate refresh period.
	select {
	case id := <-producer.repeated:
		t.Fatalf("producer repeated for queued request %q", id)
	case <-time.After(100 * time.Millisecond):
	}
	h.finish(incumbent)
	for _, result := range results {
		select {
		case id := <-producer.repeated:
			t.Fatalf("producer repeated for request %q after capacity became available", id)
		case err := <-result:
			require.NoError(t, err)
		case <-time.After(time.Second):
			t.Fatal("prepared request did not dispatch after capacity became available")
		}
	}
	require.Equal(t, int64(len(requests)), producer.calls.Load())
	require.Equal(t, int64(len(requests)), h.scheduler.calls.Load())
	require.Equal(t, int64(len(requests)), h.producer.preCalls.Load())
	require.Equal(t, int64(10*len(requests)), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	for _, request := range requests {
		h.finish(request.SchedulingRequest)
	}
	require.Zero(t, h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	require.Zero(t, h.producer.GetRequests(endpoint.GetMetadata().ID.String()))
}

func TestDirectorAdmissionPreservesEmptyCandidateStatus(t *testing.T) {
	for _, subset := range []bool{true, false} {
		name := "screener"
		if subset {
			name = "metadata_subset"
		}
		t.Run(name, func(t *testing.T) {
			endpoint := projectedEndpoint("healthy", "")
			h := newProjectedAdmissionHarness(t, 150*time.Millisecond, endpoint)
			h.director.endpointCandidates = NewDatastoreEndpointCandidates(&mockDatastore{pods: []fwkdl.Endpoint{endpoint}})
			requestMetadata := map[string]any{}
			if subset {
				requestMetadata[metadata.SubsetFilterNamespace] = map[string]any{metadata.SubsetFilterKey: []any{}}
			} else {
				h.director.requestControlPlugins.WithScreeners(&mockScreener{
					name: "empty-candidates",
					screen: func([]fwksched.Endpoint) []fwksched.Endpoint {
						return nil
					},
				})
			}
			request := &handlers.RequestContext{
				Request: &handlers.Request{
					Headers:  map[string]string{reqcommon.RequestIDHeaderKey: "empty-" + name},
					Metadata: requestMetadata,
				},
				TargetModelName: "test-model", RequestSize: 100, RequestReceivedTimestamp: time.Now(),
			}
			_, err := h.director.HandleRequest(h.ctx, request, projectedBody(50))
			requireDropped(t, err, errcommon.ServiceUnavailable, errcommon.RequestDroppedReasonNoEndpoints)
			require.Zero(t, h.scheduler.calls.Load())
			require.Zero(t, h.producer.preCalls.Load())
			require.Zero(t, h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
			require.Eventually(t, func() bool { return h.registry.Stats().Global.Len == 0 }, time.Second, time.Millisecond)
		})
	}
}

func TestDirectorAdmissionWaitsForRequiredPrefill(t *testing.T) {
	p, d := projectedEndpoint("prefill", bylabel.RolePrefill), projectedEndpoint("decode", bylabel.RoleDecode)
	h := newProjectedAdmissionHarness(t, time.Minute, p, d)
	decider, err := disagg.AlwaysDisaggPDDeciderPluginFactory("required-prefill", nil, testutils.NewTestHandle(h.ctx))
	require.NoError(t, err)
	handler := disagg.NewDisaggProfileHandler("decode", "prefill", "", decider.(*disagg.AlwaysDisaggPDDecider), nil)
	h.director.scheduler = scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(handler,
		map[string]fwksched.SchedulerProfile{
			"decode": scheduling.NewSchedulerProfile().WithFilters(bylabel.NewDecodeRole(), h.scheduler.filter).
				WithPicker(random.NewRandomPicker(1)),
			"prefill": scheduling.NewSchedulerProfile().WithFilters(bylabel.NewPrefillRole(), h.scheduler.filter).
				WithPicker(random.NewRandomPicker(1)),
		}))
	h.director.requestControlPlugins.WithPreRequestPlugins(h.producer, handler)
	incumbent := h.seed(t, p, 60)
	request, result := h.start(h.ctx, "required-prefill-incoming", 50)
	require.Equal(t, map[string]int64{"prefill": 50, "decode": 50}, awaitProjectedCosts(t, h))
	h.requireQueued(t, result)
	require.Equal(t, int64(60), h.producer.GetTokens(p.GetMetadata().ID.String()))
	require.Zero(t, h.producer.GetTokens(d.GetMetadata().ID.String()))
	h.finish(incumbent)
	require.NoError(t, waitAdmit(t, result))
	require.True(t, request.FlowControlAdmitted)
	require.Len(t, request.SchedulingRequest.SchedulingResult.ProfileResults, 2)
	require.Contains(t, request.SchedulingRequest.SchedulingResult.ProfileResults, "prefill")
	require.Contains(t, request.SchedulingRequest.SchedulingResult.ProfileResults, "decode")
	require.Equal(t, "decode", request.TargetPod.ID.Name)
	require.Equal(t, int64(1), h.producer.preCalls.Load())
	for _, endpoint := range []fwkdl.Endpoint{p, d} {
		require.Equal(t, int64(50), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
		require.Equal(t, int64(1), h.producer.GetRequests(endpoint.GetMetadata().ID.String()))
	}
	h.finish(request.SchedulingRequest)
	for _, endpoint := range []fwkdl.Endpoint{p, d} {
		require.Zero(t, h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
		require.Zero(t, h.producer.GetRequests(endpoint.GetMetadata().ID.String()))
	}
}
