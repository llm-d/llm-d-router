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
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fccontroller "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	fcregistry "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/bandselection"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/random"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// The observer delegates token estimation and accounting to the production producer.
type observedInFlightProducer struct {
	*inflightload.InFlightLoadProducer
	produced chan map[string]int64
	started  chan string
	blocked  <-chan struct{}
	preCalls atomic.Int64
	stale    atomic.Bool
}

func (p *observedInFlightProducer) Produce(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	return p.InFlightLoadProducer.Produce(ctx, request, endpoints)
}

func (p *observedInFlightProducer) PrepareForAdmission(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	return p.observePreparation(ctx, request, endpoints, false)
}

func (p *observedInFlightProducer) PrepareWithoutPrefix(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	return p.observePreparation(ctx, request, endpoints, true)
}

func (p *observedInFlightProducer) observePreparation(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint, cold bool) error {
	key := attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(p.TypedName().Name)
	for _, endpoint := range endpoints {
		if _, ok := endpoint.Get(key); ok {
			p.stale.Store(true)
		}
	}
	select {
	case p.started <- request.RequestID:
	default:
	}
	if p.blocked != nil {
		<-p.blocked
	}
	prepare := p.InFlightLoadProducer.PrepareForAdmission
	if cold {
		prepare = p.InFlightLoadProducer.PrepareWithoutPrefix
	}
	if err := prepare(ctx, request, endpoints); err != nil {
		return err
	}
	costs := make(map[string]int64, len(endpoints))
	for _, endpoint := range endpoints {
		if value, ok := endpoint.Get(key); ok {
			costs[endpoint.GetMetadata().ID.Name] = value.(*attrconcurrency.UncachedRequestTokens).Tokens
		}
	}
	select {
	case p.produced <- costs:
	default:
	}
	return nil
}

func (p *observedInFlightProducer) ProduceTimeout() time.Duration { return time.Minute }

func (p *observedInFlightProducer) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	p.preCalls.Add(1)
	return p.InFlightLoadProducer.PreRequest(ctx, request, result)
}

type projectedScheduler struct {
	filter   fwksched.Filter
	delegate Scheduler
	calls    atomic.Int64
}

func (s *projectedScheduler) Schedule(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.SchedulingResult, error) {
	s.calls.Add(1)
	if s.delegate != nil {
		return s.delegate.Schedule(ctx, request, endpoints)
	}
	fitting := s.filter.Filter(ctx, request, endpoints)
	if len(fitting) == 0 {
		return nil, errors.New("no endpoint fits profile decode")
	}
	return &fwksched.SchedulingResult{PrimaryProfileName: "decode", ProfileResults: map[string]*fwksched.ProfileRunResult{
		"decode": {TargetEndpoints: fitting[:1]},
	}}, nil
}

type projectedAdmissionHarness struct {
	ctx        context.Context
	director   *Director
	producer   *observedInFlightProducer
	scheduler  *projectedScheduler
	candidates *mockEndpointCandidates
	registry   *fcregistry.FlowRegistry
	tracker    flowcontrol.DispatchReservationTracker
}

func newProjectedAdmissionHarness(t *testing.T, ttl time.Duration, endpoints ...fwkdl.Endpoint) *projectedAdmissionHarness {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	handle := testutils.NewTestHandle(ctx)
	plugin, err := inflightload.InFlightLoadProducerFactory("projected-load", nil, handle)
	require.NoError(t, err)
	producer := &observedInFlightProducer{
		InFlightLoadProducer: plugin.(*inflightload.InFlightLoadProducer),
		produced:             make(chan map[string]int64, 100),
		started:              make(chan string, 100),
	}
	for _, endpoint := range endpoints {
		require.NoError(t, producer.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint}))
	}
	detector, err := concurrency.ConcurrencyDetectorFactory("projected-detector", json.NewDecoder(strings.NewReader(
		`{"concurrencyMode":"tokens","maxTokenConcurrency":100,"inFlightLoadProducerName":"projected-load","failOpen":false}`)), handle)
	require.NoError(t, err)
	ordering, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
	require.NoError(t, err)
	fairness, err := globalstrict.GlobalStrictFairnessPolicyFactory("global-strict", nil, handle)
	require.NoError(t, err)
	defaults := fcregistry.PriorityBandPolicyDefaults{
		OrderingPolicy: ordering.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fairness.(flowcontrol.FairnessPolicy),
	}
	band, err := fcregistry.NewPriorityBandConfig(0, defaults, fcregistry.WithBandMaxBytes(10_000_000))
	require.NoError(t, err)
	config, err := fcregistry.NewConfig(defaults, fcregistry.WithPriorityBand(band))
	require.NoError(t, err)
	registry := fcregistry.NewFlowRegistry(config, logr.Discard())
	go registry.RunMaintenanceLoop(ctx)
	candidates := &mockEndpointCandidates{result: endpoints}
	fc := fccontroller.NewFlowController(ctx, "test-pool", &fccontroller.Config{
		DefaultRequestTTL: ttl, NoEndpointRequestTTL: ttl,
		ExpiryCleanupInterval: 10 * time.Millisecond, EnqueueChannelBufferSize: 100,
	}, fccontroller.Deps{
		Registry: registry, SaturationDetector: detector.(flowcontrol.SaturationDetector),
		EndpointCandidates: candidates, UsageLimitPolicy: usagelimits.DefaultPolicy(),
		BandSelectionPolicy: bandselection.DefaultPolicy(),
	})
	scheduler := &projectedScheduler{filter: detector.(fwksched.Filter)}
	director := NewDirectorWithConfig(&mockDatastore{}, scheduler,
		NewFlowControlAdmissionController(fc, "test-pool", candidates), candidates,
		NewConfig().WithDataProducerPlugins(producer).WithPreRequestPlugins(producer))
	return &projectedAdmissionHarness{ctx: ctx, director: director, producer: producer, scheduler: scheduler, candidates: candidates, registry: registry,
		tracker: detector.(flowcontrol.DispatchReservationTracker)}
}

func projectedEndpoint(name, role string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Namespace: "default", Name: name}, Address: "127.0.0.1", Port: "8000",
		Labels: map[string]string{bylabel.RoleLabel: role},
	}, nil)
}

func projectedBody(tokens int) *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{
		Model: "test-model", TokenizedRequest: fwkrh.NewTokenizedRequest([][]uint32{make([]uint32, tokens)}),
	}
}

func (h *projectedAdmissionHarness) seed(t *testing.T, endpoint fwkdl.Endpoint, tokens int) *fwksched.InferenceRequest {
	t.Helper()
	profile := endpoint.GetMetadata().Labels[bylabel.RoleLabel]
	request := &fwksched.InferenceRequest{RequestID: "incumbent-" + endpoint.GetMetadata().ID.Name, Body: projectedBody(tokens)}
	request.SchedulingResult = &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{
		profile: {TargetEndpoints: []fwksched.Endpoint{fwksched.NewEndpoint(endpoint.GetMetadata(), endpoint.GetMetrics(), endpoint.GetAttributes())}},
	}}
	require.NoError(t, h.producer.InFlightLoadProducer.PreRequest(h.ctx, request, request.SchedulingResult))
	return request
}

func (h *projectedAdmissionHarness) start(ctx context.Context, id string, tokens int) (*handlers.RequestContext, <-chan error) {
	request := &handlers.RequestContext{
		Request:         &handlers.Request{Headers: map[string]string{reqcommon.RequestIDHeaderKey: id}, Metadata: map[string]any{}},
		TargetModelName: "test-model", RequestSize: 100, RequestReceivedTimestamp: time.Now(),
	}
	result := make(chan error, 1)
	go func() {
		_, err := h.director.HandleRequest(ctx, request, projectedBody(tokens))
		result <- err
	}()
	return request, result
}

func (h *projectedAdmissionHarness) finish(request *fwksched.InferenceRequest) {
	h.producer.ResponseBody(h.ctx, request, &fwkrc.Response{EndOfStream: true}, nil)
}

func (h *projectedAdmissionHarness) requireQueued(t *testing.T, result <-chan error) {
	t.Helper()
	deadline := time.NewTimer(time.Second)
	defer deadline.Stop()
	poll := time.NewTicker(time.Millisecond)
	defer poll.Stop()
	for h.registry.Stats().Global.Len != 1 {
		select {
		case err := <-result:
			t.Fatalf("request left the queue before capacity was available: %v", err)
		case <-poll.C:
		case <-deadline.C:
			t.Fatal("request did not enter the flow registry")
		}
	}
	select {
	case err := <-result:
		t.Fatalf("request left the queue before capacity was available: %v", err)
	case <-time.After(100 * time.Millisecond):
	}
	require.Zero(t, h.scheduler.calls.Load())
	require.Zero(t, h.producer.preCalls.Load())
}

func awaitProjectedCosts(t *testing.T, h *projectedAdmissionHarness) map[string]int64 {
	t.Helper()
	select {
	case costs := <-h.producer.produced:
		return costs
	case <-time.After(time.Second):
		t.Fatal("in-flight producer did not prepare the queued request")
		return nil
	}
}

func TestDirectorProjectedAdmissionWaitsForTokenCapacity(t *testing.T) {
	a, b := projectedEndpoint("a", ""), projectedEndpoint("b", "")
	h := newProjectedAdmissionHarness(t, time.Minute, a, b)
	incumbent := h.seed(t, a, 60)
	h.seed(t, b, 100)
	request, result := h.start(h.ctx, "incoming", 50)
	require.Equal(t, map[string]int64{"a": 50, "b": 50}, awaitProjectedCosts(t, h))
	h.requireQueued(t, result)
	require.False(t, h.producer.stale.Load(), "costs must be computed for this request")
	require.Equal(t, int64(60), h.producer.GetTokens(a.GetMetadata().ID.String()))
	require.Equal(t, int64(1), h.producer.GetRequests(a.GetMetadata().ID.String()))
	h.finish(incumbent)
	require.NoError(t, waitAdmit(t, result))
	require.True(t, request.FlowControlAdmitted)
	require.Equal(t, int64(1), h.scheduler.calls.Load())
	require.Equal(t, int64(1), h.producer.preCalls.Load())
	require.Equal(t, int64(50), h.producer.GetTokens(a.GetMetadata().ID.String()))
	require.Equal(t, int64(1), h.producer.GetRequests(a.GetMetadata().ID.String()))
	h.finish(request.SchedulingRequest)
	require.Zero(t, h.producer.GetTokens(a.GetMetadata().ID.String()))
	require.Zero(t, h.producer.GetRequests(a.GetMetadata().ID.String()))
}

func TestDirectorProjectedAdmissionReleasesDispatchReservation(t *testing.T) {
	for _, schedulerFails := range []bool{false, true} {
		t.Run(fmt.Sprintf("scheduler_fails=%v", schedulerFails), func(t *testing.T) {
			endpoint := projectedEndpoint("a", "")
			h := newProjectedAdmissionHarness(t, time.Minute, endpoint)
			var observed, missingReservation atomic.Bool
			h.director.requestControlPlugins.WithPreRequestPlugins(h.producer, &mockPreRequestPlugin{
				name: "observe-projected-reservation",
				modifyFn: func(request *fwksched.InferenceRequest) {
					observed.Store(true)
					missingReservation.Store(h.tracker.ReserveDispatch(request.RequestID))
				},
			})
			if schedulerFails {
				h.scheduler.delegate = &mockScheduler{scheduleErr: errors.New("scheduling failed")}
			}
			request, result := h.start(h.ctx, "prepared-reservation", 50)
			require.Equal(t, map[string]int64{"a": 50}, awaitProjectedCosts(t, h))
			err := waitAdmit(t, result)
			if schedulerFails {
				require.ErrorContains(t, err, "scheduling failed")
				require.False(t, observed.Load())
				require.Zero(t, h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
			} else {
				require.NoError(t, err)
				require.True(t, observed.Load())
				require.False(t, missingReservation.Load(), "reservation must cover every PreRequest hook")
				require.Equal(t, int64(50), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
				h.finish(request.SchedulingRequest)
			}
			require.False(t, h.tracker.ReleaseDispatch("prepared-reservation"), "Director must release the reservation")
		})
	}
}

func TestDirectorProjectedAdmissionPreservesOptionalPrefill(t *testing.T) {
	for _, allBlocked := range []bool{false, true} {
		t.Run(fmt.Sprintf("all_partitions_blocked=%v", allBlocked), func(t *testing.T) {
			p, d := projectedEndpoint("p", bylabel.RolePrefill), projectedEndpoint("d", bylabel.RoleDecode)
			h := newProjectedAdmissionHarness(t, time.Minute, p, d)
			handler := disagg.NewDisaggProfileHandler("decode", "prefill", "", nil, nil)
			h.scheduler.delegate = scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(handler,
				map[string]fwksched.SchedulerProfile{
					"decode": scheduling.NewSchedulerProfile().WithFilters(bylabel.NewDecodeRole(), h.scheduler.filter).
						WithPicker(random.NewRandomPicker(1)),
					"prefill": scheduling.NewSchedulerProfile().WithFilters(bylabel.NewPrefillRole(), h.scheduler.filter).
						WithPicker(random.NewRandomPicker(1)),
				}))
			h.director.requestControlPlugins.WithPreRequestPlugins(h.producer, handler)
			h.seed(t, p, 60)
			var decodeIncumbent *fwksched.InferenceRequest
			if allBlocked {
				decodeIncumbent = h.seed(t, d, 60)
			}
			request, result := h.start(h.ctx, "pd-incoming", 50)
			require.Equal(t, map[string]int64{"p": 50, "d": 50}, awaitProjectedCosts(t, h))
			if allBlocked {
				h.requireQueued(t, result)
				h.finish(decodeIncumbent)
			}
			require.NoError(t, waitAdmit(t, result))
			require.Len(t, request.SchedulingRequest.SchedulingResult.ProfileResults, 1)
			require.Contains(t, request.SchedulingRequest.SchedulingResult.ProfileResults, "decode")
			require.Equal(t, "d", request.TargetPod.ID.Name)
			require.Equal(t, int64(1), h.producer.preCalls.Load())
			require.Equal(t, int64(60), h.producer.GetTokens(p.GetMetadata().ID.String()))
			require.Equal(t, int64(50), h.producer.GetTokens(d.GetMetadata().ID.String()))
			h.finish(request.SchedulingRequest)
		})
	}
}

func TestDirectorProjectedAdmissionTerminatesWhileProducerBlocked(t *testing.T) {
	for _, cancelRequest := range []bool{false, true} {
		t.Run(fmt.Sprintf("cancel=%v", cancelRequest), func(t *testing.T) {
			ttl := 150 * time.Millisecond
			if cancelRequest {
				ttl = time.Minute
			}
			h := newProjectedAdmissionHarness(t, ttl, projectedEndpoint("a", ""))
			unblock := make(chan struct{})
			h.producer.blocked = unblock
			t.Cleanup(func() {
				select {
				case <-unblock:
				default:
					close(unblock)
				}
			})
			ctx, cancel := context.WithCancel(h.ctx)
			defer cancel()
			_, result := h.start(ctx, "blocked-producer", 50)
			select {
			case <-h.producer.started:
			case <-time.After(time.Second):
				t.Fatal("producer did not start")
			}
			if cancelRequest {
				cancel()
			}
			err := waitAdmit(t, result)
			if cancelRequest {
				requireDropped(t, err, errcommon.ServiceUnavailable, errcommon.RequestDroppedReasonContextCancelled)
			} else {
				requireDropped(t, err, errcommon.ResourceExhausted, errcommon.RequestDroppedReasonTTLExpired)
			}
			close(unblock)
			awaitProjectedCosts(t, h)
			require.Never(t, func() bool { return h.scheduler.calls.Load() != 0 }, 100*time.Millisecond, time.Millisecond,
				"a producer completing after eviction must not dispatch the request")
			require.Zero(t, h.scheduler.calls.Load())
			require.Zero(t, h.producer.preCalls.Load())
			require.Eventually(t, func() bool { return h.registry.Stats().Global.Len == 0 }, time.Second, time.Millisecond)
		})
	}
}

func TestDirectorProjectedAdmissionPreparesQueuedRequestsConcurrently(t *testing.T) {
	endpoint := projectedEndpoint("idle", "")
	h := newProjectedAdmissionHarness(t, time.Minute, endpoint)
	unblock := make(chan struct{})
	h.producer.blocked = unblock
	t.Cleanup(func() {
		select {
		case <-unblock:
		default:
			close(unblock)
		}
	})
	const requestCount = 8
	requests := make([]*handlers.RequestContext, requestCount)
	results := make([]<-chan error, requestCount)
	for i := range requestCount {
		requests[i], results[i] = h.start(h.ctx, fmt.Sprintf("concurrent-%d", i), 10)
	}
	started := make(map[string]struct{}, requestCount)
	deadline := time.After(time.Second)
	for len(started) < requestCount {
		select {
		case id := <-h.producer.started:
			started[id] = struct{}{}
		case <-deadline:
			t.Fatalf("only %d of %d queued requests entered the producer before its barrier was released", len(started), requestCount)
		}
	}
	require.Equal(t, uint64(requestCount), h.registry.Stats().Global.Len)
	require.Zero(t, h.scheduler.calls.Load())
	require.Zero(t, h.producer.preCalls.Load())
	close(unblock)
	for _, result := range results {
		require.NoError(t, waitAdmit(t, result))
	}
	require.Equal(t, int64(requestCount), h.scheduler.calls.Load())
	require.Equal(t, int64(requestCount), h.producer.preCalls.Load())
	require.Equal(t, int64(10*requestCount), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	require.Equal(t, int64(requestCount), h.producer.GetRequests(endpoint.GetMetadata().ID.String()))
	for _, request := range requests {
		h.finish(request.SchedulingRequest)
	}
	require.Zero(t, h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	require.Zero(t, h.producer.GetRequests(endpoint.GetMetadata().ID.String()))
}

type projectedPrefixProducer struct{}

func (p *projectedPrefixProducer) PrepareForAdmission(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	return p.Produce(ctx, request, endpoints)
}

func (p *projectedPrefixProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "test-prefix", Name: "test-prefix"}
}

func (p *projectedPrefixProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{attrprefix.PrefixCacheMatchInfoDataKey: attrprefix.PrefixCacheMatchInfo{}}
}

func (p *projectedPrefixProducer) Produce(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	for _, endpoint := range endpoints {
		matched := 0
		if endpoint.GetMetadata().ID.Name == "cached" {
			matched = 20
		}
		endpoint.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(matched, request.Body.TokenizedRequest.TokenCount(), 1))
	}
	return nil
}

func TestDirectorProjectedAdmissionUsesEndpointPrefixCosts(t *testing.T) {
	a, b := projectedEndpoint("uncached", ""), projectedEndpoint("cached", "")
	h := newProjectedAdmissionHarness(t, time.Minute, a, b)
	h.seed(t, a, 60)
	h.seed(t, b, 60)
	h.director.requestControlPlugins.WithDataProducerPlugins(&projectedPrefixProducer{}, h.producer)
	request, result := h.start(h.ctx, "prefix-incoming", 50)
	require.Equal(t, map[string]int64{"uncached": 50, "cached": 50}, awaitProjectedCosts(t, h))
	require.Equal(t, map[string]int64{"uncached": 50, "cached": 30}, awaitProjectedCosts(t, h))
	require.NoError(t, waitAdmit(t, result))
	require.Equal(t, "cached", request.TargetPod.ID.Name)
	require.Equal(t, int64(90), h.producer.GetTokens(b.GetMetadata().ID.String()))
	require.Equal(t, int64(60), h.producer.GetTokens(a.GetMetadata().ID.String()))
	h.finish(request.SchedulingRequest)
}

func TestDirectorProjectedAdmissionRefreshesCandidates(t *testing.T) {
	for _, fromZero := range []bool{false, true} {
		t.Run(fmt.Sprintf("from_zero=%v", fromZero), func(t *testing.T) {
			var endpoints []fwkdl.Endpoint
			if !fromZero {
				endpoints = []fwkdl.Endpoint{projectedEndpoint("busy", "")}
			}
			h := newProjectedAdmissionHarness(t, time.Minute, endpoints...)
			if !fromZero {
				h.seed(t, endpoints[0], 60)
			}
			request, result := h.start(h.ctx, "scale-incoming", 50)
			h.requireQueued(t, result)
			added := projectedEndpoint("added", "")
			require.NoError(t, h.producer.Extract(h.ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: added}))
			h.candidates.mu.Lock()
			h.candidates.result = append(append([]fwkdl.Endpoint(nil), endpoints...), added)
			h.candidates.mu.Unlock()
			require.NoError(t, waitAdmit(t, result))
			require.Equal(t, "added", request.TargetPod.ID.Name)
			require.Equal(t, int64(50), h.producer.GetTokens(added.GetMetadata().ID.String()))
			require.Equal(t, int64(1), h.producer.preCalls.Load())
			h.finish(request.SchedulingRequest)
		})
	}
}

func TestDirectorProjectedAdmissionAllowsOversizedRequestOnIdleEndpoint(t *testing.T) {
	endpoint := projectedEndpoint("idle", "")
	h := newProjectedAdmissionHarness(t, time.Minute, endpoint)
	request, result := h.start(h.ctx, "oversized", 150)
	require.Equal(t, map[string]int64{"idle": 150}, awaitProjectedCosts(t, h))
	require.NoError(t, waitAdmit(t, result))
	require.Equal(t, int64(150), h.producer.GetTokens(endpoint.GetMetadata().ID.String()))
	require.Equal(t, int64(1), h.producer.preCalls.Load())
	h.finish(request.SchedulingRequest)
}
