/*
Copyright 2026 The Kubernetes Authors.

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

package flowcontrol_test

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	contractmocks "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	fcTypes "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// ============================================================================
// Test Helpers
// ============================================================================

// --- Test Request ---

type testRequest struct {
	id        string
	key       flowcontrol.FlowKey
	byteSize  uint64
	ttl       time.Duration
	infReq    *fwksched.InferenceRequest
	timestamp time.Time
}

func (r *testRequest) FlowKey() flowcontrol.FlowKey       { return r.key }
func (r *testRequest) ByteSize() uint64                   { return r.byteSize }
func (r *testRequest) InitialEffectiveTTL() time.Duration { return r.ttl }
func (r *testRequest) ID() string                         { return r.id }
func (r *testRequest) GetMetadata() map[string]any        { return nil }
func (r *testRequest) InferencePoolName() string          { return "test-pool" }
func (r *testRequest) ModelName() string                  { return "test-model" }
func (r *testRequest) TargetModelName() string            { return "test-target" }

func (r *testRequest) InferenceRequest() *fwksched.InferenceRequest { return r.infReq }

func (r *testRequest) ReceivedTimestamp() time.Time {
	if !r.timestamp.IsZero() {
		return r.timestamp
	}
	return time.Now()
}

// --- Switchable Detector ---

type switchableDetector struct {
	flowcontrol.SaturationDetector
	blocked  atomic.Bool
	limit    atomic.Int64
	inFlight atomic.Int64
}

func newBlockedDetector() *switchableDetector {
	d := &switchableDetector{}
	d.blocked.Store(true)
	return d
}

func newGatedDetector(limit int64) *switchableDetector {
	d := &switchableDetector{}
	d.limit.Store(limit)
	return d
}

func (d *switchableDetector) Saturation(_ context.Context, _ []datalayer.Endpoint) float64 {
	if d.blocked.Load() {
		return 1.0
	}
	limit := d.limit.Load()
	if limit <= 0 {
		return 0.0
	}
	if d.inFlight.Add(1) <= limit {
		return 0.99
	}
	d.inFlight.Add(-1)
	return 1.0
}

// SaturationWithInFlight delegates to Saturation; this mock reserves slots via
// its own atomic counter, so the controller-supplied credit is not used here.
func (d *switchableDetector) SaturationWithInFlight(
	ctx context.Context, endpoints []datalayer.Endpoint, _ int,
) float64 {
	return d.Saturation(ctx, endpoints)
}

func (d *switchableDetector) Unblock(limit int64) {
	d.limit.Store(limit)
	d.blocked.Store(false)
}

func (d *switchableDetector) Release() {
	d.inFlight.Add(-1)
}

// nonEmptyCandidates returns a pool containing a single placeholder endpoint.
//
// Capacity tests block dispatch via a saturated detector (which ignores the pool), but the processor
// classifies a queue-capacity rejection as RejectedNoEndpoints when the pool is empty and as
// RejectedCapacity when it is not. These tests assert RejectedCapacity, so they need a non-empty pool
// even though the endpoint is never dispatched to.
func nonEmptyCandidates() *contractmocks.MockEndpointCandidates {
	return &contractmocks.MockEndpointCandidates{
		Candidates: []datalayer.Endpoint{datalayer.NewEndpoint(nil, nil)},
	}
}

// --- dispatchResult ---

type dispatchResult struct {
	id      string
	flowID  string
	outcome fcTypes.QueueOutcome
	err     error
}

// --- Producer and Detector ---

type producerAndDetector struct {
	producer *inflightload.InFlightLoadProducer
	detector flowcontrol.SaturationDetector
	epMeta   *datalayer.EndpointMetadata
	ep       datalayer.Endpoint
}

func newProducerAndDetector(ctx context.Context, t *testing.T, maxConcurrency int) *producerAndDetector {
	t.Helper()
	handle := testutils.NewTestHandle(ctx)

	producerName := t.Name() + "-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxConcurrency": %d, "inFlightLoadProducerName": %q}`, maxConcurrency, producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		t.Name()+"-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	det := detectorPlugin.(flowcontrol.SaturationDetector)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	return &producerAndDetector{
		producer: producer,
		detector: det,
		epMeta:   epMeta,
		ep:       ep,
	}
}

// --- Test Harness ---

type integrationHarness struct {
	t      *testing.T
	ctx    context.Context
	cancel context.CancelFunc
	fc     *controller.FlowController
	reg    *registry.FlowRegistry
}

type harnessOpts struct {
	ordering           flowcontrol.OrderingPolicy
	fairness           flowcontrol.FairnessPolicy
	detector           flowcontrol.SaturationDetector
	bands              []*registry.PriorityBandConfig
	maxRequests        uint64
	maxBytes           uint64
	bandMaxBytes       uint64
	bandMaxRequests    uint64
	controllerCfg      *controller.Config
	endpointCandidates contracts.EndpointCandidates
	usageLimitPolicy   flowcontrol.UsageLimitPolicy
}

func newHarness(t *testing.T, opts harnessOpts) *integrationHarness {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())

	handle := testutils.NewTestHandle(ctx)

	if opts.ordering == nil {
		p, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
		require.NoError(t, err)
		opts.ordering = p.(flowcontrol.OrderingPolicy)
	}
	if opts.fairness == nil {
		p, err := globalstrict.GlobalStrictFairnessPolicyFactory("global-strict", nil, handle)
		require.NoError(t, err)
		opts.fairness = p.(flowcontrol.FairnessPolicy)
	}
	if opts.detector == nil {
		opts.detector = newGatedDetector(0)
	}

	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: opts.ordering,
		FairnessPolicy: opts.fairness,
	}

	var cfgOpts []registry.ConfigOption
	if opts.maxRequests > 0 {
		cfgOpts = append(cfgOpts, registry.WithMaxRequests(opts.maxRequests))
	}
	if opts.maxBytes > 0 {
		cfgOpts = append(cfgOpts, registry.WithMaxBytes(opts.maxBytes))
	}

	if len(opts.bands) > 0 {
		for _, b := range opts.bands {
			cfgOpts = append(cfgOpts, registry.WithPriorityBand(b))
		}
	} else {
		var bandOpts []registry.PriorityBandConfigOption
		if opts.bandMaxBytes > 0 {
			bandOpts = append(bandOpts, registry.WithBandMaxBytes(opts.bandMaxBytes))
		} else {
			bandOpts = append(bandOpts, registry.WithBandMaxBytes(10_000_000_000))
		}
		if opts.bandMaxRequests > 0 {
			bandOpts = append(bandOpts, registry.WithBandMaxRequests(opts.bandMaxRequests))
		}
		band, err := registry.NewPriorityBandConfig(0, defaults, bandOpts...)
		require.NoError(t, err)
		cfgOpts = append(cfgOpts, registry.WithPriorityBand(band))
	}

	regCfg, err := registry.NewConfig(defaults, cfgOpts...)
	require.NoError(t, err)

	reg := registry.NewFlowRegistry(regCfg, logr.Discard())
	go reg.RunMaintenanceLoop(ctx)

	controllerCfg := opts.controllerCfg
	if controllerCfg == nil {
		controllerCfg = &controller.Config{
			DefaultRequestTTL:        5 * time.Minute,
			ExpiryCleanupInterval:    10 * time.Millisecond,
			EnqueueChannelBufferSize: 100,
		}
	}

	endpointCandidates := opts.endpointCandidates
	if endpointCandidates == nil {
		endpointCandidates = &contractmocks.MockEndpointCandidates{}
	}

	usageLimitPolicy := opts.usageLimitPolicy
	if usageLimitPolicy == nil {
		usageLimitPolicy = usagelimits.DefaultPolicy()
	}

	fc := controller.NewFlowController(ctx, "integration-test", controllerCfg, controller.Deps{
		Registry:           reg,
		SaturationDetector: opts.detector,
		EndpointCandidates: endpointCandidates,
		UsageLimitPolicy:   usageLimitPolicy,
	})

	t.Cleanup(func() {
		cancel()
		time.Sleep(50 * time.Millisecond)
	})

	time.Sleep(10 * time.Millisecond)

	return &integrationHarness{t: t, ctx: ctx, cancel: cancel, fc: fc, reg: reg}
}
