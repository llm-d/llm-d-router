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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	contractmocks "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/eviction"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	fcTypes "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/eviction/filtering"
	evictionordering "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/eviction/ordering"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/roundrobin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/slodeadline"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
	igwtestutils "github.com/llm-d/llm-d-router/test/utils/igw"
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

func (d *switchableDetector) Unblock(limit int64) {
	d.limit.Store(limit)
	d.blocked.Store(false)
}

func (d *switchableDetector) Release() {
	d.inFlight.Add(-1)
}

// --- dispatchResult ---

type dispatchResult struct {
	id      string
	flowID  string
	outcome fcTypes.QueueOutcome
	err     error
}

// --- Test Harness ---

type integrationHarness struct {
	t      *testing.T
	ctx    context.Context
	cancel context.CancelFunc
	fc     *controller.FlowController
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

	handle := igwtestutils.NewTestHandle(ctx)

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
	go reg.Run(ctx)

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

	return &integrationHarness{t: t, ctx: ctx, cancel: cancel, fc: fc}
}

// ============================================================================
// Saturation Data Path Tests (producer -> detector contracts)
// ============================================================================

// TestSaturationDataPath verifies the producer->DynamicAttribute->detector contract
// that was broken in issue #1474.
func TestSaturationDataPath(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	producerName := "test-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxConcurrency": 10, "inFlightLoadProducerName": %q}`, producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"test-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	detector := detectorPlugin.(flowcontrol.SaturationDetector)

	ep1Meta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep2Meta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-2", Namespace: "default"},
	}
	ep1 := datalayer.NewEndpoint(ep1Meta, datalayer.NewMetrics())
	ep2 := datalayer.NewEndpoint(ep2Meta, datalayer.NewMetrics())

	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep1,
	}))
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep2,
	}))

	endpoints := []datalayer.Endpoint{ep1, ep2}

	t.Run("zero with no in-flight requests", func(t *testing.T) {
		require.InDelta(t, 0.0, detector.Saturation(ctx, endpoints), 1e-9)
	})

	schedEndpoint := fwksched.NewEndpoint(ep1Meta, datalayer.NewMetrics(), nil)
	request := &fwksched.InferenceRequest{
		RequestID: "req-1",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 100)},
		},
	}
	result := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{schedEndpoint}},
		},
	}

	producer.PreRequest(ctx, request, result)

	t.Run("nonzero after PreRequest", func(t *testing.T) {
		sat := detector.Saturation(ctx, endpoints)
		require.Greater(t, sat, 0.0,
			"Saturation must be nonzero after PreRequest tracks a request (guards against #1474)")
		// 1 request on pod-1 (capacity 10) + 0 on pod-2 (capacity 10) = 1/20 = 0.05
		require.InDelta(t, 0.05, sat, 1e-6)
	})

	request.SchedulingResult = result
	producer.ResponseBody(ctx, request,
		&requestcontrol.Response{EndOfStream: true}, ep1Meta)

	t.Run("returns to zero after response completes", func(t *testing.T) {
		require.InDelta(t, 0.0, detector.Saturation(ctx, endpoints), 1e-9)
	})
}

// TestSaturationTokenMode verifies the concurrency detector in token mode.
func TestSaturationTokenMode(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	producerName := "token-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxTokenConcurrency": 1000, "concurrencyMode": "tokens", "inFlightLoadProducerName": %q}`,
		producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"token-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	detector := detectorPlugin.(flowcontrol.SaturationDetector)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	endpoints := []datalayer.Endpoint{ep}
	require.InDelta(t, 0.0, detector.Saturation(ctx, endpoints), 1e-9)

	schedEndpoint := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	request := &fwksched.InferenceRequest{
		RequestID: "req-token-1",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 200)},
		},
	}
	result := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{schedEndpoint}},
		},
	}

	producer.PreRequest(ctx, request, result)

	sat := detector.Saturation(ctx, endpoints)
	require.Greater(t, sat, 0.0, "token-mode saturation must be nonzero after PreRequest")
	// 200 tokens on pod-1 / 1000 max = 0.2
	require.InDelta(t, 0.2, sat, 1e-6)

	request.SchedulingResult = result
	producer.ResponseBody(ctx, request,
		&requestcontrol.Response{EndOfStream: true}, epMeta)

	require.InDelta(t, 0.0, detector.Saturation(ctx, endpoints), 1e-9,
		"token-mode saturation must return to zero after response")
}

// TestSaturationStartOfStreamTokenRelease verifies that tokens are released
// early at StartOfStream while the request counter stays held until EndOfStream.
func TestSaturationStartOfStreamTokenRelease(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	producerName := "sos-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	schedEndpoint := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	request := &fwksched.InferenceRequest{
		RequestID: "req-sos-1",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 100)},
		},
	}
	result := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{schedEndpoint}},
		},
	}
	producer.PreRequest(ctx, request, result)

	eid := epMeta.NamespacedName.String()
	tokensBeforeSOS := producer.GetTokens(eid)
	requestsBeforeSOS := producer.GetRequests(eid)
	require.Greater(t, tokensBeforeSOS, int64(0), "tokens should be tracked after PreRequest")
	require.Equal(t, int64(1), requestsBeforeSOS, "1 request should be tracked")

	request.SchedulingResult = result
	producer.ResponseBody(ctx, request,
		&requestcontrol.Response{StartOfStream: true}, epMeta)

	tokensAfterSOS := producer.GetTokens(eid)
	requestsAfterSOS := producer.GetRequests(eid)
	require.Equal(t, int64(0), tokensAfterSOS,
		"tokens should be released at StartOfStream (prefill complete)")
	require.Equal(t, int64(1), requestsAfterSOS,
		"request counter should remain held until EndOfStream")

	producer.ResponseBody(ctx, request,
		&requestcontrol.Response{EndOfStream: true}, epMeta)

	require.Equal(t, int64(0), producer.GetRequests(eid),
		"request counter should be released at EndOfStream")
}

// TestMultiProfilePrefillDecodeTracking verifies the producer correctly tracks
// requests across prefill and decode profiles (P/D disaggregation). PreRequest
// increments counters on both endpoints. StartOfStream releases tokens on all
// profiles when addEstimatedOutputTokens=false, or only the prefill profile
// when addEstimatedOutputTokens=true. EndOfStream cleans up everything.
func TestMultiProfilePrefillDecodeTracking(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		"pd-producer", fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	prefillMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "prefill-pod", Namespace: "default"},
	}
	decodeMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "decode-pod", Namespace: "default"},
	}
	prefillEp := datalayer.NewEndpoint(prefillMeta, datalayer.NewMetrics())
	decodeEp := datalayer.NewEndpoint(decodeMeta, datalayer.NewMetrics())

	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: prefillEp,
	}))
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: decodeEp,
	}))

	prefillSchedEp := fwksched.NewEndpoint(prefillMeta, datalayer.NewMetrics(), nil)
	decodeSchedEp := fwksched.NewEndpoint(decodeMeta, datalayer.NewMetrics(), nil)

	request := &fwksched.InferenceRequest{
		RequestID: "pd-req-1",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 100)},
		},
	}
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prefill": {TargetEndpoints: []fwksched.Endpoint{prefillSchedEp}},
			"decode":  {TargetEndpoints: []fwksched.Endpoint{decodeSchedEp}},
		},
	}

	producer.PreRequest(ctx, request, result)

	prefillEID := prefillMeta.NamespacedName.String()
	decodeEID := decodeMeta.NamespacedName.String()

	require.Equal(t, int64(1), producer.GetRequests(prefillEID),
		"prefill endpoint should track 1 request")
	require.Equal(t, int64(1), producer.GetRequests(decodeEID),
		"decode endpoint should track 1 request")
	require.Greater(t, producer.GetTokens(prefillEID), int64(0),
		"prefill endpoint should track tokens")
	require.Greater(t, producer.GetTokens(decodeEID), int64(0),
		"decode endpoint should track tokens")

	// StartOfStream: with addEstimatedOutputTokens=false (default), tokens are
	// released on ALL profiles (prefill done, prompt cost freed).
	request.SchedulingResult = result
	producer.ResponseBody(ctx, request,
		&requestcontrol.Response{StartOfStream: true}, decodeMeta)

	require.Equal(t, int64(0), producer.GetTokens(prefillEID),
		"prefill tokens should be released at StartOfStream")
	require.Equal(t, int64(0), producer.GetTokens(decodeEID),
		"decode tokens should be released at StartOfStream")
	require.Equal(t, int64(1), producer.GetRequests(prefillEID),
		"prefill request counter should remain held until EndOfStream")
	require.Equal(t, int64(1), producer.GetRequests(decodeEID),
		"decode request counter should remain held until EndOfStream")

	// EndOfStream: full cleanup of all profiles.
	producer.ResponseBody(ctx, request,
		&requestcontrol.Response{EndOfStream: true}, decodeMeta)

	require.Equal(t, int64(0), producer.GetRequests(prefillEID),
		"prefill request counter should be zero after EndOfStream")
	require.Equal(t, int64(0), producer.GetRequests(decodeEID),
		"decode request counter should be zero after EndOfStream")
}

// TestConcurrentSaturationReads verifies no data races when multiple goroutines
// read saturation while requests are being tracked and released concurrently.
func TestConcurrentSaturationReads(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	producerName := "race-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxConcurrency": 100, "inFlightLoadProducerName": %q}`, producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"race-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	detector := detectorPlugin.(flowcontrol.SaturationDetector)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	endpoints := []datalayer.Endpoint{ep}

	// Use a start barrier to ensure both goroutines begin concurrently.
	start := make(chan struct{})
	var wg sync.WaitGroup

	// Writer: track and release 200 requests rapidly.
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 200; i++ {
			schedEp := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
			req := &fwksched.InferenceRequest{
				RequestID: fmt.Sprintf("req-%d", i),
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 10)},
				},
			}
			result := &fwksched.SchedulingResult{
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
				},
			}
			producer.PreRequest(ctx, req, result)
			req.SchedulingResult = result
			producer.ResponseBody(ctx, req,
				&requestcontrol.Response{EndOfStream: true}, epMeta)
		}
	}()

	// Reader: read saturation 200 times concurrently with the writer.
	// Use assert (not require) and track failures via atomic — require calls
	// t.FailNow() which must not be called from non-test goroutines.
	var saturationViolations atomic.Int32
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 200; i++ {
			sat := detector.Saturation(ctx, endpoints)
			if sat < 0.0 || sat > 1.0 {
				saturationViolations.Add(1)
			}
		}
	}()

	close(start)
	wg.Wait()

	require.Equal(t, int32(0), saturationViolations.Load(),
		"saturation was outside [0.0, 1.0] during concurrent reads")
	require.InDelta(t, 0.0, detector.Saturation(ctx, endpoints), 1e-9,
		"saturation must return to exactly 0 after all concurrent operations complete")
}

// ============================================================================
// Full-Loop Controller Tests (detector wired into dispatch cycle)
// ============================================================================

// TestSaturationFullLoop wires a real InFlightLoadProducer, real concurrency
// detector, and real persistent endpoints into the FlowController's dispatch
// cycle via EndpointCandidates.Locate(). Tests the exact path that broke
// in #1474.
func TestSaturationFullLoop(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()
		time.Sleep(50 * time.Millisecond)
	})

	handle := igwtestutils.NewTestHandle(ctx)

	producerName := "fullloop-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxConcurrency": 2, "inFlightLoadProducerName": %q}`, producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"fullloop-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	realDetector := detectorPlugin.(flowcontrol.SaturationDetector)

	// Create persistent endpoints and register DynamicAttributes via Extract().
	ep1Meta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep1 := datalayer.NewEndpoint(ep1Meta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep1,
	}))

	// EndpointCandidates returns the SAME persistent endpoint objects that
	// Extract() registered DynamicAttributes on. This is the critical contract.
	persistentEndpoints := []datalayer.Endpoint{ep1}
	endpointCandidates := &contractmocks.MockEndpointCandidates{
		Candidates: persistentEndpoints,
	}

	oPolicy, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
	require.NoError(t, err)
	fPolicy, err := globalstrict.GlobalStrictFairnessPolicyFactory("gs", nil, handle)
	require.NoError(t, err)

	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: oPolicy.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fPolicy.(flowcontrol.FairnessPolicy),
	}
	band, err := registry.NewPriorityBandConfig(0, defaults,
		registry.WithBandMaxBytes(10_000_000_000),
	)
	require.NoError(t, err)
	regCfg, err := registry.NewConfig(defaults, registry.WithPriorityBand(band))
	require.NoError(t, err)

	reg := registry.NewFlowRegistry(regCfg, logr.Discard())
	go reg.Run(ctx)

	fc := controller.NewFlowController(ctx, "fullloop-test", &controller.Config{
		DefaultRequestTTL:        5 * time.Minute,
		ExpiryCleanupInterval:    10 * time.Millisecond,
		EnqueueChannelBufferSize: 100,
	}, controller.Deps{
		Registry:           reg,
		SaturationDetector: realDetector,
		EndpointCandidates: endpointCandidates,
		UsageLimitPolicy:   usagelimits.DefaultPolicy(),
	})

	time.Sleep(10 * time.Millisecond)

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Simulate 2 in-flight requests (maxConcurrency=2 on 1 endpoint -> saturation=1.0).
	for i := 0; i < 2; i++ {
		schedEp := fwksched.NewEndpoint(ep1Meta, datalayer.NewMetrics(), nil)
		req := &fwksched.InferenceRequest{
			RequestID: fmt.Sprintf("prefill-%d", i),
			Body: &fwkrh.InferenceRequestBody{
				TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 50)},
			},
		}
		result := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
			},
		}
		producer.PreRequest(ctx, req, result)
	}

	// Verify the detector sees saturation via the real Locate->Saturation path.
	sat := realDetector.Saturation(ctx, persistentEndpoints)
	require.InDelta(t, 1.0, sat, 1e-9,
		"2 in-flight requests with maxConcurrency=2 should report full saturation")

	// Now enqueue a request through the FlowController. Since saturation=1.0,
	// the dispatch cycle should NOT dispatch it -- it should queue.
	results := make(chan dispatchResult, 1)
	go func() {
		reqCtx, reqCancel := context.WithTimeout(ctx, 1*time.Second)
		defer reqCancel()
		req := &testRequest{id: "queued-req", key: key, byteSize: 100, ttl: 1 * time.Second}
		outcome, err := fc.EnqueueAndWait(reqCtx, req)
		results <- dispatchResult{id: "queued-req", outcome: outcome, err: err}
	}()

	// The request has a 1s context timeout. It must NOT dispatch and must
	// return with an eviction/rejection outcome within that timeout.
	select {
	case r := <-results:
		require.NotEqual(t, fcTypes.QueueOutcomeDispatched, r.outcome,
			"request dispatched despite full saturation -- "+
				"the Locate->Saturation data path is broken (regression of #1474)")
		require.Error(t, r.err, "saturated request should return an error (TTL or deadline)")
	case <-time.After(3 * time.Second):
		t.Fatal("request did not finalize within 3s -- possible dispatch cycle hang under full saturation")
	}
}

// ============================================================================
// Utilization Detector Test
// ============================================================================

// TestUtilizationDetectorMetricsPath verifies that the utilization detector
// reads WaitingQueueSize and KVCacheUsagePercent from real endpoint metrics.
func TestUtilizationDetectorMetricsPath(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	detectorPlugin, err := utilization.UtilizationDetectorFactory(
		"util-detector",
		fwkplugin.StrictDecoder([]byte(`{"queueDepthThreshold": 10, "kvCacheUtilThreshold": 0.8}`)),
		handle,
	)
	require.NoError(t, err)
	detector := detectorPlugin.(flowcontrol.SaturationDetector)

	makeEndpoint := func(name string, queueDepth int, kvCache float64) datalayer.Endpoint {
		m := datalayer.NewMetrics()
		m.WaitingQueueSize = queueDepth
		m.KVCacheUsagePercent = kvCache
		m.UpdateTime = time.Now()
		return datalayer.NewEndpoint(
			&datalayer.EndpointMetadata{
				NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
			}, m,
		)
	}

	t.Run("healthy endpoints report low saturation", func(t *testing.T) {
		eps := []datalayer.Endpoint{
			makeEndpoint("pod-1", 2, 0.3),
			makeEndpoint("pod-2", 1, 0.2),
		}
		sat := detector.Saturation(ctx, eps)
		require.Less(t, sat, 0.5, "healthy endpoints should have low saturation")
	})

	t.Run("queue-saturated endpoint drives saturation up", func(t *testing.T) {
		eps := []datalayer.Endpoint{
			makeEndpoint("pod-1", 15, 0.1),
		}
		sat := detector.Saturation(ctx, eps)
		require.Greater(t, sat, 1.0,
			"queue depth exceeding threshold should report saturation > 1.0")
	})

	t.Run("kv-cache-saturated endpoint drives saturation up", func(t *testing.T) {
		eps := []datalayer.Endpoint{
			makeEndpoint("pod-1", 0, 0.95),
		}
		sat := detector.Saturation(ctx, eps)
		require.Greater(t, sat, 1.0,
			"KV cache exceeding threshold should report saturation > 1.0")
	})

	t.Run("stale metrics report full saturation", func(t *testing.T) {
		m := datalayer.NewMetrics()
		m.WaitingQueueSize = 0
		m.KVCacheUsagePercent = 0.0
		m.UpdateTime = time.Now().Add(-1 * time.Hour)
		ep := datalayer.NewEndpoint(
			&datalayer.EndpointMetadata{
				NamespacedName: types.NamespacedName{Name: "stale-pod", Namespace: "default"},
			}, m,
		)
		sat := detector.Saturation(ctx, []datalayer.Endpoint{ep})
		require.InDelta(t, 1.0, sat, 1e-9,
			"stale metrics should be treated as fully saturated")
	})

	t.Run("empty endpoint list fails closed", func(t *testing.T) {
		sat := detector.Saturation(ctx, []datalayer.Endpoint{})
		require.InDelta(t, 1.0, sat, 1e-9)
	})
}

// ============================================================================
// Dispatch Ordering Tests (SLO deadline)
// ============================================================================

// TestDispatchOrderingSLODeadline verifies that the SLO deadline ordering policy
// reads the x-llm-d-slo-ttft-ms header from real InferenceRequests.
func TestDispatchOrderingSLODeadline(t *testing.T) {
	t.Parallel()

	handle := igwtestutils.NewTestHandle(t.Context())
	sloPlugin, err := slodeadline.SLODeadlineOrderingPolicyFactory("slo", nil, handle)
	require.NoError(t, err)

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{
		ordering: sloPlugin.(flowcontrol.OrderingPolicy),
		detector: detector,
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}
	now := time.Now()

	type reqSpec struct {
		id    string
		sloMs string
	}
	specs := []reqSpec{
		{"loose-10s", "10000"},
		{"tight-500ms", "500"},
		{"mid-2s", "2000"},
	}

	results := make(chan dispatchResult, len(specs))
	for _, s := range specs {
		go func() {
			req := &testRequest{
				id: s.id, key: key, byteSize: 100, ttl: 5 * time.Minute,
				timestamp: now,
				infReq: &fwksched.InferenceRequest{
					RequestID: s.id,
					Headers:   map[string]string{metadata.TTFTSLOHeaderKey: s.sloMs},
				},
			}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			results <- dispatchResult{id: s.id, outcome: outcome, err: err}
			detector.Release()
		}()
		time.Sleep(5 * time.Millisecond)
	}

	time.Sleep(20 * time.Millisecond)
	detector.Unblock(1)

	var dispatchOrder []string
	for i := 0; i < len(specs); i++ {
		select {
		case r := <-results:
			require.NoError(t, r.err)
			require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
			dispatchOrder = append(dispatchOrder, r.id)
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for dispatch %d", i)
		}
	}

	require.Equal(t, "tight-500ms", dispatchOrder[0], "tightest SLO should dispatch first")
	require.Equal(t, "mid-2s", dispatchOrder[1], "middle SLO should dispatch second")
	require.Equal(t, "loose-10s", dispatchOrder[2], "loosest SLO should dispatch last")
}

// ============================================================================
// Priority and Fairness Tests
// ============================================================================

func TestPriorityBackpressure(t *testing.T) {
	t.Parallel()

	handle := igwtestutils.NewTestHandle(t.Context())

	oPolicy, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
	require.NoError(t, err)
	fPolicy, err := globalstrict.GlobalStrictFairnessPolicyFactory("gs", nil, handle)
	require.NoError(t, err)

	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: oPolicy.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fPolicy.(flowcontrol.FairnessPolicy),
	}

	highBand, err := registry.NewPriorityBandConfig(10, defaults,
		registry.WithBandMaxBytes(10_000_000_000),
	)
	require.NoError(t, err)
	lowBand, err := registry.NewPriorityBandConfig(0, defaults,
		registry.WithBandMaxBytes(10_000_000_000),
	)
	require.NoError(t, err)

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{
		detector: detector,
		bands:    []*registry.PriorityBandConfig{highBand, lowBand},
	})

	highKey := flowcontrol.FlowKey{ID: "high-flow", Priority: 10}
	lowKey := flowcontrol.FlowKey{ID: "low-flow", Priority: 0}

	results := make(chan dispatchResult, 4)

	enqueue := func(id string, key flowcontrol.FlowKey) {
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			results <- dispatchResult{id: id, outcome: outcome, err: err}
			detector.Release()
		}()
	}

	enqueue("low-1", lowKey)
	time.Sleep(5 * time.Millisecond)
	enqueue("high-1", highKey)
	time.Sleep(20 * time.Millisecond)

	detector.Unblock(1)

	var dispatchOrder []string
	for i := 0; i < 2; i++ {
		select {
		case r := <-results:
			require.NoError(t, r.err)
			require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
			dispatchOrder = append(dispatchOrder, r.id)
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for dispatch %d", i)
		}
	}

	require.Equal(t, "high-1", dispatchOrder[0],
		"high-priority should dispatch before low-priority under backpressure")
	require.Equal(t, "low-1", dispatchOrder[1])
}

func TestFairnessRoundRobin(t *testing.T) {
	t.Parallel()

	handle := igwtestutils.NewTestHandle(t.Context())
	rrPlugin, err := roundrobin.RoundRobinFairnessPolicyFactory("rr", nil, handle)
	require.NoError(t, err)

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{
		fairness: rrPlugin.(flowcontrol.FairnessPolicy),
		detector: detector,
	})

	flows := []string{"flow-a", "flow-b", "flow-c"}
	const reqsPerFlow = 3
	total := len(flows) * reqsPerFlow

	results := make(chan dispatchResult, total)

	for _, flow := range flows {
		for i := 0; i < reqsPerFlow; i++ {
			id := fmt.Sprintf("%s-req-%d", flow, i)
			key := flowcontrol.FlowKey{ID: flow, Priority: 0}
			go func() {
				req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
				outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
				results <- dispatchResult{id: id, flowID: key.ID, outcome: outcome, err: err}
				detector.Release()
			}()
			time.Sleep(2 * time.Millisecond)
		}
	}

	time.Sleep(30 * time.Millisecond)
	detector.Unblock(1)

	var dispatchOrder []string
	for i := 0; i < total; i++ {
		select {
		case r := <-results:
			require.NoError(t, r.err)
			require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
			dispatchOrder = append(dispatchOrder, r.flowID)
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for dispatch %d of %d", i, total)
		}
	}

	require.Len(t, dispatchOrder, total)

	for round := 0; round < reqsPerFlow; round++ {
		start := round * len(flows)
		end := start + len(flows)
		if end > len(dispatchOrder) {
			break
		}
		chunk := dispatchOrder[start:end]
		seen := map[string]bool{}
		for _, f := range chunk {
			seen[f] = true
		}
		require.Len(t, seen, len(flows),
			"round %d: expected all %d flows in dispatch chunk %v", round, len(flows), chunk)
	}
}

// TestUsageLimitThresholdGatesDispatch verifies that a UsageLimitPolicy with
// threshold < 1.0 triggers HoL blocking at partial saturation.
// With threshold=0.5, the dispatch cycle should block when saturation >= 0.5.
func TestUsageLimitThresholdGatesDispatch(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()
		time.Sleep(50 * time.Millisecond)
	})

	handle := igwtestutils.NewTestHandle(ctx)

	producerName := "threshold-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	// maxConcurrency=10 on 1 endpoint. 5 in-flight -> saturation=0.5.
	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxConcurrency": 10, "inFlightLoadProducerName": %q}`, producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"threshold-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	realDetector := detectorPlugin.(flowcontrol.SaturationDetector)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	oPolicy, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
	require.NoError(t, err)
	fPolicy, err := globalstrict.GlobalStrictFairnessPolicyFactory("gs", nil, handle)
	require.NoError(t, err)

	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: oPolicy.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fPolicy.(flowcontrol.FairnessPolicy),
	}
	band, err := registry.NewPriorityBandConfig(0, defaults,
		registry.WithBandMaxBytes(10_000_000_000),
	)
	require.NoError(t, err)
	regCfg, err := registry.NewConfig(defaults, registry.WithPriorityBand(band))
	require.NoError(t, err)

	reg := registry.NewFlowRegistry(regCfg, logr.Discard())
	go reg.Run(ctx)

	// Threshold=0.5: HoL blocking triggers at 50% saturation.
	halfThresholdPolicy := usagelimits.NewConstPolicy("half", 0.5)

	fc := controller.NewFlowController(ctx, "threshold-test", &controller.Config{
		DefaultRequestTTL:        5 * time.Minute,
		ExpiryCleanupInterval:    10 * time.Millisecond,
		EnqueueChannelBufferSize: 100,
	}, controller.Deps{
		Registry:           reg,
		SaturationDetector: realDetector,
		EndpointCandidates: &contractmocks.MockEndpointCandidates{
			Candidates: []datalayer.Endpoint{ep},
		},
		UsageLimitPolicy: halfThresholdPolicy,
	})

	time.Sleep(10 * time.Millisecond)
	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Drive 5 in-flight requests -> saturation=5/10=0.5 -> meets threshold.
	for i := 0; i < 5; i++ {
		schedEp := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
		req := &fwksched.InferenceRequest{
			RequestID: fmt.Sprintf("inflight-%d", i),
			Body: &fwkrh.InferenceRequestBody{
				TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 10)},
			},
		}
		result := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
			},
		}
		producer.PreRequest(ctx, req, result)
	}

	// saturation=0.5, threshold=0.5: 0.5 >= 0.5 -> HoL blocking should trigger.
	results := make(chan dispatchResult, 1)
	go func() {
		reqCtx, reqCancel := context.WithTimeout(ctx, 500*time.Millisecond)
		defer reqCancel()
		req := &testRequest{id: "gated-req", key: key, byteSize: 100, ttl: 500 * time.Millisecond}
		outcome, err := fc.EnqueueAndWait(reqCtx, req)
		results <- dispatchResult{id: "gated-req", outcome: outcome, err: err}
	}()

	select {
	case r := <-results:
		require.NotEqual(t, fcTypes.QueueOutcomeDispatched, r.outcome,
			"request should NOT dispatch at saturation=0.5 with threshold=0.5")
		require.Error(t, r.err,
			"gated request should return an error (TTL or deadline)")
	case <-time.After(3 * time.Second):
		t.Fatal("request did not finalize within 3s -- possible dispatch cycle hang under partial saturation")
	}
}

// ============================================================================
// Capacity Enforcement Tests (bytes, requests, global vs band)
// ============================================================================

// TestGlobalAndBandCapacityInteraction verifies that the global MaxRequests
// limit rejects requests even when the per-band limit has capacity.
func TestGlobalAndBandCapacityInteraction(t *testing.T) {
	t.Parallel()

	// Band allows 10 requests, but global allows only 3.
	detector := &switchableDetector{}
	detector.blocked.Store(true)

	h := newHarness(t, harnessOpts{
		detector:        detector,
		maxRequests:     3,
		bandMaxRequests: 10,
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	results := make(chan dispatchResult, 5)
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("req-%d", i)
		go func() {
			reqCtx, reqCancel := context.WithTimeout(h.ctx, 500*time.Millisecond)
			defer reqCancel()
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 500 * time.Millisecond}
			outcome, err := h.fc.EnqueueAndWait(reqCtx, req)
			results <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
		time.Sleep(5 * time.Millisecond)
	}

	var admitted, rejected int
	for i := 0; i < 5; i++ {
		select {
		case r := <-results:
			if r.outcome == fcTypes.QueueOutcomeRejectedCapacity {
				rejected++
			} else {
				admitted++
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for result %d", i)
		}
	}

	require.LessOrEqual(t, admitted, 3,
		"global MaxRequests=3 should cap admissions even though band allows 10")
	require.GreaterOrEqual(t, rejected, 2,
		"at least 2 requests should be rejected by global limit")
}

// ============================================================================
// Eviction Pipeline Tests
// ============================================================================

// TestEvictionPipeline wires the real eviction components together:
// RequestEvictor + SheddableFilter + PriorityTimeOrdering + ImmediateResponseEvictor.
// Verifies: PreRequest->queue tracking->EvictN->channel closure->cleanup.
func TestEvictionPipeline(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	orderPlugin, err := evictionordering.PriorityThenTimeOrderingFactory("evict-order", nil, handle)
	require.NoError(t, err)
	filterPlugin, err := filtering.SheddableFilterFactory("evict-filter", nil, handle)
	require.NoError(t, err)

	evictor := eviction.NewImmediateResponseEvictor()
	requestEvictor := eviction.NewRequestEvictor(
		orderPlugin.(flowcontrol.EvictionOrderingPolicy),
		filterPlugin.(flowcontrol.EvictionFilterPolicy),
		evictor,
	)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
		Address:        "10.0.0.1",
		Port:           "8000",
	}
	schedEndpoint := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	makeResult := func() *fwksched.SchedulingResult {
		return &fwksched.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode": {TargetEndpoints: []fwksched.Endpoint{schedEndpoint}},
			},
		}
	}

	// Sheddable requests have priority < 0.
	sheddableReq := &fwksched.InferenceRequest{
		RequestID:  "shed-1",
		Headers:    map[string]string{reqcommon.RequestIDHeaderKey: "shed-1"},
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}
	// Non-sheddable requests have priority >= 0.
	protectedReq := &fwksched.InferenceRequest{
		RequestID:  "protect-1",
		Headers:    map[string]string{reqcommon.RequestIDHeaderKey: "protect-1"},
		Objectives: fwksched.RequestObjectives{Priority: 1},
	}

	reqCtx, reqCancel := context.WithCancel(ctx)
	defer reqCancel()

	requestEvictor.PreRequest(reqCtx, sheddableReq, makeResult())
	requestEvictor.PreRequest(reqCtx, protectedReq, makeResult())

	inFlight, evictable := requestEvictor.Stats()
	require.Equal(t, 2, inFlight, "both requests should be tracked in-flight")
	require.Equal(t, 1, evictable,
		"only the sheddable request (priority < 0) should be evictable")

	reg := requestEvictor.EvictionRegistry()
	sheddableCh := reg.Get("shed-1")
	require.NotNil(t, sheddableCh, "sheddable request should have an eviction channel")

	evictedIDs, err := requestEvictor.EvictN(ctx, 1)
	require.NoError(t, err)
	require.Equal(t, []string{"shed-1"}, evictedIDs)

	select {
	case <-sheddableCh:
		// Channel was closed by ImmediateResponseEvictor -- eviction signaled.
	default:
		t.Fatal("eviction channel should be closed after EvictN")
	}

	// Protected request's channel should still be open.
	protectedCh := reg.Get("protect-1")
	require.NotNil(t, protectedCh)
	select {
	case <-protectedCh:
		t.Fatal("protected request's channel should not be closed")
	default:
	}

	// Complete the protected request normally.
	requestEvictor.ResponseBody(ctx, protectedReq,
		&requestcontrol.Response{EndOfStream: true}, epMeta)

	inFlight, evictable = requestEvictor.Stats()
	require.Equal(t, 0, inFlight, "all requests should be cleaned up after eviction and completion")
	require.Equal(t, 0, evictable)
}

// ============================================================================
// Error/Non-Happy-Path Tests (TTL, context cancel)
// ============================================================================

// TestTTLExpiryEvictsQueuedRequest verifies that a request queued under
// saturation is evicted with QueueOutcomeEvictedTTL + ErrTTLExpired when its
// TTL expires.
func TestTTLExpiryEvictsQueuedRequest(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{detector: detector})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	results := make(chan dispatchResult, 1)
	go func() {
		req := &testRequest{id: "ttl-req", key: key, byteSize: 100, ttl: 100 * time.Millisecond}
		outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
		results <- dispatchResult{id: "ttl-req", outcome: outcome, err: err}
	}()

	select {
	case r := <-results:
		require.Error(t, r.err)
		require.ErrorIs(t, r.err, fcTypes.ErrEvicted,
			"TTL-expired request should be wrapped with ErrEvicted")
		require.ErrorIs(t, r.err, fcTypes.ErrTTLExpired,
			"TTL-expired request should contain ErrTTLExpired")
		require.Equal(t, fcTypes.QueueOutcomeEvictedTTL, r.outcome,
			"TTL-expired request should have outcome QueueOutcomeEvictedTTL")
	case <-time.After(5 * time.Second):
		t.Fatal("request did not return after TTL expiry")
	}
}

// TestCallerContextCancellationEvictsRequest verifies that cancelling the
// caller's context while a request is queued produces the correct eviction
// outcome and error chain.
func TestCallerContextCancellationEvictsRequest(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{detector: detector})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	reqCtx, reqCancel := context.WithCancel(h.ctx)
	results := make(chan dispatchResult, 1)
	go func() {
		req := &testRequest{id: "cancel-req", key: key, byteSize: 100, ttl: 5 * time.Minute}
		outcome, err := h.fc.EnqueueAndWait(reqCtx, req)
		results <- dispatchResult{id: "cancel-req", outcome: outcome, err: err}
	}()

	time.Sleep(30 * time.Millisecond)
	reqCancel()

	select {
	case r := <-results:
		require.Error(t, r.err)
		require.ErrorIs(t, r.err, fcTypes.ErrEvicted,
			"cancelled request should be wrapped with ErrEvicted")
		require.Equal(t, fcTypes.QueueOutcomeEvictedContextCancelled, r.outcome,
			"cancelled request should have QueueOutcomeEvictedContextCancelled")
	case <-time.After(5 * time.Second):
		t.Fatal("request did not return after context cancellation")
	}
}

// ============================================================================
// Shutdown and Lifecycle Tests
// ============================================================================

// TestConcurrentEnqueueDuringShutdown verifies there are no races or panics
// when requests are being enqueued concurrently with controller shutdown.
func TestConcurrentEnqueueDuringShutdown(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()

	h := newHarness(t, harnessOpts{
		detector: detector,
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}
	const numRequests = 50

	results := make(chan dispatchResult, numRequests)
	for i := 0; i < numRequests; i++ {
		id := fmt.Sprintf("req-%d", i)
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			results <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
	}

	// Cancel mid-flight while goroutines are still enqueuing.
	time.Sleep(10 * time.Millisecond)
	h.cancel()

	for i := 0; i < numRequests; i++ {
		select {
		case r := <-results:
			// Every request must reach a terminal state -- no panics, no hangs.
			require.NotEqual(t, fcTypes.QueueOutcomeDispatched, r.outcome,
				"no request should dispatch (detector is blocked and controller is shutting down)")
			require.Error(t, r.err,
				"every request should receive an error during shutdown")
		case <-time.After(5 * time.Second):
			t.Fatalf("request %d hung during concurrent shutdown", i)
		}
	}
}

// TestGracefulShutdownDrainsQueuedRequests verifies that when the controller's
// context is cancelled (simulating pod termination), all queued requests receive
// a clean eviction outcome rather than hanging or panicking.
func TestGracefulShutdownDrainsQueuedRequests(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()

	h := newHarness(t, harnessOpts{
		detector: detector,
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}
	const numRequests = 10

	results := make(chan dispatchResult, numRequests)
	for i := 0; i < numRequests; i++ {
		id := fmt.Sprintf("req-%d", i)
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			results <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
		time.Sleep(2 * time.Millisecond)
	}

	// Let all requests queue (detector is blocked).
	time.Sleep(30 * time.Millisecond)

	// Simulate pod termination: cancel the controller context.
	h.cancel()

	var evicted, errored int
	for i := 0; i < numRequests; i++ {
		select {
		case r := <-results:
			require.Error(t, r.err, "queued request should receive an error on shutdown")
			switch r.outcome {
			case fcTypes.QueueOutcomeEvictedOther,
				fcTypes.QueueOutcomeEvictedContextCancelled,
				fcTypes.QueueOutcomeRejectedOther:
				evicted++
			default:
				errored++
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("request %d did not return within 5s of shutdown -- possible hang", i)
		}
	}

	require.Equal(t, numRequests, evicted+errored,
		"all queued requests must reach a terminal state on shutdown")
	require.Equal(t, numRequests, evicted,
		"all queued requests should be evicted or rejected, not silently dropped")
}

// ============================================================================
// Production Edge Cases
// ============================================================================

// TestZombieCapacityStarvation verifies that TTL-expired items still in the
// queue (zombies) consume capacity until the cleanup sweep runs. If the sweep
// interval is long, new requests are falsely rejected because capacity is held
// by dead items.
func TestZombieCapacityStarvation(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()

	h := newHarness(t, harnessOpts{
		detector:        detector,
		bandMaxRequests: 3,
		controllerCfg: &controller.Config{
			DefaultRequestTTL:        50 * time.Millisecond,
			ExpiryCleanupInterval:    10 * time.Second,
			EnqueueChannelBufferSize: 100,
		},
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Fill capacity with 3 requests that will expire via TTL.
	expired := make(chan dispatchResult, 3)
	for i := 0; i < 3; i++ {
		id := fmt.Sprintf("zombie-%d", i)
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 50 * time.Millisecond}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			expired <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
		time.Sleep(5 * time.Millisecond)
	}

	// Wait for all to expire.
	for i := 0; i < 3; i++ {
		select {
		case r := <-expired:
			require.ErrorIs(t, r.err, fcTypes.ErrTTLExpired)
		case <-time.After(5 * time.Second):
			t.Fatalf("zombie %d did not expire", i)
		}
	}

	// All 3 expired, but cleanup hasn't run (interval=10s).
	// The new request is rejected because zombies still consume capacity
	// in the registry's atomic counters.
	newResult := make(chan dispatchResult, 1)
	go func() {
		reqCtx, reqCancel := context.WithTimeout(h.ctx, 200*time.Millisecond)
		defer reqCancel()
		req := &testRequest{id: "post-zombie", key: key, byteSize: 100, ttl: 200 * time.Millisecond}
		outcome, err := h.fc.EnqueueAndWait(reqCtx, req)
		newResult <- dispatchResult{id: "post-zombie", outcome: outcome, err: err}
	}()

	select {
	case r := <-newResult:
		// Zombie capacity starvation: the 3 expired items still occupy
		// capacity slots until the cleanup sweep reclaims them (interval=10s).
		require.Equal(t, fcTypes.QueueOutcomeRejectedCapacity, r.outcome,
			"post-zombie request should be rejected -- expired items consume capacity until cleanup sweep runs")
	case <-time.After(5 * time.Second):
		t.Fatal("post-zombie request hung")
	}
}

// TestSingleChunkResponseLifecycle verifies the producer handles a single-chunk
// response correctly, where StartOfStream=true and EndOfStream=true arrive in
// the same ResponseBody call. Tokens should release at StartOfStream, then
// PluginState.Delete at EndOfStream fires OnEvicted which should be idempotent.
func TestSingleChunkResponseLifecycle(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		"single-chunk", fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	schedEndpoint := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	request := &fwksched.InferenceRequest{
		RequestID: "single-chunk-1",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 100)},
		},
	}
	result := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{schedEndpoint}},
		},
	}

	producer.PreRequest(ctx, request, result)
	eid := epMeta.NamespacedName.String()
	require.Equal(t, int64(1), producer.GetRequests(eid))
	require.Greater(t, producer.GetTokens(eid), int64(0))

	// Single-chunk response: both flags set simultaneously.
	request.SchedulingResult = result
	require.NotPanics(t, func() {
		producer.ResponseBody(ctx, request,
			&requestcontrol.Response{StartOfStream: true, EndOfStream: true}, epMeta)
	})

	require.Equal(t, int64(0), producer.GetRequests(eid),
		"request counter should be zero after single-chunk response")
	require.Equal(t, int64(0), producer.GetTokens(eid),
		"token counter should be zero after single-chunk response")
}

// TestEndpointReregistrationSaturationAccuracy verifies that when an endpoint
// is deleted and re-added (pod cycling), the saturation detector accurately
// reflects the state -- in-flight requests from before the delete should not
// leak into the new tracker.
func TestEndpointReregistrationSaturationAccuracy(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	producerName := "rereg-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxConcurrency": 10, "inFlightLoadProducerName": %q}`, producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"rereg-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	detector := detectorPlugin.(flowcontrol.SaturationDetector)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	// Track a request on the original endpoint.
	schedEp := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	oldReq := &fwksched.InferenceRequest{
		RequestID: "old-req",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 50)},
		},
	}
	oldResult := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
		},
	}
	producer.PreRequest(ctx, oldReq, oldResult)

	sat := detector.Saturation(ctx, []datalayer.Endpoint{ep})
	require.Greater(t, sat, 0.0, "saturation should be nonzero with in-flight request")

	// Simulate pod cycling: delete + re-add.
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventDelete, Endpoint: ep,
	}))

	// Re-create the endpoint (simulates new pod with same name).
	newEp := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: newEp,
	}))

	// The new endpoint should have a fresh tracker -- saturation should be 0.
	newSat := detector.Saturation(ctx, []datalayer.Endpoint{newEp})
	require.InDelta(t, 0.0, newSat, 1e-9,
		"re-registered endpoint should have zero saturation (fresh tracker)")

	// Complete the old request -- should not panic or corrupt the new tracker.
	require.NotPanics(t, func() {
		oldReq.SchedulingResult = oldResult
		producer.ResponseBody(ctx, oldReq,
			&requestcontrol.Response{EndOfStream: true}, epMeta)
	})

	// New tracker should still be at zero -- old request's cleanup must not
	// affect the new tracker.
	finalSat := detector.Saturation(ctx, []datalayer.Endpoint{newEp})
	require.InDelta(t, 0.0, finalSat, 1e-9,
		"old request completion should not affect re-registered endpoint's tracker")

	// Verify the deleted endpoint's tracker is clean.
	eid := epMeta.NamespacedName.String()
	require.Equal(t, int64(0), producer.GetRequests(eid),
		"deleted endpoint tracker should report 0 requests")

	// Track a NEW request on the re-registered endpoint, then complete it.
	// Verify the counter returns to exactly 0. In this sequential execution
	// the old OnEvicted already fired (no corruption). A negative counter
	// here would indicate the old cleanup path leaked a decrement into
	// the new tracker's lifecycle.
	newSchedEp := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	newReq := &fwksched.InferenceRequest{
		RequestID: "new-req",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 50)},
		},
	}
	newResult := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{newSchedEp}},
		},
	}
	producer.PreRequest(ctx, newReq, newResult)
	require.Equal(t, int64(1), producer.GetRequests(eid),
		"new request on re-registered endpoint should be tracked")

	newReq.SchedulingResult = newResult
	producer.ResponseBody(ctx, newReq,
		&requestcontrol.Response{EndOfStream: true}, epMeta)
	require.Equal(t, int64(0), producer.GetRequests(eid),
		"counter must be exactly 0 after new request completes -- negative value indicates old OnEvicted corrupted the new tracker")
}

// TestDetectorPanicsOnNilEndpoint documents a known bug: the concurrency
// detector does not guard against nil endpoints in the candidate list.
// It checks e.GetMetadata() == nil but not e == nil, causing a nil pointer
// dereference. This can occur if EndpointCandidates.Locate() returns a list
// with nil entries during endpoint churn.
//
// When the fix lands (add `e == nil` guard in detector.go), change
// require.Panics to require.NotPanics.
func TestDetectorPanicsOnNilEndpoint(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"nil-detector", fwkplugin.StrictDecoder([]byte(`{"maxConcurrency": 10}`)), handle,
	)
	require.NoError(t, err)
	detector := detectorPlugin.(flowcontrol.SaturationDetector)

	require.Panics(t, func() {
		detector.Saturation(ctx, []datalayer.Endpoint{nil})
	}, "KNOWN BUG: nil endpoint in candidate list causes panic")
}

// TestEndpointIdentityCollisionDuringPodReplacement documents a known bug:
// when a pod is replaced (delete old + add new with same NamespacedName), the
// stale delete event clears the new pod's tracker, causing saturation to read 0
// and flooding the replacement pod with traffic.
//
// This test asserts the BROKEN behavior. When the fix lands (tracking endpoint
// identity in Extract), flip the final assertion to require.Greater.
func TestEndpointIdentityCollisionDuringPodReplacement(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	producerName := "collision-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxConcurrency": 10, "inFlightLoadProducerName": %q}`, producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"collision-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	detector := detectorPlugin.(flowcontrol.SaturationDetector)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}

	oldEp := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: oldEp,
	}))

	newEp := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: newEp,
	}))

	schedEp := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	req := &fwksched.InferenceRequest{
		RequestID: "new-pod-req",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 50)},
		},
	}
	result := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
		},
	}
	producer.PreRequest(ctx, req, result)

	require.Greater(t, detector.Saturation(ctx, []datalayer.Endpoint{newEp}), 0.0,
		"new endpoint should show in-flight load before stale delete")

	// Stale delete for old pod clears the new pod's tracker (BUG).
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventDelete, Endpoint: oldEp,
	}))

	// BUG: saturation reads 0 because the stale delete cleared the tracker.
	// When fixed, this should be require.Greater(t, sat, 0.0, ...).
	sat := detector.Saturation(ctx, []datalayer.Endpoint{newEp})
	require.InDelta(t, 0.0, sat, 1e-9,
		"KNOWN BUG: stale delete clears new pod's tracker, saturation drops to 0")
}

// TestPluginStateDeleteTriggersCounterCleanup verifies that when PluginState
// evicts an entry (simulating the janitor's staleness reaper), the OnEvicted
// callback on addedTokensEntry correctly decrements the producer's counters.
// This is the recovery path for requests where ResponseBody never fires.
func TestPluginStateDeleteTriggersCounterCleanup(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		"janitor-test", fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	schedEp := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	req := &fwksched.InferenceRequest{
		RequestID: "orphaned-req",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{TokenIDs: make([]uint32, 100)},
		},
	}
	result := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
		},
	}

	producer.PreRequest(ctx, req, result)

	eid := epMeta.NamespacedName.String()
	require.Equal(t, int64(1), producer.GetRequests(eid))
	require.Greater(t, producer.GetTokens(eid), int64(0))

	// Simulate what the PluginState janitor does: delete the entry by requestID.
	// This fires OnEvicted on the addedTokensEntry, which should decrement counters.
	producer.PluginState.Delete("orphaned-req")

	require.Equal(t, int64(0), producer.GetRequests(eid),
		"PluginState.Delete must trigger OnEvicted which decrements the request counter")
	require.Equal(t, int64(0), producer.GetTokens(eid),
		"PluginState.Delete must trigger OnEvicted which decrements the token counter")
}

// ============================================================================
// Metrics Emission Tests
// ============================================================================

// TestFlowControlMetricsEmitted verifies that EnqueueAndWait emits queue_size
// metrics. A blocked request holds the gauge > 0 while queued; after TTL
// expiry the gauge returns to 0.
func TestFlowControlMetricsEmitted(t *testing.T) {
	t.Parallel()

	eppmetrics.Register()

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{detector: detector})

	key := flowcontrol.FlowKey{ID: "metrics-flow", Priority: 0}

	results := make(chan dispatchResult, 1)
	go func() {
		req := &testRequest{id: "metrics-req", key: key, byteSize: 512, ttl: 200 * time.Millisecond}
		outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
		results <- dispatchResult{outcome: outcome, err: err}
	}()

	// While the request is queued (blocked detector), the gauge should be > 0.
	time.Sleep(50 * time.Millisecond)

	families, gatherErr := ctrlmetrics.Registry.Gather()
	require.NoError(t, gatherErr)

	var queueSizeWhileQueued float64
	var foundQueueSize bool
	for _, f := range families {
		if f.GetName() == "llm_d_router_epp_flow_control_queue_size" {
			foundQueueSize = true
			for _, m := range f.GetMetric() {
				queueSizeWhileQueued += m.GetGauge().GetValue()
			}
		}
	}
	require.True(t, foundQueueSize,
		"llm_d_router_epp_flow_control_queue_size metric should exist")
	require.Greater(t, queueSizeWhileQueued, 0.0,
		"queue_size should be > 0 while a request is actively queued")

	select {
	case <-results:
	case <-time.After(5 * time.Second):
		t.Fatal("request did not expire")
	}
}

// ============================================================================
// Agentic Churn and Stress Tests
// ============================================================================

// TestHighConcurrencyFlowChurnNoDeadlock sends requests from many goroutines,
// each with a unique flow ID, while the registry GC runs concurrently.
// Verifies no deadlocks, panics, or data races under contention between
// WithConnection (write lock for new flows) and executeGCCycle (write lock
// for cleanup).
func TestHighConcurrencyFlowChurnNoDeadlock(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()
		time.Sleep(100 * time.Millisecond)
	})

	handle := igwtestutils.NewTestHandle(ctx)
	oPolicy, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
	require.NoError(t, err)
	fPolicy, err := globalstrict.GlobalStrictFairnessPolicyFactory("gs", nil, handle)
	require.NoError(t, err)

	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: oPolicy.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fPolicy.(flowcontrol.FairnessPolicy),
	}
	band, err := registry.NewPriorityBandConfig(0, defaults,
		registry.WithBandMaxBytes(10_000_000_000),
	)
	require.NoError(t, err)

	// Very aggressive GC to maximize contention with WithConnection.
	regCfg, err := registry.NewConfig(defaults,
		registry.WithPriorityBand(band),
		registry.WithFlowGCTimeout(50*time.Millisecond),
	)
	require.NoError(t, err)

	reg := registry.NewFlowRegistry(regCfg, logr.Discard())
	go reg.Run(ctx)

	detector := newGatedDetector(0)

	fc := controller.NewFlowController(ctx, "contention-test", &controller.Config{
		DefaultRequestTTL:        5 * time.Minute,
		ExpiryCleanupInterval:    50 * time.Millisecond,
		EnqueueChannelBufferSize: 200,
	}, controller.Deps{
		Registry:           reg,
		SaturationDetector: detector,
		EndpointCandidates: &contractmocks.MockEndpointCandidates{},
		UsageLimitPolicy:   usagelimits.DefaultPolicy(),
	})

	time.Sleep(10 * time.Millisecond)

	const numGoroutines = 50
	const reqsPerGoroutine = 20

	results := make(chan dispatchResult, numGoroutines*reqsPerGoroutine)

	for g := 0; g < numGoroutines; g++ {
		go func() {
			for r := 0; r < reqsPerGoroutine; r++ {
				id := fmt.Sprintf("g%d-r%d", g, r)
				key := flowcontrol.FlowKey{ID: id, Priority: 0}
				req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
				outcome, err := fc.EnqueueAndWait(ctx, req)
				results <- dispatchResult{id: id, outcome: outcome, err: err}
			}
		}()
	}

	total := numGoroutines * reqsPerGoroutine
	dispatched := 0
	for i := 0; i < total; i++ {
		select {
		case r := <-results:
			require.NoError(t, r.err,
				"request %s should not error under contention", r.id)
			require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
			dispatched++
		case <-time.After(30 * time.Second):
			t.Fatalf("deadlock detected: only %d/%d requests completed", dispatched, total)
		}
	}

	require.Equal(t, total, dispatched,
		"all requests should dispatch without deadlock under concurrent flow churn + GC")

	// Wait for flow GC to reclaim idle flows.
	time.Sleep(200 * time.Millisecond)

	// Verify aggregate stats are consistent after dispatch + GC.
	stats := reg.Stats()
	require.Equal(t, uint64(0), stats.TotalLen,
		"registry should have 0 queued items after all dispatched + GC")
	require.Equal(t, uint64(0), stats.TotalByteSize,
		"registry should have 0 bytes after all dispatched + GC")
}

// TestSustainedLoadCounterAccuracy verifies that under sustained concurrent
// load (many PreRequest/ResponseBody cycles), the producer's request and token
// counters return to exactly zero. Any drift indicates a counter leak.
func TestSustainedLoadCounterAccuracy(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := igwtestutils.NewTestHandle(ctx)

	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		"sustained", fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	eid := epMeta.NamespacedName.String()
	const numWorkers = 20
	const reqsPerWorker = 50

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := 0; r < reqsPerWorker; r++ {
				reqID := fmt.Sprintf("w%d-r%d", w, r)
				schedEp := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
				req := &fwksched.InferenceRequest{
					RequestID: reqID,
					Body: &fwkrh.InferenceRequestBody{
						TokenizedPrompt: &fwkrh.TokenizedPrompt{
							TokenIDs: make([]uint32, 10+r),
						},
					},
				}
				result := &fwksched.SchedulingResult{
					ProfileResults: map[string]*fwksched.ProfileRunResult{
						"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
					},
				}

				producer.PreRequest(ctx, req, result)

				req.SchedulingResult = result
				producer.ResponseBody(ctx, req,
					&requestcontrol.Response{StartOfStream: true}, epMeta)
				producer.ResponseBody(ctx, req,
					&requestcontrol.Response{EndOfStream: true}, epMeta)
			}
		}()
	}

	wg.Wait()

	require.Equal(t, int64(0), producer.GetRequests(eid),
		"request counter must be exactly zero after all requests complete -- any drift is a leak")
	require.Equal(t, int64(0), producer.GetTokens(eid),
		"token counter must be exactly zero after all requests complete -- any drift is a leak")
}

// TestEndpointChurnUnderLoad verifies that when endpoints disappear and reappear
// while requests are queued, the dispatch cycle reacts correctly:
//   - Endpoints gone -> Saturation()=1.0 (fail closed) -> requests stay queued
//   - Endpoints return -> Saturation() drops -> requests dispatch
func TestEndpointChurnUnderLoad(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()
		time.Sleep(50 * time.Millisecond)
	})

	handle := igwtestutils.NewTestHandle(ctx)

	producerName := "churn-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxConcurrency": 100, "inFlightLoadProducerName": %q}`, producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"churn-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	realDetector := detectorPlugin.(flowcontrol.SaturationDetector)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	// Mutable endpoint list that the dispatch cycle reads via Locate().
	var endpointsMu atomic.Value
	endpointsMu.Store([]datalayer.Endpoint{ep})

	endpointCandidates := &contractmocks.MockEndpointCandidates{
		LocateFunc: func(_ context.Context, _ map[string]any) []datalayer.Endpoint {
			return endpointsMu.Load().([]datalayer.Endpoint)
		},
	}

	oPolicy, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
	require.NoError(t, err)
	fPolicy, err := globalstrict.GlobalStrictFairnessPolicyFactory("gs", nil, handle)
	require.NoError(t, err)

	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: oPolicy.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fPolicy.(flowcontrol.FairnessPolicy),
	}
	band, err := registry.NewPriorityBandConfig(0, defaults,
		registry.WithBandMaxBytes(10_000_000_000),
	)
	require.NoError(t, err)
	regCfg, err := registry.NewConfig(defaults, registry.WithPriorityBand(band))
	require.NoError(t, err)

	reg := registry.NewFlowRegistry(regCfg, logr.Discard())
	go reg.Run(ctx)

	fc := controller.NewFlowController(ctx, "churn-test", &controller.Config{
		DefaultRequestTTL:        5 * time.Minute,
		ExpiryCleanupInterval:    10 * time.Millisecond,
		EnqueueChannelBufferSize: 100,
	}, controller.Deps{
		Registry:           reg,
		SaturationDetector: realDetector,
		EndpointCandidates: endpointCandidates,
		UsageLimitPolicy:   usagelimits.DefaultPolicy(),
	})

	time.Sleep(10 * time.Millisecond)

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Phase 1: Endpoints present, no load -> requests should dispatch.
	results := make(chan dispatchResult, 1)
	go func() {
		req := &testRequest{id: "before-churn", key: key, byteSize: 100, ttl: 5 * time.Second}
		outcome, err := fc.EnqueueAndWait(ctx, req)
		results <- dispatchResult{id: "before-churn", outcome: outcome, err: err}
	}()

	select {
	case r := <-results:
		require.NoError(t, r.err)
		require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome,
			"request should dispatch when endpoints are healthy")
	case <-time.After(3 * time.Second):
		t.Fatal("request did not dispatch with healthy endpoints")
	}

	// Phase 2: Remove all endpoints -> Locate() returns empty -> saturation=1.0.
	endpointsMu.Store([]datalayer.Endpoint{})

	go func() {
		reqCtx, reqCancel := context.WithTimeout(ctx, 500*time.Millisecond)
		defer reqCancel()
		req := &testRequest{id: "during-churn", key: key, byteSize: 100, ttl: 500 * time.Millisecond}
		outcome, err := fc.EnqueueAndWait(reqCtx, req)
		results <- dispatchResult{id: "during-churn", outcome: outcome, err: err}
	}()

	select {
	case r := <-results:
		// Pool is empty → saturation=1.0 → request queues → 500ms timeout fires.
		require.NotEqual(t, fcTypes.QueueOutcomeDispatched, r.outcome,
			"request should not dispatch when all endpoints are gone")
		require.Error(t, r.err, "request should return an error when endpoints are gone")
		require.ErrorIs(t, r.err, fcTypes.ErrEvicted,
			"request should be evicted (TTL/deadline) when endpoints are gone, not silently dropped")
	case <-time.After(3 * time.Second):
		t.Fatal("request did not terminate after endpoint removal")
	}

	// Phase 3: Restore endpoints -> requests should dispatch again.
	endpointsMu.Store([]datalayer.Endpoint{ep})

	go func() {
		req := &testRequest{id: "after-churn", key: key, byteSize: 100, ttl: 5 * time.Second}
		outcome, err := fc.EnqueueAndWait(ctx, req)
		results <- dispatchResult{id: "after-churn", outcome: outcome, err: err}
	}()

	select {
	case r := <-results:
		require.NoError(t, r.err)
		require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome,
			"request should dispatch after endpoints are restored")
	case <-time.After(3 * time.Second):
		t.Fatal("request did not dispatch after endpoint restoration")
	}
}
