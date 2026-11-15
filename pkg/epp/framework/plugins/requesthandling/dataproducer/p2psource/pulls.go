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

package p2psource

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

var inFlightPulls = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: metricsutil.LLMDRouterEndpointPickerSubsystem,
		Name:      "p2p_inflight_pulls",
		Help:      metricsutil.HelpMsgWithStability("Current number of P2P KV cache pulls directed at each source endpoint, as tracked by the p2p-source-producer.", compbasemetrics.ALPHA),
	},
	[]string{"endpoint_name", "namespace", "producer_name"},
)

var inFlightPullTokens = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: metricsutil.LLMDRouterEndpointPickerSubsystem,
		Name:      "p2p_inflight_pull_tokens",
		Help:      metricsutil.HelpMsgWithStability("Current number of prompt tokens in the P2P KV cache pulls directed at each source endpoint, as tracked by the p2p-source-producer.", compbasemetrics.ALPHA),
	},
	[]string{"endpoint_name", "namespace", "producer_name"},
)

// registerPullMetrics registers the in-flight pull gauges. They are
// package-level vectors, so a second producer instance registering them is a
// no-op; producer_name keeps each instance's series distinct.
func registerPullMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("in-flight pull metrics registerer is required")
	}
	for _, gauge := range []*prometheus.GaugeVec{inFlightPulls, inFlightPullTokens} {
		if err := registerer.Register(gauge); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == gauge {
				continue
			}
			return fmt.Errorf("register in-flight pull metric: %w", err)
		}
	}
	return nil
}

const pullStateKey plugin.StateKey = "inflight-pull"

// pullTracker counts the pulls one producer has directed at each source
// endpoint and not yet released, and the prompt tokens they pull. Each
// request holds at most one pull, kept in PluginState so the janitor releases
// it if no response or abandonment ever arrives.
type pullTracker struct {
	producerName string
	state        *plugin.PluginState

	mu    sync.Mutex
	loads map[fwkdl.ID]*pullLoad
	// endpoints holds the endpoint last added under each ID, so a delete for
	// an endpoint that has since been replaced is ignored.
	endpoints map[fwkdl.ID]fwkdl.Endpoint
}

// pullLoad is a source's unreleased pulls and the prompt tokens they pull.
type pullLoad struct {
	pulls  int
	tokens int
}

func newPullTracker(ctx context.Context, producerName string) *pullTracker {
	return &pullTracker{
		producerName: producerName,
		state:        plugin.NewPluginState(ctx),
		loads:        map[fwkdl.ID]*pullLoad{},
		endpoints:    map[fwkdl.ID]fwkdl.Endpoint{},
	}
}

// acquire counts a pull from source for the request and adds its tokens to
// the source's load. A second acquire for the same request is ignored. ctx
// keeps the janitor from releasing the pull while the request is alive.
func (t *pullTracker) acquire(ctx context.Context, requestID string, source fwkdl.ID, tokens int) {
	if requestID == "" {
		return
	}
	// Count before the entry is stored, so a release can't run before the count.
	load := t.add(source, tokens)
	if _, existed := t.state.ReadOrWrite(requestID, pullStateKey, &pullEntry{tracker: t, source: source, load: load, tokens: tokens}); existed {
		t.remove(source, load, tokens)
		return
	}
	t.state.BindLiveness(ctx, requestID)
}

// release drops the request's pull, if it holds one. Repeated calls are no-ops.
func (t *pullTracker) release(requestID string) {
	if requestID == "" {
		return
	}
	t.state.Delete(requestID)
}

// add counts a pull of tokens against source and returns the load it was
// counted in, for the pull's release.
func (t *pullTracker) add(source fwkdl.ID, tokens int) *pullLoad {
	t.mu.Lock()
	defer t.mu.Unlock()
	load := t.loads[source]
	if load == nil {
		load = &pullLoad{}
		t.loads[source] = load
	}
	load.pulls++
	load.tokens += tokens
	t.setGauges(source, load)
	return load
}

// remove takes a pull of tokens out of load, the load add counted it in. A
// source whose pull count reaches zero is dropped with its series. When the
// source's endpoint was deleted or replaced after the pull, load is no longer
// the source's, and the source's current load and series are left alone.
func (t *pullTracker) remove(source fwkdl.ID, load *pullLoad, tokens int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.loads[source] != load {
		return
	}
	load.pulls--
	load.tokens -= tokens
	if load.pulls <= 0 {
		t.drop(source)
		return
	}
	t.setGauges(source, load)
}

// endpointAdded records ep as the endpoint behind its ID. An endpoint that
// takes over the ID of a deleted or replaced one starts with no pulls: those
// counted under the ID before are dropped with their series.
func (t *pullTracker) endpointAdded(ep fwkdl.Endpoint) {
	id := ep.GetMetadata().ID
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.endpoints[id] == ep {
		return
	}
	t.endpoints[id] = ep
	t.drop(id)
}

// endpointDeleted drops the pulls counted against ep's ID and their series.
// It returns false and drops nothing when another endpoint has been added
// under the ID since ep, so a delete that arrives after the replacement's add
// leaves the replacement's pulls alone. Like inflightload, it relies on the
// datalayer delivering the same Endpoint for an endpoint's add and delete.
func (t *pullTracker) endpointDeleted(ep fwkdl.Endpoint) bool {
	id := ep.GetMetadata().ID
	t.mu.Lock()
	defer t.mu.Unlock()
	if registered, ok := t.endpoints[id]; ok && registered != ep {
		return false
	}
	delete(t.endpoints, id)
	t.drop(id)
	return true
}

// setGauges writes the source's load to its series. t.mu must be held.
func (t *pullTracker) setGauges(source fwkdl.ID, load *pullLoad) {
	inFlightPulls.WithLabelValues(source.Name, source.Namespace, t.producerName).Set(float64(load.pulls))
	inFlightPullTokens.WithLabelValues(source.Name, source.Namespace, t.producerName).Set(float64(load.tokens))
}

// drop removes the source's load and its series. The release of a pull
// counted in the removed load then changes nothing. t.mu must be held.
func (t *pullTracker) drop(source fwkdl.ID) {
	delete(t.loads, source)
	inFlightPulls.DeleteLabelValues(source.Name, source.Namespace, t.producerName)
	inFlightPullTokens.DeleteLabelValues(source.Name, source.Namespace, t.producerName)
}

// pullEntry is a request's pull. PluginState evicts an entry at most once,
// on release or by the janitor, so its load is returned exactly once.
type pullEntry struct {
	tracker *pullTracker
	source  fwkdl.ID
	// load is the load the pull was counted in, so a release after the
	// source's endpoint was replaced does not reach the replacement's load.
	load   *pullLoad
	tokens int
}

var _ plugin.EvictableStateData = (*pullEntry)(nil)

func (e *pullEntry) Clone() plugin.StateData {
	return &pullEntry{tracker: e.tracker, source: e.source, load: e.load, tokens: e.tokens}
}

func (e *pullEntry) OnEvicted(_ string, _ plugin.StateKey) {
	e.tracker.remove(e.source, e.load, e.tokens)
}

// trackingProducer is the Producer with the hooks that release in-flight
// pulls. PluginFactory returns it only when TrackInFlightPulls is set, so a
// producer without tracking is not called for every response chunk.
type trackingProducer struct {
	*Producer
	// ttft records TTFT by decision from the same response hook. Nil unless
	// ttftByDecision is set.
	ttft *ttftRecorder
}

var (
	_ fwkrh.DataProducer          = &trackingProducer{}
	_ fwkrh.PreRequest            = &trackingProducer{}
	_ fwkrh.ResponseBodyProcessor = &trackingProducer{}
	_ fwkrh.RequestAbandoned      = &trackingProducer{}
	_ fwkdl.Registrant            = &trackingProducer{}
	_ fwkdl.EndpointExtractor     = &trackingProducer{}
)

// RegisterDependencies subscribes the producer to endpoint lifecycle events,
// adding an endpoint-notification-source when the config has none.
func (p *trackingProducer) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      p.TypedName(),
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType,
		),
	})
}

// Extract drops a source's pulls and their series when its endpoint is
// deleted, so a request still holding a pull from it keeps no series alive
// and an endpoint that later takes the same ID starts at zero.
func (p *trackingProducer) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}
	switch event.Type {
	case fwkdl.EventAddOrUpdate:
		p.pulls.endpointAdded(event.Endpoint)
	case fwkdl.EventDelete:
		if !p.pulls.endpointDeleted(event.Endpoint) {
			log.FromContext(ctx).WithName(p.typedName.String()).V(logging.DEFAULT).Info(
				"Ignoring stale delete for replaced endpoint", "endpoint", event.Endpoint.GetMetadata().ID)
		}
	}
	return nil
}

// ResponseBody releases the request's pull at its first response chunk, by
// which time the computing pod has prefilled and the pull is done. The
// end-of-stream call, which the director also makes for a request that fails
// after scheduling, releases a pull that no earlier chunk did. With
// ttftByDecision set, it then passes the chunk to the TTFT recorder. The two
// keep separate per-request state, so the order changes no result; releasing
// first keeps the release independent of the recorder's early returns.
func (p *trackingProducer) ResponseBody(ctx context.Context, request *scheduling.InferenceRequest, response *fwkrh.Response, md *fwkdl.EndpointMetadata) {
	if request == nil || response == nil {
		return
	}
	if response.StartOfStream || response.EndOfStream {
		p.pulls.release(request.RequestID)
	}
	if p.ttft != nil {
		p.ttft.ResponseBody(ctx, request, response, md)
	}
}

// RequestAbandoned releases the request's pull when the director drops the
// request before sending it, for example after another PreRequest plugin
// fails.
func (p *trackingProducer) RequestAbandoned(_ context.Context, request *scheduling.InferenceRequest) {
	if request == nil {
		return
	}
	p.pulls.release(request.RequestID)
}
