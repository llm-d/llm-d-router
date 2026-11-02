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

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

var inFlightPulls = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Subsystem: metricsutil.LLMDRouterEndpointPickerSubsystem,
		Name:      "p2p_inflight_pulls",
		Help:      metricsutil.HelpMsgWithStability("Current number of P2P KV cache pulls directed at each source endpoint, as tracked by the p2p-source-producer.", compbasemetrics.ALPHA),
	},
	[]string{"endpoint_name", "namespace", "producer_name"},
)

// registerPullMetrics registers the in-flight pull gauge. The gauge is a
// package-level vector, so a second producer instance registering it is a
// no-op; producer_name keeps each instance's series distinct.
func registerPullMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("in-flight pull metrics registerer is required")
	}
	if err := registerer.Register(inFlightPulls); err != nil {
		var alreadyRegistered prometheus.AlreadyRegisteredError
		if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == inFlightPulls {
			return nil
		}
		return fmt.Errorf("register in-flight pull metric: %w", err)
	}
	return nil
}

const pullStateKey plugin.StateKey = "inflight-pull"

// pullTracker counts the pulls one producer has directed at each source
// endpoint and not yet released. Each request holds at most one pull, kept
// in PluginState so the janitor releases it if no response or abandonment
// ever arrives.
type pullTracker struct {
	producerName string
	state        *plugin.PluginState

	mu     sync.Mutex
	counts map[fwkdl.ID]int
}

func newPullTracker(ctx context.Context, producerName string) *pullTracker {
	return &pullTracker{
		producerName: producerName,
		state:        plugin.NewPluginState(ctx),
		counts:       map[fwkdl.ID]int{},
	}
}

// acquire counts a pull from source for the request. A second acquire for
// the same request is ignored. ctx keeps the janitor from releasing the pull
// while the request is alive.
func (t *pullTracker) acquire(ctx context.Context, requestID string, source fwkdl.ID) {
	if requestID == "" {
		return
	}
	// Count before the entry is stored, so a release can't run before the count.
	t.add(source, 1)
	if _, existed := t.state.ReadOrWrite(requestID, pullStateKey, &pullEntry{tracker: t, source: source}); existed {
		t.add(source, -1)
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

// add changes the source's count and its gauge series together. A source
// whose count reaches zero is dropped with its series, so an endpoint that
// leaves the pool leaves no series behind once its pulls are released.
func (t *pullTracker) add(source fwkdl.ID, delta int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	n := t.counts[source] + delta
	if n <= 0 {
		delete(t.counts, source)
		inFlightPulls.DeleteLabelValues(source.Name, source.Namespace, t.producerName)
		return
	}
	t.counts[source] = n
	inFlightPulls.WithLabelValues(source.Name, source.Namespace, t.producerName).Set(float64(n))
}

// pullEntry is a request's pull. PluginState evicts an entry at most once,
// on release or by the janitor, so the count is returned exactly once.
type pullEntry struct {
	tracker *pullTracker
	source  fwkdl.ID
}

var _ plugin.EvictableStateData = (*pullEntry)(nil)

func (e *pullEntry) Clone() plugin.StateData {
	return &pullEntry{tracker: e.tracker, source: e.source}
}

func (e *pullEntry) OnEvicted(_ string, _ plugin.StateKey) {
	e.tracker.add(e.source, -1)
}

// trackingProducer is the Producer with the hooks that release in-flight
// pulls. PluginFactory returns it only when TrackInFlightPulls is set, so a
// producer without tracking is not called for every response chunk.
type trackingProducer struct {
	*Producer
}

var (
	_ requestcontrol.DataProducer          = &trackingProducer{}
	_ requestcontrol.PreRequest            = &trackingProducer{}
	_ requestcontrol.ResponseBodyProcessor = &trackingProducer{}
	_ requestcontrol.RequestAbandoned      = &trackingProducer{}
)

// ResponseBody releases the request's pull at its first response chunk, by
// which time the computing pod has prefilled and the pull is done. The
// end-of-stream call, which the director also makes for a request that fails
// after scheduling, releases a pull that no earlier chunk did.
func (p *trackingProducer) ResponseBody(_ context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil {
		return
	}
	if response.StartOfStream || response.EndOfStream {
		p.pulls.release(request.RequestID)
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
