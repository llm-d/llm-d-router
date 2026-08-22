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

package latencyobserver

import (
	"context"
	"maps"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

var (
	_ requestcontrol.DataProducer          = &Observer{}
	_ requestcontrol.PreRequest            = &Observer{}
	_ requestcontrol.ResponseBodyProcessor = &Observer{}
)

// dispatchInfo links a dispatch to the response that answers it.
type dispatchInfo struct {
	endpointID   string
	inflight     int64
	dispatchedAt time.Time
}

// Clone implements [fwkplugin.StateData].
func (d *dispatchInfo) Clone() fwkplugin.StateData {
	if d == nil {
		return nil
	}
	cp := *d
	return &cp
}

// inflightAtDispatch is what each candidate was carrying when Produce ran, keyed
// by endpoint ID. It lives in PluginState, not on the endpoint: the data-parallel
// and disaggregated profile handlers rebuild endpoints with empty attribute maps.
type inflightAtDispatch map[string]int64

// Clone implements [fwkplugin.StateData].
func (m inflightAtDispatch) Clone() fwkplugin.StateData {
	cp := make(inflightAtDispatch, len(m))
	maps.Copy(cp, m)
	return cp
}

// Produces declares the percentile snapshot this producer publishes.
func (p *Observer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.percentilesDataKey: attrlatency.TTFTPercentiles{}}
}

// Consumes declares the in-flight load as Required, so the DAG runs
// inflight-load-producer first and auto-creates it when the config omits it.
func (p *Observer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{p.inFlightLoadDataKey: attrconcurrency.InFlightLoad{}},
	}
}

// Produce pins every candidate's in-flight load, so PreRequest sees the load
// carried before this request landed. Reading it there instead would race:
// inflight-load-producer increments its counters in its own PreRequest, and hook
// order between plugins is undefined. Produce is DAG-ordered.
func (p *Observer) Produce(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	if request == nil || request.RequestID == "" || p.PluginState == nil {
		return nil
	}
	pinned := make(inflightAtDispatch, len(endpoints))
	for _, endpoint := range endpoints {
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		pinned[endpoint.GetMetadata().ID.String()] = readInFlightRequests(endpoint, p.inFlightLoadDataKey)
	}
	p.PluginState.Write(request.RequestID, inflightStateKey, pinned)
	return nil
}

// readInFlightRequests returns the endpoint's request count under key, or zero
// when the attribute is absent or malformed.
func readInFlightRequests(endpoint fwksched.Endpoint, key fwkplugin.DataKey) int64 {
	if raw, ok := endpoint.Get(key); ok {
		if load, ok := raw.(*attrconcurrency.InFlightLoad); ok && load != nil {
			return load.Requests
		}
	}
	return 0
}

// PreRequest records the dispatch — which endpoint won, what it was carrying,
// when the request left — for ResponseBody to turn into a TTFT.
//
// Always returns nil: a returned error fails the request, and failing to record
// an observation is never a reason to reject one.
func (p *Observer) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	endpoint := primaryTarget(result)
	if request == nil || request.RequestID == "" || endpoint == nil {
		log.FromContext(ctx).V(logutil.DEBUG).Info("Skipping TTFT tracking: no request ID or no primary target")
		return nil
	}
	endpointID := endpoint.GetMetadata().ID.String()
	p.PluginState.Write(request.RequestID, dispatchStateKey, &dispatchInfo{
		endpointID:   endpointID,
		inflight:     p.pinnedInflight(request.RequestID, endpointID),
		dispatchedAt: time.Now(),
	})
	return nil
}

// pinnedInflight returns what Produce recorded, or zero if it never ran.
func (p *Observer) pinnedInflight(requestID, endpointID string) int64 {
	pinned, err := fwkplugin.ReadPluginStateKey[inflightAtDispatch](p.PluginState, requestID, inflightStateKey)
	if err != nil {
		return 0
	}
	return pinned[endpointID]
}

// primaryTarget returns the endpoint the primary profile selected, or nil. TTFT
// belongs to whichever endpoint produced the first token.
func primaryTarget(result *fwksched.SchedulingResult) fwksched.Endpoint {
	if result == nil {
		return nil
	}
	primary := result.ProfileResults[result.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return nil
	}
	if endpoint := primary.TargetEndpoints[0]; endpoint != nil && endpoint.GetMetadata() != nil {
		return endpoint
	}
	return nil
}

// ResponseBody turns the first response chunk into a TTFT observation and
// releases the dispatch record when the stream ends. The first chunk of a
// stream is not the final chunk, so this runs on the director's async
// response-body queue rather than in front of Envoy.
func (p *Observer) ResponseBody(ctx context.Context, request *fwksched.InferenceRequest,
	response *requestcontrol.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || request.RequestID == "" || p.PluginState == nil {
		return
	}
	if response.StartOfStream {
		p.observeFirstChunk(ctx, request.RequestID, response.EndOfStream)
	}
	if response.EndOfStream {
		p.PluginState.Delete(request.RequestID)
	}
}

// observeFirstChunk records the TTFT for the request's dispatch record.
//
// A response whose first chunk is also its last is discarded: its latency is
// end-to-end, which includes decode and sits far above the real TTFT, so
// recording it would corrupt both the floor and the load anchors.
func (p *Observer) observeFirstChunk(ctx context.Context, requestID string, endOfStream bool) {
	if endOfStream {
		return
	}
	dispatch, err := fwkplugin.ReadPluginStateKey[*dispatchInfo](p.PluginState, requestID, dispatchStateKey)
	if err != nil {
		// PreRequest was skipped, or the janitor already reaped the request.
		return
	}

	now := time.Now()
	ttft := now.Sub(dispatch.dispatchedAt).Seconds()
	p.record(dispatch.endpointID, ttft, dispatch.inflight, now)

	if debugLogger := log.FromContext(ctx).V(logutil.DEBUG); debugLogger.Enabled() {
		debugLogger.Info("ttft-observation", "requestID", requestID, "endpoint", dispatch.endpointID,
			"ttftSeconds", ttft, "inflightAtDispatch", dispatch.inflight)
	}
}
