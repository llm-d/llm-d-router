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

package groupsched

// THE REQUEST SIDE, IN THE ORDER THE ROUTER CALLS IT:
//
//	Pick          the dispatch decision (flow control)
//	Filter        the candidate set, narrowed to the seat engine (scheduling)
//	PreRequest    the request left the scheduler for a pod: the record, the observed engine
//	ResponseBody  the first chunk is prefill completion; the end of the stream settles
//	RequestAbandoned  the request left the path without reaching the stream
//
// The declarations the framework reads at setup (Produces/Produce/NewState) follow the hooks, and
// then the helpers every one of them shares.

import (
	"context"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// Pick is the dispatch decision.
func (r *GroupScheduler) Pick(ctx context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	now := r.now()
	var queues []fwkfc.FlowQueueAccessor
	var heads []Head
	band.IterateQueues(func(q fwkfc.FlowQueueAccessor) bool {
		if q == nil || q.Len() == 0 {
			return true
		}
		h := q.Peek()
		if h == nil {
			return true
		}
		thread := ""
		reqID, fid, wire := "", "", ""
		returner := false
		var chains []Chain
		if req := h.OriginalRequest(); req != nil {
			if ir := req.InferenceRequest(); ir != nil {
				thread = r.threadOf(ir)
				reqID = ir.RequestID
				fid = ir.FairnessID // the queue key (the thread, unless a header named one)
				wire = wireID(ir)   // the client's own id — the join to its request record
				returner = isReturner(ir)
				chains = chargeChains(ir)
			}
		}
		queues = append(queues, q)
		heads = append(heads, Head{Thread: thread, Returner: returner,
			FairnessID: fid, WireID: wire, ReqID: reqID, Arrival: h.EnqueueTime(), Chains: chains,
		})
		return true
	})
	if len(heads) == 0 {
		return nil, nil //nolint:nilnil
	}
	// before the measured capacity arrives: FCFS pass-through, logged once per process and counted
	// on dispatches_total under engine="nocapacity" (the label that belongs to no engine)
	if !r.sched.HasCapacity() {
		r.fcfsOnce.Do(func() {
			r.logger.V(logutil.DEFAULT).Info("group-scheduler: num_gpu_blocks not yet measured; serving FCFS until the endpoint metrics arrive")
		})
		r.sched.FCFSServed()
		r.metrics.ObserveServed(nocapacityEngine)
		if i := FCFSPick(heads); i >= 0 {
			return queues[i], nil
		}
		return nil, nil //nolint:nilnil
	}
	if i := r.sched.Pass(heads, now); i >= 0 {
		return queues[i], nil
	}
	return nil, nil //nolint:nilnil
}

// Filter narrows the candidate endpoints to the thread's CURRENT SEAT ENGINE.
func (r *GroupScheduler) Filter(_ context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	if len(pods) <= 1 {
		return pods
	}
	if request == nil {
		return pods
	}
	engine, seated := r.sched.SeatEngineOfRequest(request.RequestID)
	if !seated {
		return pods
	}
	out := make([]fwksched.Endpoint, 0, 1)
	for _, p := range pods {
		if p != nil && endpointKey(p.GetMetadata()) == engine {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		// the seat engine is not a candidate: pass through and let the observed endpoint win
		// (PreRequest rebinds + counts the violation).
		r.logger.V(logutil.DEBUG).Info("group-scheduler: FILTER PASS-THROUGH — the seat engine is not a candidate; passing every candidate through (the observed endpoint wins the rebind)",
			"seatedOn", engineLabel(engine), "candidates", len(pods))
		return pods
	}
	return out
}

// PreRequest runs after the request leaves the queue (dispatch).
func (r *GroupScheduler) PreRequest(_ context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	// the engine the framework chose is authoritative, so it is recorded even for a request
	// nothing tracks (anonymous traffic): the check below must not depend on trackability.
	engine := ""
	if endpoint := pickedEndpoint(schedulingResult); endpoint != nil {
		r.rememberEndpoint(endpoint.GetMetadata())
		engine = endpointKey(endpoint.GetMetadata())
		// PreRequest only checks the block size; capacity is learned from the live candidates.
		if err := r.checkBlockSize(engine, endpoint.GetMetrics()); err != nil {
			return err
		}
	}
	if request == nil {
		return nil
	}
	// The engine here is AUTHORITATIVE: the seat was placed on the thread's engine at Pick, and
	// the framework's actual choice moves it if it disagrees.
	r.sched.AtPod(request.RequestID, engine, r.now())
	return nil
}

// ResponseBody: the first chunk is prefill complete (the cold-entry charge is released there);
// the end of stream is the finalizer for every termination cause.
func (r *GroupScheduler) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, target *fwkdl.EndpointMetadata) {
	if request == nil || response == nil {
		return
	}
	if response.StartOfStream {
		r.sched.FirstChunk(request.RequestID, r.now())
	}
	if !response.EndOfStream {
		return
	}
	// The settle's engine comes from the response's own endpoint metadata: a settle with no
	// endpoint would fold this engine's keys into another's record.
	engine := ""
	if target != nil && target.ID.String() != "" {
		engine = target.ID.String()
	}
	r.sched.Settle(request.RequestID, engine, r.now())
}

// RequestAbandoned ends the record of a request that left the request path without reaching
// PreRequest.
func (r *GroupScheduler) RequestAbandoned(_ context.Context, request *fwksched.InferenceRequest) {
	if request == nil {
		return
	}
	r.sched.Abandon(request.RequestID, r.now())
}

// Produces declares no request attribute: the plugin publishes NOTHING onto the request. (Not
// declaring ConsumerPlugin says the same about what it reads — see datalayer.buildScopeSpec.)
func (r *GroupScheduler) Produces() map[fwkplugin.DataKey]any { return nil }

// Produce implements DataProducer ONLY to receive RequestAbandoned: the framework has the event but
// delivers it to data producers, so the plugin has to be one to hear it.
func (r *GroupScheduler) Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return nil
}

// NewState satisfies the FairnessPolicy contract.
func (r *GroupScheduler) NewState(_ context.Context) any { return nil }

// chargeChains is the request's EXACT prefill chains — what the groups are charged — as thread
// identity published them at arrival. Reading the attribute is the whole cost: the blocks were
// hashed once, when the thread was derived from the same tokens.
func chargeChains(request *fwksched.InferenceRequest) []Chain {
	if request == nil {
		return nil
	}
	chains, _ := fwksched.ReadRequestAttribute[[]Chain](request, threadidentity.ThreadBlocksKey)
	return chains
}

// wireIDHeaderNames are the client's own request-identity headers, in priority order: the
// correlation id first, then the standard request id.
var wireIDHeaderNames = []string{"x-correlation-id", reqcommon.RequestIDHeaderKey}

// wireID is the client's own id for the request ("" when it sent neither header).
func wireID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	for _, name := range wireIDHeaderNames {
		if v, ok := metadata.GetLowerCaseHeaderValue(request.Headers, name); ok && v != "" {
			return v
		}
	}
	return ""
}

// readID is the string attribute at key ("" when it is absent).
func readID(request *fwksched.InferenceRequest, key fwkplugin.DataKey) string {
	if request == nil {
		return ""
	}
	id, _ := fwksched.ReadRequestAttribute[string](request, key)
	return id
}

// threadOf is the request's thread id — the queue key, read from thread identity.
func (r *GroupScheduler) threadOf(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	return readID(request, threadidentity.ThreadIdentityKey)
}

// isReturner reports whether the request is NOT its thread's first turn — the fairness class
// the entry order sorts on.
func isReturner(request *fwksched.InferenceRequest) bool {
	if request == nil {
		return false
	}
	rec, ok := fwksched.ReadRequestAttribute[threadidentity.ThreadArrival](request, threadidentity.ThreadArrivalKey)
	if !ok {
		return false
	}
	return rec.Turn > 1
}

// pickedEndpoint is the endpoint the framework placed the request on, for the hooks that are handed a
// scheduling result instead of a target.
func pickedEndpoint(schedulingResult *fwksched.SchedulingResult) fwksched.Endpoint {
	if schedulingResult == nil {
		return nil
	}
	result, ok := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if !ok || result == nil || len(result.TargetEndpoints) == 0 {
		return nil
	}
	return result.TargetEndpoints[0]
}

// nocapacityEngine is the dispatches_total label for a pre-capacity pass-through dispatch.
const nocapacityEngine = "nocapacity"
