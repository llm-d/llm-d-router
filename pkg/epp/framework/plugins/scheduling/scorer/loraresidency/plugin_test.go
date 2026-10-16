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

package loraresidency

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraresidency/lorascore/residency"
)

var ctx = context.Background()

// rig drives the plugin the way the framework does. The runtime announces each
// endpoint object with an add event before the datastore stores it, and the
// director scores each request on a candidate snapshot cloned from those
// objects and passes the picked snapshot endpoint to PreRequest. Score sees the
// snapshot through the scheduler's data scope, which hides every key the
// plugin does not declare.
type rig struct {
	t     *testing.T
	p     *Plugin
	order []string
	cur   map[string]fwkdl.Endpoint // each pod's current datalayer object
	gone  map[string]bool           // deleted, so absent from fresh snapshots
	held  map[string]map[string]fwksched.Endpoint
	now   time.Time
	seq   int
}

// newRig builds a plugin over pods with the given max_loras and no activity,
// announcing each pod as the runtime does.
func newRig(t *testing.T, params Parameters, maxLoras int, pods ...string) *rig {
	t.Helper()
	if params.BaseModels == nil {
		params.BaseModels = []string{"base"}
	}
	p, err := New("test", params)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	r := &rig{
		t: t, p: p, now: time.Unix(1_000_000, 0),
		cur: map[string]fwkdl.Endpoint{}, gone: map[string]bool{},
		held: map[string]map[string]fwksched.Endpoint{},
	}
	p.now = func() time.Time { return r.now }
	for _, name := range pods {
		obj := fwkdl.NewEndpoint(
			&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "ns", Name: name}},
			&fwkdl.Metrics{ActiveModels: map[string]int{}, WaitingModels: map[string]int{}, MaxActiveModels: maxLoras})
		r.order = append(r.order, name)
		r.cur[name] = obj
		r.extract(fwkdl.EventAddOrUpdate, obj)
	}
	return r
}

func (r *rig) id(pod string) string { return "ns/" + pod }

func (r *rig) extract(eventType fwkdl.EventType, ep fwkdl.Endpoint) {
	r.t.Helper()
	if err := r.p.Extract(ctx, fwkdl.EndpointEvent{Type: eventType, Endpoint: ep}); err != nil {
		r.t.Fatalf("Extract: %v", err)
	}
}

// metrics is pod's live metrics, shared by every object for that pod.
func (r *rig) metrics(pod string) *fwkdl.Metrics { return r.cur[pod].GetMetrics() }

// fresh is a new endpoint object for pod's ID, as NewEndpoint builds for an
// add; it is not announced.
func (r *rig) fresh(pod string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(r.cur[pod].GetMetadata(), r.cur[pod].GetMetrics())
}

// replace announces a new object for pod and returns the one it replaced.
func (r *rig) replace(pod string) fwkdl.Endpoint {
	old := r.cur[pod]
	obj := r.fresh(pod)
	r.extract(fwkdl.EventAddOrUpdate, obj)
	r.cur[pod] = obj
	delete(r.gone, pod)
	return old
}

// update re-announces pod's current object, as UpdateEndpoint does.
func (r *rig) update(pod string) { r.extract(fwkdl.EventAddOrUpdate, r.cur[pod]) }

// del announces the deletion of pod's current object.
func (r *rig) del(pod string) {
	r.extract(fwkdl.EventDelete, r.cur[pod])
	r.gone[pod] = true
}

// snapshot clones the live pods' current objects as the director does.
func (r *rig) snapshot() (map[string]fwksched.Endpoint, []fwksched.Endpoint) {
	byPod := map[string]fwksched.Endpoint{}
	var list []fwksched.Endpoint
	for _, name := range r.order {
		if r.gone[name] {
			continue
		}
		o := r.cur[name]
		ep := fwksched.NewEndpoint(o.GetMetadata(), o.GetMetrics(), o.GetAttributes())
		byPod[name] = ep
		list = append(list, ep)
	}
	return byPod, list
}

// runScore calls Score as the scheduler does: on endpoints scoped to the
// plugin's declarations, unwrapping each key of the scores it returns.
func (r *rig) runScore(req *fwksched.InferenceRequest, list []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{r.p})
	scopedReq, scoped, _ := datalayer.ScopeInvocation(logr.Discard(), "Scorer", r.p, req, list)
	out := map[fwksched.Endpoint]float64{}
	for ep, v := range r.p.Score(ctx, scopedReq, scoped) {
		out[datalayer.UnscopeEndpoint(ep)] = v
	}
	return out
}

func (r *rig) score(model string) map[string]float64 {
	_, list := r.snapshot()
	out := map[string]float64{}
	for ep, v := range r.runScore(&fwksched.InferenceRequest{TargetModel: model}, list) {
		out[ep.GetMetadata().ID.Name] = v
	}
	return out
}

// scoreOn scores a new request on the given snapshot and holds the snapshot
// for its PreRequest.
func (r *rig) scoreOn(model string, snap map[string]fwksched.Endpoint) *fwksched.InferenceRequest {
	r.seq++
	req := &fwksched.InferenceRequest{RequestID: string(rune('A' + r.seq)), TargetModel: model}
	var list []fwksched.Endpoint
	for _, name := range r.order {
		if ep, ok := snap[name]; ok {
			list = append(list, ep)
		}
	}
	r.runScore(req, list)
	r.held[req.RequestID] = snap
	return req
}

// scoreReq scores a new request on a fresh snapshot.
func (r *rig) scoreReq(model string) *fwksched.InferenceRequest {
	snap, _ := r.snapshot()
	return r.scoreOn(model, snap)
}

// preRequest delivers a scheduling result picking pod from the snapshot req
// was scored on.
func (r *rig) preRequest(req *fwksched.InferenceRequest, pod string) {
	r.t.Helper()
	ep, ok := r.held[req.RequestID][pod]
	if !ok {
		r.t.Fatalf("pod %s is not in request %s's snapshot", pod, req.RequestID)
	}
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{ep}},
		},
	}
	req.SchedulingResult = result
	if err := r.p.PreRequest(ctx, req, result); err != nil {
		r.t.Fatalf("PreRequest: %v", err)
	}
}

// route runs Score and PreRequest for a request sent to pod, and returns it.
func (r *rig) route(model, pod string) *fwksched.InferenceRequest {
	req := r.scoreReq(model)
	r.preRequest(req, pod)
	return req
}

// header delivers response headers as the EPP does: Envoy's header keys
// copied verbatim, so the status is under the ":status" pseudo-header.
func (r *rig) header(req *fwksched.InferenceRequest, status string) {
	r.headerKey(req, ":status", status)
}

// headerKey delivers response headers with the status under key.
func (r *rig) headerKey(req *fwksched.InferenceRequest, key, status string) {
	r.p.ResponseHeader(ctx, req, &fwkrc.Response{RequestID: req.RequestID, Headers: map[string]string{key: status}}, nil)
}

// chunk delivers a streamed chunk that does not end the stream.
func (r *rig) chunk(req *fwksched.InferenceRequest, start bool) {
	r.p.ResponseBody(ctx, req, &fwkrc.Response{RequestID: req.RequestID, StartOfStream: start}, nil)
}

// end delivers the end-of-stream call. As Director.HandleResponseBody builds
// it, StartOfStream is true when no body call came before, and cause is
// natural unless the stream was cut short.
func (r *rig) end(req *fwksched.InferenceRequest, start bool, cause fwkrc.TerminationCause) {
	r.p.ResponseBody(ctx, req, &fwkrc.Response{RequestID: req.RequestID, StartOfStream: start, EndOfStream: true, TerminationCause: cause}, nil)
}

// succeed delivers a successful streamed response for req.
func (r *rig) succeed(req *fwksched.InferenceRequest) {
	r.header(req, "200")
	r.chunk(req, true)
	r.end(req, false, fwkrc.TerminationCauseNatural)
}

// serve runs one successful streamed request to completion.
func (r *rig) serve(model, pod string) {
	req := r.route(model, pod)
	r.header(req, "200")
	r.chunk(req, true)
	r.chunk(req, false)
	r.now = r.now.Add(time.Second)
	r.end(req, false, fwkrc.TerminationCauseNatural)
}

func TestNewValidates(t *testing.T) {
	neg := -1.0
	for name, p := range map[string]Parameters{
		"no base models":     {},
		"bad half-life":      {BaseModels: []string{"b"}, DemandHalfLife: "soon"},
		"zero half-life":     {BaseModels: []string{"b"}, DemandHalfLife: "0s"},
		"negative weight":    {BaseModels: []string{"b"}, CopyCost: &neg},
		"negative cpu loras": {BaseModels: []string{"b"}, MaxCPULoras: -1},
	} {
		if _, err := New("x", p); err == nil {
			t.Errorf("%s: New succeeded, want an error", name)
		}
	}
}

func TestFactoryDecodesAndRejectsUnknownFields(t *testing.T) {
	good := json.RawMessage(`{"baseModels":["base"],"maxCpuLoras":4,"demandHalfLife":"30s","evictionWeight":0.5}`)
	pl, err := Factory("n", fwkplugin.StrictDecoder(good), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	p := pl.(*Plugin)
	if p.weights.EvictionWeight != 0.5 || p.weights.CopyCost != 0.2 {
		t.Fatalf("weights = %+v, want eviction 0.5 and default copy 0.2", p.weights)
	}
	if got := p.TypedName(); got.Type != Type || got.Name != "n" {
		t.Fatalf("TypedName = %+v", got)
	}
	bad := json.RawMessage(`{"baseModels":["base"],"maxCpuLora":4}`)
	if _, err := Factory("n", fwkplugin.StrictDecoder(bad), nil); err == nil {
		t.Fatal("Factory accepted a misspelled field")
	}
	if _, err := Factory("n", nil, nil); err == nil {
		t.Fatal("Factory with no parameters must fail: baseModels is required")
	}
}

func TestBaseModelRequestIsNeutralAndUntracked(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0", "p1")
	r.serve("a", "p0")
	if s := r.score("base"); s["p0"] != s["p1"] {
		t.Fatalf("base-model request must score pods equally, got %v", s)
	}
	req := r.route("base", "p1")
	r.end(req, true, fwkrc.TerminationCauseNatural)
	if got := r.p.model.Tier(r.id("p1"), "base"); got != residency.Absent {
		t.Fatalf("base model tracked as an adapter: Tier = %v", got)
	}
}

func TestServedAdapterAttractsItsNextRequest(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0", "p1")
	r.serve("a", "p0")
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.GPU {
		t.Fatalf("Tier(p0, a) = %v, want GPU after a served request", got)
	}
	if s := r.score("a"); !(s["p0"] > s["p1"]) {
		t.Fatalf("want p0 preferred for a, got %v", s)
	}
	if len(r.p.requests) != 0 {
		t.Fatalf("request state leaked: %v", r.p.requests)
	}
}

// Started comes from the first body chunk, not from routing or headers.
func TestRoutedOnlyIsPendingNotResident(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0", "p1")
	req := r.route("a", "p0")
	r.header(req, "200")
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent before the first chunk", got)
	}
	if s := r.score("a"); !(s["p0"] > s["p1"]) {
		t.Fatalf("pending pod should attract the burst, got %v", s)
	}
	r.chunk(req, true)
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU after the first chunk", got)
	}
}

func TestErrorStatusIsNotExecution(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	req := r.route("a", "p0")
	r.header(req, "503")
	r.end(req, true, fwkrc.TerminationCauseNatural) // an error delivered in full is natural
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent after an error response", got)
	}
	if len(r.p.requests) != 0 {
		t.Fatalf("request state leaked: %v", r.p.requests)
	}
}

// A plain "status" key, which the EPP's own check also accepts, counts as an
// error status too.
func TestPlainStatusKeyErrorIsNotExecution(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	req := r.route("a", "p0")
	r.headerKey(req, "status", "404")
	r.end(req, true, fwkrc.TerminationCauseNatural)
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent after an error response", got)
	}
}

// When a scheduled request exits before any body, the server's deferred
// cleanup makes the director send one call with StartOfStream AND EndOfStream set and a non-natural
// cause. That is not evidence of execution.
func TestSyntheticAbortCleanupLeavesNoTrace(t *testing.T) {
	for _, cause := range []fwkrc.TerminationCause{
		fwkrc.TerminationCauseClientDisconnect, fwkrc.TerminationCauseError, fwkrc.TerminationCauseEvicted,
	} {
		r := newRig(t, Parameters{}, 1, "p0")
		req := r.route("a", "p0")
		r.header(req, "200")
		r.end(req, true, cause)
		if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.Absent {
			t.Fatalf("%s: Tier = %v, want Absent after the cleanup call", cause, got)
		}
		if got := r.p.model.Admit(r.id("p0"), "a"); got.Pending {
			t.Fatalf("%s: Admit = %+v, want no pending request after the cleanup", cause, got)
		}
		if len(r.p.requests) != 0 {
			t.Fatalf("%s: request state leaked: %v", cause, r.p.requests)
		}
	}
}

// A non-streaming success is one call with both flags and a natural cause:
// that is execution.
func TestNaturalOneShotResponseIsExecution(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	req := r.route("a", "p0")
	r.header(req, "200")
	r.end(req, true, fwkrc.TerminationCauseNatural)
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU after a complete one-shot response", got)
	}
}

// A stream cut short after its first chunk did execute.
func TestDisconnectAfterFirstChunkStillExecuted(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	req := r.route("a", "p0")
	r.header(req, "200")
	r.chunk(req, true)
	r.end(req, false, fwkrc.TerminationCauseClientDisconnect)
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU: the adapter ran before the disconnect", got)
	}
}

// An error status in the framework's header map must stop the first chunk from counting, on the streaming path too.
func TestStreamedErrorStatusIsNotExecution(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	req := r.route("a", "p0")
	r.header(req, "500")
	r.chunk(req, true)
	r.end(req, false, fwkrc.TerminationCauseNatural)
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent after a 500 response", got)
	}
}

func TestDuplicatePreRequestCountedOnce(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	req := r.route("a", "p0")
	if err := r.p.PreRequest(ctx, req, req.SchedulingResult); err != nil {
		t.Fatal(err)
	}
	r.end(req, true, fwkrc.TerminationCauseClientDisconnect) // one abort clears the one pending route
	if got := r.p.model.Admit(r.id("p0"), "a"); got.Pending {
		t.Fatalf("Admit = %+v, want the duplicate PreRequest ignored", got)
	}
}

// ActiveModels feeds ObserveActive: an adapter the pod reports active is
// protected, so the pod counts as blocked rather than offering it as a victim.
func TestActiveModelsProtectResidentAdapter(t *testing.T) {
	blocked, noEviction := 10.0, 0.0
	r := newRig(t, Parameters{BlockedCost: &blocked, EvictionWeight: &noEviction}, 1, "p0", "p1")
	r.serve("a", "p0")
	r.serve("x", "p1")
	if s := r.score("b"); s["p0"] != s["p1"] {
		t.Fatalf("both pods offer an idle victim, want equal, got %v", s)
	}
	r.metrics("p0").ActiveModels["a"] = 1
	if s := r.score("b"); !(s["p1"] > s["p0"]) {
		t.Fatalf("p0's only slot is reported active, want p1 preferred, got %v", s)
	}
}

func TestCapacityComesFromMaxActiveModels(t *testing.T) {
	r := newRig(t, Parameters{}, 2, "p0")
	r.serve("a", "p0")
	r.serve("b", "p0")
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.GPU {
		t.Fatalf("Tier(a) = %v, want GPU with max_loras 2", got)
	}
}

func TestEndpointDeleteForgetsPod(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	r.serve("a", "p0")
	r.del("p0")
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent after the endpoint is deleted", got)
	}
	if got := len(r.p.endpoints) + len(r.p.genSeq); got != 0 {
		t.Fatalf("endpoint/generation records = %d after delete, want 0 (pods that never return must not leak)", got)
	}
}

// An update re-dispatches the same endpoint object (Runtime.UpdateEndpoint),
// so it must keep the pod's state.
func TestEndpointUpdateKeepsPod(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	r.serve("a", "p0")
	r.update("p0")
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU kept across an update of the same endpoint", got)
	}
	r.serve("b", "p0") // a snapshot after the update is still the current generation
	if got := r.p.model.Tier(r.id("p0"), "b"); got != residency.GPU {
		t.Fatalf("Tier(b) = %v, want GPU: an update must not stale later snapshots", got)
	}
}

// Every announced object carries its generation into the director's snapshot.
func TestAddStampsGenerationIntoSnapshots(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	snap, _ := r.snapshot()
	v, ok := snap["p0"].Get(generationKey)
	if !ok {
		t.Fatal("snapshot endpoint carries no generation stamp")
	}
	if g := uint64(v.(generation)); g != r.p.genSeq[r.id("p0")] {
		t.Fatalf("stamp %d, want the pod's current generation %d", g, r.p.genSeq[r.id("p0")])
	}
}

// The stamp must stay readable through the scorer's data scope, and declaring
// it must pass the EPP's startup validation. Extract writes the stamp, so it is
// declared in Produces(); listing it in Consumes() as well would draw the
// "optional data key has no producer" warning at startup.
func TestGenerationKeyDeclaredAndStartupValid(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	_, list := r.snapshot()
	scoped, _ := datalayer.Scope(logr.Discard(), "Scorer", r.p, list)
	if _, ok := scoped[0].Get(generationKey); !ok {
		t.Fatal("generation stamp hidden by the data scope; declare it in Produces()")
	}
	deps := r.p.Consumes()
	if _, ok := deps.Required[generationKey]; ok {
		t.Fatal("generation stamp listed in Consumes().Required")
	}
	if _, ok := deps.Optional[generationKey]; ok {
		t.Fatal("generation stamp listed in Consumes().Optional")
	}
	// The core metrics extractor is the producer the EPP runs beside the plugin.
	ext, err := metrics.NewCoreMetricsExtractor(metrics.NewMappingRegistry(), "")
	if err != nil {
		t.Fatalf("NewCoreMetricsExtractor: %v", err)
	}
	plugins := []fwkplugin.Plugin{ext, r.p}
	reg, err := fwkdl.BuildRegistry(plugins)
	if err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	if !reg.Has(generationKey) {
		t.Fatal("registry does not know the generation stamp")
	}
	if _, err := datalayer.ValidateAndOrderDataDependencies(plugins); err != nil {
		t.Fatalf("ValidateAndOrderDataDependencies: %v", err)
	}
}

// Callbacks for requests on a deleted pod must not
// recreate it, so a pod recreated under the same name starts clean.
func TestLateCallbacksDoNotResurrectDeletedPod(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	running := r.route("a", "p0")
	r.header(running, "200")
	r.chunk(running, true)
	pending := r.route("b", "p0")
	r.del("p0")
	r.chunk(pending, true)
	r.end(pending, false, fwkrc.TerminationCauseNatural)
	r.end(running, false, fwkrc.TerminationCauseNatural)
	if r.p.model.Has(r.id("p0")) {
		t.Fatal("late callbacks recreated the deleted pod")
	}
	if len(r.p.requests) != 0 {
		t.Fatalf("request state for the deleted pod survived: %v", r.p.requests)
	}
}

// A pod deleted and added again under the same name is tracked afresh.
func TestReaddedPodTrackedAgain(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	r.serve("a", "p0")
	r.del("p0")
	r.replace("p0")
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.Absent {
		t.Fatalf("Tier(a) = %v, want Absent: the re-added pod inherited old residency", got)
	}
	r.serve("b", "p0")
	if got := r.p.model.Tier(r.id("p0"), "b"); got != residency.GPU {
		t.Fatalf("Tier(b) = %v, want GPU once the pod is added again", got)
	}
}

// An add carrying a different endpoint object under an existing ID is a new
// generation (Runtime.NewEndpoint builds a fresh object). Order add(old) -> activity(old) -> add(new) -> delete(old).
func TestReplacementAddStartsNewGeneration(t *testing.T) {
	r := newRig(t, Parameters{}, 2, "p0")
	r.serve("a", "p0") // old generation: a resident
	running := r.route("c", "p0")
	r.header(running, "200")
	r.chunk(running, true) // old generation: c running
	pending := r.route("d", "p0")
	old := r.replace("p0")
	for _, adapter := range []string{"a", "c"} {
		if got := r.p.model.Tier(r.id("p0"), adapter); got != residency.Absent {
			t.Fatalf("Tier(%s) = %v, want Absent: the replacement inherited old residency", adapter, got)
		}
	}
	r.chunk(pending, true)
	r.end(pending, false, fwkrc.TerminationCauseNatural)
	r.end(running, false, fwkrc.TerminationCauseNatural)
	for _, adapter := range []string{"c", "d"} {
		if got := r.p.model.Tier(r.id("p0"), adapter); got != residency.Absent {
			t.Fatalf("Tier(%s) = %v, want Absent: an old-generation callback reached the replacement", adapter, got)
		}
	}
	r.serve("x", "p0") // new generation
	r.extract(fwkdl.EventDelete, old)
	if got := r.p.model.Tier(r.id("p0"), "x"); got != residency.GPU {
		t.Fatalf("Tier(x) = %v, want GPU: the old object's delete wiped the replacement", got)
	}
	if got := len(r.p.requests); got != 0 {
		t.Fatalf("request records = %d, want 0", got)
	}
}

// The director clones candidates before the scheduler invokes the
// scorer, so a replacement can land after the old endpoint was cloned and
// before Score runs. The decision must still be bound to the clone's
// generation, not to when Score ran.
func TestSnapshotBeforeReplacementScoredAfterRejected(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	snap, _ := r.snapshot() // director clones the old endpoint
	r.replace("p0")
	req := r.scoreOn("a", snap) // Score runs after the replacement
	r.preRequest(req, "p0")
	r.succeed(req)
	if r.p.model.Tier(r.id("p0"), "a") != residency.Absent || len(r.p.requests) != 0 {
		t.Fatalf("an old-snapshot decision was recorded against the replacement: Tier=%v requests=%v",
			r.p.model.Tier(r.id("p0"), "a"), r.p.requests)
	}
}

// A replacement between Score and PreRequest.
func TestReplacementBetweenScoreAndPreRequestRejected(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	req := r.scoreReq("a")
	r.replace("p0")
	r.preRequest(req, "p0")
	r.succeed(req)
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.Absent {
		t.Fatalf("Tier(replacement, a) = %v, want Absent: a pre-replacement decision was recorded", got)
	}
}

// The same race across a delete and re-add of the same name.
func TestStaleSnapshotAcrossDeleteAndReaddRejected(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	req := r.scoreReq("a")
	r.del("p0")
	r.replace("p0")
	r.preRequest(req, "p0")
	r.succeed(req)
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent: the decision predates the delete", got)
	}
}

// A delete between Score and PreRequest, with no re-add.
func TestDeleteBetweenScoreAndPreRequestRejected(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	req := r.scoreReq("a")
	r.del("p0")
	r.preRequest(req, "p0")
	r.succeed(req)
	if r.p.model.Has(r.id("p0")) || len(r.p.requests) != 0 {
		t.Fatalf("a decision for a deleted pod was recorded: requests=%v", r.p.requests)
	}
}

// A same-object update anywhere in the decision is not a new generation.
func TestUpdateAroundScoreAccepted(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	snap, _ := r.snapshot()
	r.update("p0")
	req := r.scoreOn("a", snap)
	r.update("p0")
	r.preRequest(req, "p0")
	r.succeed(req)
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU: an update is not a generation change", got)
	}
}

// A generation change or an update on another pod must not reject a decision
// for this one.
func TestOtherPodGenerationDoesNotReject(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0", "p1")
	req := r.scoreReq("a")
	r.replace("p1")
	r.update("p1")
	r.preRequest(req, "p0")
	r.succeed(req)
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.GPU {
		t.Fatalf("Tier(p0) = %v, want GPU: only p1 changed generation", got)
	}
}

// A same-object update in the middle of a request keeps it.
func TestUpdateMidRequestKeepsRequest(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	req := r.route("a", "p0")
	r.header(req, "200")
	r.update("p0")
	r.chunk(req, true)
	r.update("p0")
	r.end(req, false, fwkrc.TerminationCauseNatural)
	if got := r.p.model.Tier(r.id("p0"), "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU: updates mid-request must not drop it", got)
	}
}

// A stale candidate snapshot scored after a delete must not recreate an
// estimate for the deleted pod, which nothing would ever remove.
func TestScoreAfterDeleteDoesNotRecreatePod(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	snap, _ := r.snapshot()
	r.del("p0")
	r.scoreOn("a", snap)
	if r.p.model.Has(r.id("p0")) {
		t.Fatal("Score recreated an estimate for a deleted pod")
	}
}

// An endpoint the plugin never stamped (announced before it was configured)
// cannot be checked by generation; the tombstone still rejects a decision for
// it within tombstoneTTL, after which it is accepted, a stated limit.
func TestUnstampedEndpointGuardedByTombstone(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	o := r.cur["p0"]
	unstamped := map[string]fwksched.Endpoint{"p0": fwksched.NewEndpoint(o.GetMetadata(), o.GetMetrics(), nil)}
	r.del("p0")
	req := r.scoreOn("a", unstamped)
	r.preRequest(req, "p0")
	if len(r.p.requests) != 0 {
		t.Fatal("a decision for a deleted, unstamped pod was accepted within tombstoneTTL")
	}
	r.now = r.now.Add(tombstoneTTL)
	req = r.scoreOn("a", unstamped)
	r.preRequest(req, "p0")
	if len(r.p.requests) != 1 {
		t.Fatal("after tombstoneTTL an unstamped decision should be accepted (stated limit)")
	}
}

// Tombstones expire and are pruned on later deletes.
func TestTombstonesPruned(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0", "p1")
	r.del("p0")
	r.now = r.now.Add(tombstoneTTL)
	r.del("p1")
	if _, ok := r.p.deleted[r.id("p0")]; ok {
		t.Fatalf("expired tombstone for p0 not pruned: %v", r.p.deleted)
	}
	if _, ok := r.p.deleted[r.id("p1")]; !ok {
		t.Fatalf("fresh tombstone for p1 missing: %v", r.p.deleted)
	}
}

// Replacements concurrent with scoring. Whatever the interleaving, a
// request record may exist only for a decision made on the pod's current
// generation; run with -race.
func TestConcurrentReplacementNeverAcceptsStaleSnapshot(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	meta, metrics := r.cur["p0"].GetMetadata(), r.cur["p0"].GetMetrics()
	var objMu sync.Mutex
	current := r.cur["p0"]
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { // the runtime replacing the endpoint
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			obj := fwkdl.NewEndpoint(meta, metrics)
			_ = r.p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: obj})
			objMu.Lock()
			current = obj
			objMu.Unlock()
		}
	}()
	for i := 0; i < 500; i++ { // the director scheduling requests
		objMu.Lock()
		o := current
		objMu.Unlock()
		ep := fwksched.NewEndpoint(o.GetMetadata(), o.GetMetrics(), o.GetAttributes())
		req := &fwksched.InferenceRequest{RequestID: string(rune(0x1000 + i)), TargetModel: "a"}
		r.runScore(req, []fwksched.Endpoint{ep})
		result := &fwksched.SchedulingResult{PrimaryProfileName: "default", ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{ep}},
		}}
		if err := r.p.PreRequest(ctx, req, result); err != nil {
			t.Fatal(err)
		}
		v, ok := ep.Get(generationKey)
		r.p.mu.Lock()
		_, recorded := r.p.requests[req.RequestID]
		if recorded && (!ok || uint64(v.(generation)) != r.p.genSeq[r.id("p0")]) {
			r.p.mu.Unlock()
			t.Fatalf("request %d recorded from a stale snapshot", i)
		}
		r.p.mu.Unlock()
	}
	close(stop)
	wg.Wait()
}

// A first add for an ID is a new pod too: state recorded under the ID before
// the add (which the runtime's ordering should rule out) is not inherited.
func TestFirstAddForgetsEarlierState(t *testing.T) {
	r := newRig(t, Parameters{}, 1)
	r.p.model.Routed("ns/p0", "a")
	r.p.model.Started("ns/p0", "a", r.now)
	obj := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "ns", Name: "p0"}},
		&fwkdl.Metrics{ActiveModels: map[string]int{}, WaitingModels: map[string]int{}, MaxActiveModels: 1})
	r.extract(fwkdl.EventAddOrUpdate, obj)
	if r.p.model.Has("ns/p0") {
		t.Fatal("the first add kept state recorded before it")
	}
}

// Scoring a snapshot cloned before a replacement must not write the old
// object's metrics into the replacement's estimate.
func TestStaleSnapshotMetricsNotObserved(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	snap, _ := r.snapshot()
	snap["p0"].GetMetrics().ActiveModels["x"] = 1 // the old object's last scrape
	old := r.cur["p0"]
	repl := fwkdl.NewEndpoint(old.GetMetadata(),
		&fwkdl.Metrics{ActiveModels: map[string]int{}, WaitingModels: map[string]int{}, MaxActiveModels: 1})
	r.extract(fwkdl.EventAddOrUpdate, repl)
	r.cur["p0"] = repl
	r.serve("x", "p0") // the replacement holds x, idle
	r.scoreOn("b", snap)
	if got := r.p.model.Admit(r.id("p0"), "b"); got.Blocked {
		t.Fatalf("Admit(b) = %+v: the old object's ActiveModels protected x on the replacement", got)
	}
}

// For an unstamped snapshot endpoint only the tombstone shows the pod is gone,
// so Score must consult it too.
func TestUnstampedScoreAfterDeleteDoesNotRecreatePod(t *testing.T) {
	r := newRig(t, Parameters{}, 1, "p0")
	o := r.cur["p0"]
	unstamped := map[string]fwksched.Endpoint{"p0": fwksched.NewEndpoint(o.GetMetadata(), o.GetMetrics(), nil)}
	r.del("p0")
	r.scoreOn("a", unstamped)
	if r.p.model.Has(r.id("p0")) {
		t.Fatal("Score recreated an estimate for a deleted, unstamped pod")
	}
}
