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
	"math"
	"strconv"
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
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraresidency/lorascore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraresidency/lorascore/residency"
)

var ctx = context.Background()

// rig drives the plugin the way the framework does. The runtime announces each
// endpoint object with an add event, normally before its first scrape, and
// stops the old object's collector before announcing its delete. The director
// scores each request on a candidate snapshot cloned from those objects and
// passes the picked snapshot endpoint to PreRequest. Score sees the snapshot
// through the scheduler's data scope, which hides every key the plugin does
// not declare.
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
// announcing each pod as the runtime does. set, if not nil, edits the default
// parameters.
func newRig(t *testing.T, set func(*Parameters), maxLoras int, pods ...string) *rig {
	t.Helper()
	params := DefaultParameters()
	params.BaseModels = []string{"base"}
	if set != nil {
		set(&params)
	}
	p, err := newPlugin("test", params)
	if err != nil {
		t.Fatalf("newPlugin: %v", err)
	}
	r := &rig{
		t: t, p: p, now: time.Unix(1_000_000, 0),
		cur: map[string]fwkdl.Endpoint{}, gone: map[string]bool{},
		held: map[string]map[string]fwksched.Endpoint{},
	}
	p.now = func() time.Time { return r.now }
	for _, name := range pods {
		obj := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "ns", Name: name}}, nil)
		r.order = append(r.order, name)
		r.cur[name] = obj
		r.extract(fwkdl.EventAddOrUpdate, obj)
		r.scrape(name, maxLoras)
	}
	return r
}

// scrape writes a scrape of pod's current object, with no active adapters, as
// the metrics extractor does: a new snapshot stamped with the scrape time.
func (r *rig) scrape(pod string, maxLoras int) {
	r.cur[pod].UpdateMetrics(&fwkdl.Metrics{
		ActiveModels: map[string]int{}, WaitingModels: map[string]int{},
		MaxActiveModels: maxLoras, UpdateTime: r.now,
	})
}

func (r *rig) id(pod string) string { return "ns/" + pod }

// tier is where adapter is estimated to live on pod.
func (r *rig) tier(pod, adapter string) residency.Tier {
	return r.admit(pod, adapter).Tier
}

// admit is the estimated admission of adapter on pod.
func (r *rig) admit(pod, adapter string) residency.Admission {
	r.p.mu.Lock()
	defer r.p.mu.Unlock()
	if e := r.p.pods[r.id(pod)]; e != nil {
		return e.Admit(adapter)
	}
	return residency.Admission{Tier: residency.Absent}
}

// tracked reports whether the plugin holds an estimate for pod.
func (r *rig) tracked(pod string) bool {
	r.p.mu.Lock()
	defer r.p.mu.Unlock()
	_, ok := r.p.pods[r.id(pod)]
	return ok
}

func (r *rig) extract(eventType fwkdl.EventType, ep fwkdl.Endpoint) {
	r.t.Helper()
	if err := r.p.Extract(ctx, fwkdl.EndpointEvent{Type: eventType, Endpoint: ep}); err != nil {
		r.t.Fatalf("Extract: %v", err)
	}
}

// metrics is pod's current object's latest scrape.
func (r *rig) metrics(pod string) *fwkdl.Metrics { return r.cur[pod].GetMetrics() }

// replace announces a new object for pod, unscraped as NewEndpoint builds it,
// a moment after the old object's last scrape, then scrapes it with the old
// object's max_loras. It returns the object it replaced.
func (r *rig) replace(pod string) fwkdl.Endpoint {
	old := r.cur[pod]
	r.now = r.now.Add(time.Millisecond)
	obj := fwkdl.NewEndpoint(old.GetMetadata(), nil)
	r.extract(fwkdl.EventAddOrUpdate, obj)
	r.cur[pod] = obj
	delete(r.gone, pod)
	r.now = r.now.Add(time.Millisecond)
	r.scrape(pod, old.GetMetrics().MaxActiveModels)
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

func TestNewPluginValidates(t *testing.T) {
	for name, set := range map[string]func(*Parameters){
		"no base models":     func(p *Parameters) { p.BaseModels = nil },
		"bad half-life":      func(p *Parameters) { p.DemandHalfLife = "soon" },
		"zero half-life":     func(p *Parameters) { p.DemandHalfLife = "0s" },
		"negative weight":    func(p *Parameters) { p.CopyCost = -1 },
		"negative cpu loras": func(p *Parameters) { p.MaxCPULoras = -1 },
	} {
		p := DefaultParameters()
		p.BaseModels = []string{"b"}
		set(&p)
		if _, err := newPlugin("x", p); err == nil {
			t.Errorf("%s: newPlugin succeeded, want an error", name)
		}
	}
}

func TestFactoryDecodesAndRejectsUnknownFields(t *testing.T) {
	good := json.RawMessage(`{"baseModels":["base"],"maxCpuLoras":4,"demandHalfLife":"30s",` +
		`"copyCost":0.3,"loadCost":2,"blockedCost":3,"pendingCost":0.4,"evictionWeight":0.5}`)
	pl, err := Factory("n", fwkplugin.StrictDecoder(good), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	p := pl.(*Plugin)
	if want := (lorascore.Weights{CopyCost: 0.3, LoadCost: 2, BlockedCost: 3, PendingCost: 0.4, EvictionWeight: 0.5}); p.weights != want {
		t.Fatalf("weights = %+v, want %+v", p.weights, want)
	}
	if p.cpuSlots != 4 {
		t.Fatalf("cpuSlots = %d, want 4", p.cpuSlots)
	}
	partial := json.RawMessage(`{"baseModels":["base"],"evictionWeight":0.5}`)
	pl, err = Factory("n", fwkplugin.StrictDecoder(partial), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	want := lorascore.DefaultWeights()
	want.EvictionWeight = 0.5
	if got := pl.(*Plugin).weights; got != want {
		t.Fatalf("weights = %+v, want defaults with eviction 0.5: %+v", got, want)
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
	r := newRig(t, nil, 1, "p0", "p1")
	r.serve("a", "p0")
	if s := r.score("base"); s["p0"] != s["p1"] {
		t.Fatalf("base-model request must score pods equally, got %v", s)
	}
	req := r.route("base", "p1")
	r.end(req, true, fwkrc.TerminationCauseNatural)
	if got := r.tier("p1", "base"); got != residency.Absent {
		t.Fatalf("base model tracked as an adapter: Tier = %v", got)
	}
}

// A request without a target model names no adapter: it is scored neutrally
// and leaves no trace.
func TestEmptyTargetModelIsNeutralAndUntracked(t *testing.T) {
	r := newRig(t, nil, 1, "p0", "p1")
	r.serve("a", "p0")
	if s := r.score(""); s["p0"] != 1 || s["p1"] != 1 {
		t.Fatalf("empty target model must score every pod 1, got %v", s)
	}
	req := r.route("", "p1")
	if len(r.p.requests) != 0 {
		t.Fatalf("empty target model recorded: %v", r.p.requests)
	}
	r.end(req, true, fwkrc.TerminationCauseNatural)
	if got := r.admit("p1", ""); got.Tier != residency.Absent || got.Pending {
		t.Fatalf("empty target model tracked as an adapter: %+v", got)
	}
	r.route("a", "p0")
	if got := r.p.demand.Shares([]string{"a"}, r.now)["a"]; got != 1 {
		t.Fatalf("Share(a) = %v, want 1: the empty target model was counted as demand", got)
	}
}

// The datalayer binds one endpoint extractor per type, so a second instance
// would never see endpoint events.
func TestFactoryRejectsMultipleInstances(t *testing.T) {
	params := json.RawMessage(`{"baseModels":["base"]}`)
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	first, err := Factory("first", fwkplugin.StrictDecoder(params), handle)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	handle.AddPlugin(first.TypedName().Name, first)
	if _, err := Factory("second", fwkplugin.StrictDecoder(params), handle); err == nil {
		t.Fatal("Factory accepted a second instance")
	}
}

// maxCpuLoras sizes the CPU tier: with max_loras 1 and two CPU slots, the
// adapter displaced from GPU is still in the CPU cache.
func TestMaxCPULorasSizesCPUTier(t *testing.T) {
	for cpu, want := range map[int]residency.Tier{0: residency.Absent, 2: residency.CPU} {
		r := newRig(t, func(p *Parameters) { p.MaxCPULoras = cpu }, 1, "p0")
		r.serve("a", "p0")
		r.serve("b", "p0")
		if got := r.tier("p0", "a"); got != want {
			t.Errorf("maxCpuLoras %d: Tier(a) = %v, want %v", cpu, got, want)
		}
	}
}

// demandHalfLife sets how fast demand decays.
func TestDemandHalfLifeApplied(t *testing.T) {
	r := newRig(t, func(p *Parameters) { p.DemandHalfLife = "10s" }, 1, "p0")
	r.route("a", "p0")
	r.now = r.now.Add(10 * time.Second)
	r.route("b", "p0")
	// a has decayed by one half-life when b arrives: rates 1/2 and 1.
	if got := r.p.demand.Shares([]string{"a"}, r.now)["a"]; math.Abs(got-1.0/3) > 1e-9 {
		t.Fatalf("Share(a) = %v, want 1/3 after one half-life", got)
	}
}

func TestServedAdapterAttractsItsNextRequest(t *testing.T) {
	r := newRig(t, nil, 1, "p0", "p1")
	r.serve("a", "p0")
	if got := r.tier("p0", "a"); got != residency.GPU {
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
	r := newRig(t, nil, 1, "p0", "p1")
	req := r.route("a", "p0")
	r.header(req, "200")
	if got := r.tier("p0", "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent before the first chunk", got)
	}
	if s := r.score("a"); !(s["p0"] > s["p1"]) {
		t.Fatalf("pending pod should attract the burst, got %v", s)
	}
	r.chunk(req, true)
	if got := r.tier("p0", "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU after the first chunk", got)
	}
}

func TestErrorStatusIsNotExecution(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	req := r.route("a", "p0")
	r.header(req, "503")
	r.end(req, true, fwkrc.TerminationCauseNatural) // an error delivered in full is natural
	if got := r.tier("p0", "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent after an error response", got)
	}
	if len(r.p.requests) != 0 {
		t.Fatalf("request state leaked: %v", r.p.requests)
	}
}

// A plain "status" key, which the EPP's own check also accepts, counts as an
// error status too.
func TestPlainStatusKeyErrorIsNotExecution(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	req := r.route("a", "p0")
	r.headerKey(req, "status", "404")
	r.end(req, true, fwkrc.TerminationCauseNatural)
	if got := r.tier("p0", "a"); got != residency.Absent {
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
		r := newRig(t, nil, 1, "p0")
		req := r.route("a", "p0")
		r.header(req, "200")
		r.end(req, true, cause)
		if got := r.tier("p0", "a"); got != residency.Absent {
			t.Fatalf("%s: Tier = %v, want Absent after the cleanup call", cause, got)
		}
		if got := r.admit("p0", "a"); got.Pending {
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
	r := newRig(t, nil, 1, "p0")
	req := r.route("a", "p0")
	r.header(req, "200")
	r.end(req, true, fwkrc.TerminationCauseNatural)
	if got := r.tier("p0", "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU after a complete one-shot response", got)
	}
}

// A stream cut short after its first chunk did execute.
func TestDisconnectAfterFirstChunkStillExecuted(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	req := r.route("a", "p0")
	r.header(req, "200")
	r.chunk(req, true)
	r.end(req, false, fwkrc.TerminationCauseClientDisconnect)
	if got := r.tier("p0", "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU: the adapter ran before the disconnect", got)
	}
}

// An error status in the framework's header map must stop the first chunk from counting, on the streaming path too.
func TestStreamedErrorStatusIsNotExecution(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	req := r.route("a", "p0")
	r.header(req, "500")
	r.chunk(req, true)
	r.end(req, false, fwkrc.TerminationCauseNatural)
	if got := r.tier("p0", "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent after a 500 response", got)
	}
}

func TestDuplicatePreRequestCountedOnce(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	req := r.route("a", "p0")
	if err := r.p.PreRequest(ctx, req, req.SchedulingResult); err != nil {
		t.Fatal(err)
	}
	r.end(req, true, fwkrc.TerminationCauseClientDisconnect) // one abort clears the one pending route
	if got := r.admit("p0", "a"); got.Pending {
		t.Fatalf("Admit = %+v, want the duplicate PreRequest ignored", got)
	}
}

// ActiveModels feeds ObserveActive: an adapter the pod reports active is
// protected, so the pod counts as blocked rather than offering it as a victim.
func TestActiveModelsProtectResidentAdapter(t *testing.T) {
	r := newRig(t, func(p *Parameters) { p.BlockedCost, p.EvictionWeight = 10, 0 }, 1, "p0", "p1")
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
	r := newRig(t, nil, 2, "p0")
	r.serve("a", "p0")
	r.serve("b", "p0")
	if got := r.tier("p0", "a"); got != residency.GPU {
		t.Fatalf("Tier(a) = %v, want GPU with max_loras 2", got)
	}
}

func TestEndpointDeleteForgetsPod(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	r.serve("a", "p0")
	r.del("p0")
	if got := r.tier("p0", "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent after the endpoint is deleted", got)
	}
	if got := len(r.p.endpoints) + len(r.p.pods); got != 0 {
		t.Fatalf("endpoint/estimate records = %d after delete, want 0 (pods that never return must not leak)", got)
	}
}

// Deleting one pod keeps the others' estimates.
func TestDeleteForgetsOnlyThatPod(t *testing.T) {
	r := newRig(t, nil, 1, "p0", "p1")
	r.serve("a", "p0")
	r.serve("b", "p1")
	r.del("p1")
	if r.tracked("p1") || !r.tracked("p0") {
		t.Fatalf("after deleting p1: tracked p0=%v p1=%v, want only p0", r.tracked("p0"), r.tracked("p1"))
	}
}

// An update re-dispatches the same endpoint object (Runtime.UpdateEndpoint),
// so it must keep the pod's state.
func TestEndpointUpdateKeepsPod(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	r.serve("a", "p0")
	r.update("p0")
	if got := r.tier("p0", "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU kept across an update of the same endpoint", got)
	}
	r.serve("b", "p0")
	if got := r.tier("p0", "b"); got != residency.GPU {
		t.Fatalf("Tier(b) = %v, want GPU: an update must not drop later requests", got)
	}
}

// The declared dependencies must pass the EPP's startup validation beside the
// core metrics extractor, the producer the EPP runs with the plugin.
func TestDependenciesStartupValid(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	ext, err := metrics.NewCoreMetricsExtractor(metrics.NewMappingRegistry(), "")
	if err != nil {
		t.Fatalf("NewCoreMetricsExtractor: %v", err)
	}
	plugins := []fwkplugin.Plugin{ext, r.p}
	if _, err := fwkdl.BuildRegistry(plugins); err != nil {
		t.Fatalf("BuildRegistry: %v", err)
	}
	if _, err := datalayer.ValidateAndOrderDataDependencies(plugins); err != nil {
		t.Fatalf("ValidateAndOrderDataDependencies: %v", err)
	}
}

type recordingRegistrar struct {
	registrations []fwkdl.PendingRegistration
}

func (r *recordingRegistrar) Register(registration fwkdl.PendingRegistration) error {
	r.registrations = append(r.registrations, registration)
	return nil
}

// Extract only runs if the plugin registers for endpoint notifications.
func TestRegistersEndpointExtractor(t *testing.T) {
	r := newRig(t, nil, 1)
	reg := &recordingRegistrar{}
	if err := r.p.RegisterDependencies(reg); err != nil {
		t.Fatalf("RegisterDependencies: %v", err)
	}
	if len(reg.registrations) != 1 {
		t.Fatalf("registrations = %d, want 1", len(reg.registrations))
	}
	got := reg.registrations[0]
	if got.Owner != r.p.TypedName() || got.SourceType != sourcenotifications.EndpointNotificationSourceType ||
		got.Extractor != fwkdl.EndpointExtractor(r.p) || got.DefaultSource == nil {
		t.Fatalf("registration = %+v, want owner %v, source type %q, the plugin as extractor and a default source",
			got, r.p.TypedName(), sourcenotifications.EndpointNotificationSourceType)
	}
}

// Callbacks for requests on a deleted pod must not
// recreate it, so a pod recreated under the same name starts clean.
func TestLateCallbacksDoNotResurrectDeletedPod(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	running := r.route("a", "p0")
	r.header(running, "200")
	r.chunk(running, true)
	pending := r.route("b", "p0")
	r.del("p0")
	r.chunk(pending, true)
	r.end(pending, false, fwkrc.TerminationCauseNatural)
	r.end(running, false, fwkrc.TerminationCauseNatural)
	if r.tracked("p0") {
		t.Fatal("late callbacks recreated the deleted pod")
	}
	if len(r.p.requests) != 0 {
		t.Fatalf("request state for the deleted pod survived: %v", r.p.requests)
	}
}

// A pod deleted and added again under the same name is tracked afresh.
func TestReaddedPodTrackedAgain(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	r.serve("a", "p0")
	r.del("p0")
	r.replace("p0")
	if got := r.tier("p0", "a"); got != residency.Absent {
		t.Fatalf("Tier(a) = %v, want Absent: the re-added pod inherited old residency", got)
	}
	r.serve("b", "p0")
	if got := r.tier("p0", "b"); got != residency.GPU {
		t.Fatalf("Tier(b) = %v, want GPU once the pod is added again", got)
	}
}

// An add carrying a different endpoint object under an existing ID is a new pod
// (Runtime.NewEndpoint builds a fresh object). Order: add(old), activity(old),
// add(new), delete(old).
func TestReplacementAddStartsEmpty(t *testing.T) {
	r := newRig(t, nil, 2, "p0")
	r.serve("a", "p0") // old object: a resident
	running := r.route("c", "p0")
	r.header(running, "200")
	r.chunk(running, true) // old object: c running
	pending := r.route("d", "p0")
	old := r.replace("p0")
	for _, adapter := range []string{"a", "c"} {
		if got := r.tier("p0", adapter); got != residency.Absent {
			t.Fatalf("Tier(%s) = %v, want Absent: the replacement inherited old residency", adapter, got)
		}
	}
	r.chunk(pending, true)
	r.end(pending, false, fwkrc.TerminationCauseNatural)
	r.end(running, false, fwkrc.TerminationCauseNatural)
	for _, adapter := range []string{"c", "d"} {
		if got := r.tier("p0", adapter); got != residency.Absent {
			t.Fatalf("Tier(%s) = %v, want Absent: an old-object callback reached the replacement", adapter, got)
		}
	}
	r.serve("x", "p0") // new object
	r.extract(fwkdl.EventDelete, old)
	if got := r.tier("p0", "x"); got != residency.GPU {
		t.Fatalf("Tier(x) = %v, want GPU: the old object's delete wiped the replacement", got)
	}
	if got := len(r.p.requests); got != 0 {
		t.Fatalf("request records = %d, want 0", got)
	}
}

// A delete between Score and PreRequest, with no re-add.
func TestDeleteBetweenScoreAndPreRequestRejected(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	req := r.scoreReq("a")
	r.del("p0")
	r.preRequest(req, "p0")
	r.succeed(req)
	if r.tracked("p0") || len(r.p.requests) != 0 {
		t.Fatalf("a decision for a deleted pod was recorded: requests=%v", r.p.requests)
	}
}

// Scoring a snapshot cloned before a replacement must not write the old
// object's metrics into the replacement's estimate.
func TestStaleSnapshotMetricsNotObserved(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	r.metrics("p0").ActiveModels["x"] = 1 // the old object's last scrape
	snap, _ := r.snapshot()
	r.replace("p0")
	r.serve("x", "p0") // the replacement holds x, idle
	r.scoreOn("b", snap)
	if got := r.admit("p0", "b"); got.Blocked {
		t.Fatalf("Admit(b) = %+v: the old object's ActiveModels protected x on the replacement", got)
	}
}

// The director clones candidates before the scheduler invokes the scorer, so a
// replacement can land after the old endpoint was cloned and before Score
// runs. The decision is bound to the clone, not to when Score ran.
func TestSnapshotBeforeReplacementScoredAfterRejected(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	snap, _ := r.snapshot()
	r.replace("p0")
	req := r.scoreOn("a", snap)
	r.preRequest(req, "p0")
	r.succeed(req)
	if got := r.tier("p0", "a"); got != residency.Absent || len(r.p.requests) != 0 {
		t.Fatalf("an old-snapshot decision was recorded against the replacement: Tier=%v requests=%v", got, r.p.requests)
	}
}

// A replacement between Score and PreRequest.
func TestReplacementBetweenScoreAndPreRequestRejected(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	req := r.scoreReq("a")
	r.replace("p0")
	r.preRequest(req, "p0")
	r.succeed(req)
	if got := r.tier("p0", "a"); got != residency.Absent {
		t.Fatalf("Tier(replacement, a) = %v, want Absent: a pre-replacement decision was recorded", got)
	}
}

// The same race across a delete and re-add of the same name.
func TestStaleSnapshotAcrossDeleteAndReaddRejected(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	req := r.scoreReq("a")
	r.del("p0")
	r.replace("p0")
	r.preRequest(req, "p0")
	r.succeed(req)
	if got := r.tier("p0", "a"); got != residency.Absent {
		t.Fatalf("Tier = %v, want Absent: the decision predates the delete", got)
	}
}

// A replacement's first snapshots, taken before its first scrape, carry no
// metrics from the old object and are accepted.
func TestUnscrapedReplacementSnapshotAccepted(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	r.extract(fwkdl.EventDelete, r.cur["p0"])
	obj := fwkdl.NewEndpoint(r.cur["p0"].GetMetadata(), nil)
	r.extract(fwkdl.EventAddOrUpdate, obj)
	r.cur["p0"] = obj
	r.serve("a", "p0")
	if got := r.tier("p0", "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU: a decision on the unscraped replacement was rejected", got)
	}
}

// A deleted candidate is still scored, as a pod that holds nothing.
func TestDeletedCandidateScoredAsEmpty(t *testing.T) {
	r := newRig(t, nil, 1, "p0", "p1")
	r.serve("a", "p0")
	r.serve("a", "p1")
	snap, _ := r.snapshot()
	r.del("p1")
	_, list := r.snapshot()
	list = append(list, snap["p1"])
	s := map[string]float64{}
	for ep, v := range r.runScore(&fwksched.InferenceRequest{TargetModel: "a"}, list) {
		s[ep.GetMetadata().ID.Name] = v
	}
	if _, scored := s["p1"]; !scored || s["p0"] != 1 || s["p1"] != 0 {
		t.Fatalf("want the live holder p0 = 1 and the deleted p1 scored 0, got %v", s)
	}
}

// A same-object update anywhere in the decision is not a new pod.
func TestUpdateAroundScoreAccepted(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	snap, _ := r.snapshot()
	r.update("p0")
	req := r.scoreOn("a", snap)
	r.update("p0")
	r.preRequest(req, "p0")
	r.succeed(req)
	if got := r.tier("p0", "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU: an update is not a new pod", got)
	}
}

// A replacement or an update of another pod must not reject a decision for
// this one.
func TestOtherPodReplacementDoesNotReject(t *testing.T) {
	r := newRig(t, nil, 1, "p0", "p1")
	req := r.scoreReq("a")
	r.replace("p1")
	r.update("p1")
	r.preRequest(req, "p0")
	r.succeed(req)
	if got := r.tier("p0", "a"); got != residency.GPU {
		t.Fatalf("Tier(p0) = %v, want GPU: only p1 was replaced", got)
	}
}

// A same-object update in the middle of a request keeps it.
func TestUpdateMidRequestKeepsRequest(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	req := r.route("a", "p0")
	r.header(req, "200")
	r.update("p0")
	r.chunk(req, true)
	r.update("p0")
	r.end(req, false, fwkrc.TerminationCauseNatural)
	if got := r.tier("p0", "a"); got != residency.GPU {
		t.Fatalf("Tier = %v, want GPU: updates mid-request must not drop it", got)
	}
}

// A stale candidate snapshot scored after a delete must not recreate an
// estimate for the deleted pod, which nothing would ever remove.
func TestScoreAfterDeleteDoesNotRecreatePod(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	snap, _ := r.snapshot()
	r.del("p0")
	r.scoreOn("a", snap)
	if r.tracked("p0") {
		t.Fatal("Score recreated an estimate for a deleted pod")
	}
}

// Replacements and deletes concurrent with scheduling and responses. Whatever
// the interleaving, every estimate and request record must belong to a live
// endpoint; run with -race.
func TestConcurrentLifecycleLeavesNoOrphans(t *testing.T) {
	r := newRig(t, nil, 1, "p0")
	meta, m := r.cur["p0"].GetMetadata(), r.cur["p0"].GetMetrics()
	var objMu sync.Mutex
	current := r.cur["p0"]
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { // the runtime replacing and deleting the endpoint
		defer wg.Done()
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			objMu.Lock()
			old := current
			objMu.Unlock()
			if i%2 == 0 {
				_ = r.p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: old})
			}
			obj := fwkdl.NewEndpoint(meta, m)
			_ = r.p.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: obj})
			objMu.Lock()
			current = obj
			objMu.Unlock()
		}
	}()
	accepted := 0
	for i := 0; i < 2000; i++ { // the director scheduling requests and the server answering
		objMu.Lock()
		o := current
		objMu.Unlock()
		ep := fwksched.NewEndpoint(o.GetMetadata(), o.GetMetrics(), o.GetAttributes())
		req := &fwksched.InferenceRequest{RequestID: strconv.Itoa(i), TargetModel: "a"}
		r.p.Score(ctx, req, []fwksched.Endpoint{ep})
		result := &fwksched.SchedulingResult{PrimaryProfileName: "default", ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{ep}},
		}}
		if err := r.p.PreRequest(ctx, req, result); err != nil {
			t.Fatal(err)
		}
		r.p.mu.Lock()
		if _, ok := r.p.requests[req.RequestID]; ok {
			accepted++
		}
		r.p.mu.Unlock()
		r.p.ResponseHeader(ctx, req, &fwkrc.Response{RequestID: req.RequestID, Headers: map[string]string{":status": "200"}}, nil)
		r.p.ResponseBody(ctx, req, &fwkrc.Response{RequestID: req.RequestID, StartOfStream: true}, nil)
		r.p.ResponseBody(ctx, req, &fwkrc.Response{RequestID: req.RequestID, EndOfStream: true, TerminationCause: fwkrc.TerminationCauseNatural}, nil)
		r.p.mu.Lock()
		for id := range r.p.pods {
			if _, live := r.p.endpoints[id]; !live {
				r.p.mu.Unlock()
				t.Fatalf("iteration %d: estimate for %s, which has no live endpoint", i, id)
			}
		}
		for rid, rec := range r.p.requests {
			if _, live := r.p.endpoints[rec.pod]; !live {
				r.p.mu.Unlock()
				t.Fatalf("iteration %d: request %s recorded on %s, which has no live endpoint", i, rid, rec.pod)
			}
		}
		r.p.mu.Unlock()
	}
	close(stop)
	wg.Wait()
	if accepted == 0 {
		t.Fatal("no request was ever recorded, so the invariant was never exercised")
	}
}

// A first add for an ID is a new pod too: state recorded under the ID before
// the add (which the runtime's ordering should rule out) is not inherited.
func TestFirstAddForgetsEarlierState(t *testing.T) {
	r := newRig(t, nil, 1)
	r.p.pod("ns/p0").Routed("a")
	r.p.pod("ns/p0").Started("a", r.now)
	obj := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "ns", Name: "p0"}},
		&fwkdl.Metrics{ActiveModels: map[string]int{}, WaitingModels: map[string]int{}, MaxActiveModels: 1})
	r.extract(fwkdl.EventAddOrUpdate, obj)
	if r.tracked("p0") {
		t.Fatal("the first add kept state recorded before it")
	}
}
