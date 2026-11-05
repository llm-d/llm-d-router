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

// THE GOLDEN HARNESS — a decision trace driven through the plugin's real hooks.
//
// It exists so that every simplification step can be proved behaviour-neutral: the trace is
// frozen once and must stay byte-identical. Three rules make that proof mean something:
//
//  1. Inputs are production reality. Endpoint events carry NO metrics; capacity arrives only
//     through Saturation's live candidates; Saturation runs before every Pick.
//  2. The hooks are called the way the router calls them. Pick sees the UNSCOPED queue item;
//     Filter and PreRequest receive a SCOPED request (datalayer.ScopeRequest / ScopeInvocation,
//     exactly as the director does), so a hook that reads an undeclared attribute is a violation
//     in the trace instead of a live failure.
//  3. Only decisions are recorded, and nothing a later step deletes (no p50s, no refill share,
//     no event ring, no per-via counters). One helper reads the state digest from the internals.

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
)

const (
	goldenNamespace = "golden"
	// goldenEpoch is a fixed start instant so traces do not depend on the machine's clock.
	goldenEpoch = "2026-01-01T00:00:00Z"
	// goldenBlocks is the block size every engine reports.
	goldenBlocks = 64
)

// goldenClock is the plugin's clock, advanced by the script.
type goldenClock struct{ t time.Time }

func (c *goldenClock) Now() time.Time          { return c.t }
func (c *goldenClock) Advance(d time.Duration) { c.t = c.t.Add(d) }

// goldenQueue is one flow's queue: a stateful head-first list the harness dequeues from, which is
// what the flow controller does after a Pick returns its queue.
type goldenQueue struct {
	key   string
	items []*fcmocks.MockQueueItemAccessor
}

func (q *goldenQueue) Len() int { return len(q.items) }
func (q *goldenQueue) ByteSize() uint64 {
	var n uint64
	for _, it := range q.items {
		n += it.OriginalRequest().ByteSize()
	}
	return n
}
func (q *goldenQueue) Peek() fwkfc.QueueItemAccessor {
	if len(q.items) == 0 {
		return nil
	}
	return q.items[0]
}
func (q *goldenQueue) OrderingPolicy() fwkfc.OrderingPolicy { return nil }
func (q *goldenQueue) FlowKey() fwkfc.FlowKey               { return fwkfc.FlowKey{ID: q.key} }
func (q *goldenQueue) drop()                                { q.items = q.items[1:] }

var _ fwkfc.FlowQueueAccessor = &goldenQueue{}

// goldenHarness is one scenario's world: the plugin, its clock, its pods and its queues.
type goldenHarness struct {
	t     *testing.T
	r     *GroupScheduler
	clock *goldenClock
	tr    *strings.Builder

	pods   map[string]int // pod id -> num_gpu_blocks (0 = metrics not yet visible)
	podBlk map[string]int // pod id -> reported block size (0 = the default)
	queues []*goldenQueue
	band   *fcmocks.MockPriorityBandAccessor
	seq    int
}

// newGoldenHarness builds the plugin exactly as the runner does and wires the trace.
func newGoldenHarness(t *testing.T, pods map[string]int) *goldenHarness {
	t.Helper()
	return newGoldenHarnessWith(t, pods, "")
}

// newGoldenHarnessWith is newGoldenHarness with extra config keys, for the scenarios that need a
// different knob.
func newGoldenHarnessWith(t *testing.T, pods map[string]int, extra string) *goldenHarness {
	t.Helper()
	base := fmt.Sprintf(`{"groupBudgetFraction":0.95,"quantum":4,"idleSeconds":20,"inFlightCap":64,`+
		`"blockSizeTokens":%d,"inflightReapSeconds":9300}`, goldenBlocks)
	// extra keys replace the defaults they name, so a scenario can change one knob
	cfg := base
	if extra != "" {
		cfg = strings.TrimSuffix(base, "}") + extra + "}"
	}
	dec := json.NewDecoder(strings.NewReader(cfg))
	p, err := Factory("gs-golden", dec, nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	r := p.(*GroupScheduler)
	start, err := time.Parse(time.RFC3339, goldenEpoch)
	if err != nil {
		t.Fatalf("epoch: %v", err)
	}
	clock := &goldenClock{t: start}
	r.now = clock.Now
	h := &goldenHarness{t: t, r: r, clock: clock, tr: &strings.Builder{}, pods: map[string]int{},
		podBlk: map[string]int{}, seq: 0}
	r.sched.onEvent = func(ev Event) {
		h.line("EVENT %s thread=%s engine=%s detail=%s", ev.Kind, ev.ID, ev.Engine, ev.Detail)
	}
	h.band = &fcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(fwkfc.FlowQueueAccessor) bool) {
			for _, q := range h.queues {
				if !cb(q) {
					return
				}
			}
		},
	}
	// the pods are announced the production way: an ADD event with NO metrics
	ids := make([]string, 0, len(pods))
	for id := range pods {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		h.pods[id] = pods[id]
		if err := r.Extract(context.Background(), fwkdl.EndpointEvent{
			Type: fwkdl.EventAddOrUpdate, Endpoint: goldenDLEndpoint(id, 0, 0),
		}); err != nil {
			t.Fatalf("Extract add %s: %v", id, err)
		}
	}
	return h
}

// goldenDLEndpoint builds a data-layer pod (what the detector and the endpoint events carry).
// blocks==0 means the metrics have not been scraped yet, which is what every event carries.
func goldenDLEndpoint(id string, numBlocks, blockSize int) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Name: id, Address: id + ":8000",
		ID: types.NamespacedName{Namespace: goldenNamespace, Name: id},
	}, &fwkdl.Metrics{CacheBlockSize: blockSize, CacheNumBlocks: numBlocks})
}

// goldenEndpoint builds a scheduling-layer pod (what the Filter receives).
func goldenEndpoint(id string, numBlocks, blockSize int) fwksched.Endpoint {
	m := &fwkdl.Metrics{CacheBlockSize: blockSize, CacheNumBlocks: numBlocks}
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		Name: id, Address: id + ":8000",
		ID: types.NamespacedName{Namespace: goldenNamespace, Name: id},
	}, m, nil)
}

// candidates is what the detector is offered: the live endpoints WITH their metrics.
func (h *goldenHarness) candidates() []fwkdl.Endpoint {
	ids := make([]string, 0, len(h.pods))
	for id := range h.pods {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]fwkdl.Endpoint, 0, len(ids))
	for _, id := range ids {
		out = append(out, goldenDLEndpoint(id, h.pods[id], h.blockSizeOf(id)))
	}
	return out
}

func (h *goldenHarness) line(format string, args ...any) {
	fmt.Fprintf(h.tr, "t=%d %s\n", h.clock.Now().Sub(mustEpoch()).Milliseconds(), fmt.Sprintf(format, args...))
}

func mustEpoch() time.Time {
	t, err := time.Parse(time.RFC3339, goldenEpoch)
	if err != nil {
		panic(err)
	}
	return t
}

// busyPeriodsFor is a scenario's window into the rotation unit (asserts, never decisions).
func (h *goldenHarness) busyPeriodsFor(thread string) int {
	h.r.sched.mu.Lock()
	defer h.r.sched.mu.Unlock()
	for _, g := range h.r.sched.groups {
		if m, ok := g.threads[thread]; ok {
			return m.busyPeriods
		}
	}
	return -1
}

// advance moves the virtual clock, which is what makes S, Q and the reap reachable.
func (h *goldenHarness) advance(d time.Duration) { h.clock.Advance(d) }

// saturate calls the detector the way the flow controller does before each Pick. That call is the
// capacity handshake too: Saturation is where geometry is learned. There is no tick to call any
// more — the pass tidies at its top, so a scenario that needs housekeeping runs a pass.
func (h *goldenHarness) saturate(_ bool) {
	if got := h.r.Saturation(context.Background(), h.candidates()); got != 0 {
		h.t.Fatalf("Saturation must stay 0, got %v", got)
	}
}

// enqueue puts a request at the tail of its flow's queue, as the flow controller does.
func (h *goldenHarness) enqueue(req *fwksched.InferenceRequest) {
	key := req.FairnessID
	var q *goldenQueue
	for _, cand := range h.queues {
		if cand.key == key {
			q = cand
			break
		}
	}
	if q == nil {
		q = &goldenQueue{key: key}
		h.queues = append(h.queues, q)
	}
	mock := fcmocks.NewMockFlowControlRequest(uint64(len(req.RequestID)+1), req.RequestID,
		fwkfc.FlowKey{ID: key}, func(m *fcmocks.MockFlowControlRequest) { m.InferenceRequestV = req })
	item := &fcmocks.MockQueueItemAccessor{
		EnqueueTimeV:     h.clock.Now(),
		OriginalRequestV: mock,
		HandleV:          &fcmocks.MockQueueItemHandle{},
	}
	q.items = append(q.items, item)
}

// dispatched is one Pick result, handed to the script so it can run that request's hooks.
type dispatched struct {
	reqID  string
	engine string
	req    *fwksched.InferenceRequest
}

// pickLoop runs the flow controller's dispatch loop: Pick until it returns nothing, dequeuing
// the chosen queue's head after each pick, and recording the decision.
func (h *goldenHarness) pickLoop() []dispatched {
	var out []dispatched
	for {
		q, err := h.r.Pick(context.Background(), h.band)
		if err != nil {
			h.t.Fatalf("Pick: %v", err)
		}
		if q == nil {
			h.line("PICK req=none engine=none")
			h.digest()
			return out
		}
		gq := q.(*goldenQueue)
		item := gq.Peek()
		req := item.OriginalRequest().InferenceRequest()
		engine := h.r.sched.bindingOf(req.FairnessID)
		h.line("PICK req=%s engine=%s", req.RequestID, orNone(engine))
		h.digest()
		out = append(out, dispatched{reqID: req.RequestID, engine: engine, req: req})
		gq.drop()
	}
}

// podNameOf turns a seat key ("ns/pod") back into the pod's name.
func podNameOf(seat string) string {
	if i := strings.LastIndex(seat, "/"); i >= 0 {
		return seat[i+1:]
	}
	return seat
}

func orNone(s string) string {
	if s == "" {
		return "none"
	}
	return s
}

// filter runs the Filter the way the scheduling profile does: a scoped request and scoped pods.
func (h *goldenHarness) filter(req *fwksched.InferenceRequest) []fwksched.Endpoint {
	pods := goldenPods(h.pods)
	scopedReq, scopedPods, violations := datalayer.ScopeInvocation(logr.Discard(), "Filter", h.r, req, pods)
	if err := violations.Write(); err != nil {
		h.t.Fatalf("the Filter wrote an undeclared attribute: %v", err)
	}
	kept := h.r.Filter(context.Background(), scopedReq, scopedPods)
	ids := make([]string, 0, len(kept))
	for _, ep := range kept {
		ids = append(ids, ep.GetMetadata().Name)
	}
	sort.Strings(ids)
	h.line("FILTER req=%s -> %s", req.RequestID, strings.Join(ids, ","))
	return kept
}

// blockSizeOf is the block size a pod reports (the default unless a scenario says otherwise).
func (h *goldenHarness) blockSizeOf(id string) int {
	if b, ok := h.podBlk[id]; ok && b > 0 {
		return b
	}
	return goldenBlocks
}

// setBlockSize makes one pod report a different block size (the loud-mismatch fixture).
func (h *goldenHarness) setBlockSize(id string, n int) { h.podBlk[id] = n }

// goldenPods is the scheduling layer's view: every pod, with its metrics.
func goldenPods(pods map[string]int) []fwksched.Endpoint {
	ids := make([]string, 0, len(pods))
	for id := range pods {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]fwksched.Endpoint, 0, len(ids))
	for _, id := range ids {
		out = append(out, goldenEndpoint(id, pods[id], goldenBlocks))
	}
	return out
}

// preRequest runs the director's PreRequest call: a scoped request and the scheduling result
// naming the engine the framework actually chose (which may disagree with the seat).
func (h *goldenHarness) preRequest(req *fwksched.InferenceRequest, landing string) {
	scopedReq, violations := datalayer.ScopeRequest(logr.Discard(), fwkrc.PreRequestExtensionPoint, h.r, req)
	if err := violations.Write(); err != nil {
		h.t.Fatalf("PreRequest wrote an undeclared attribute: %v", err)
	}
	var res *fwksched.SchedulingResult
	if landing != "" {
		res = &fwksched.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{goldenEndpoint(landing, h.pods[landing], goldenBlocks)}},
			},
		}
	}
	if err := h.r.PreRequest(context.Background(), scopedReq, res); err != nil {
		h.t.Fatalf("PreRequest: %v", err)
	}
}

// firstChunk is ResponseBody at the start of the stream (the cold charge is released there).
func (h *goldenHarness) firstChunk(req *fwksched.InferenceRequest, at string) {
	scopedReq, violations := datalayer.ScopeRequest(logr.Discard(), "ResponseBody", h.r, req)
	if err := violations.Write(); err != nil {
		h.t.Fatalf("ResponseBody wrote an undeclared attribute: %v", err)
	}
	h.r.ResponseBody(context.Background(), scopedReq, &fwkrc.Response{
		RequestID: req.RequestID, StartOfStream: true,
	}, h.target(at))
}

// settle ends the stream: prompt/cached/completion usage, the termination cause, the endpoint.
func (h *goldenHarness) settle(req *fwksched.InferenceRequest, at string, prompt, cached, completion int) {
	scopedReq, violations := datalayer.ScopeRequest(logr.Discard(), "ResponseBody", h.r, req)
	if err := violations.Write(); err != nil {
		h.t.Fatalf("ResponseBody wrote an undeclared attribute: %v", err)
	}
	h.r.ResponseBody(context.Background(), scopedReq, &fwkrc.Response{
		RequestID: req.RequestID, EndOfStream: true,
		TerminationCause: fwkrc.TerminationCauseNatural,
		Usage: fwkrequest.Usage{
			PromptTokens:       prompt,
			PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: cached},
			CompletionTokens:   completion,
		},
	}, h.target(at))
}

func (h *goldenHarness) target(pod string) *fwkdl.EndpointMetadata {
	if pod == "" {
		return nil
	}
	return &fwkdl.EndpointMetadata{
		Name: pod, Address: pod + ":8000",
		ID: types.NamespacedName{Namespace: goldenNamespace, Name: pod},
	}
}

// settleCause settles with an explicit termination cause (the natural one is the common path).
func (h *goldenHarness) settleCause(req *fwksched.InferenceRequest, at string, cause fwkrc.TerminationCause) {
	scopedReq, violations := datalayer.ScopeRequest(logr.Discard(), "ResponseBody", h.r, req)
	if err := violations.Write(); err != nil {
		h.t.Fatalf("ResponseBody wrote an undeclared attribute: %v", err)
	}
	h.r.ResponseBody(context.Background(), scopedReq, &fwkrc.Response{
		RequestID: req.RequestID, EndOfStream: true, TerminationCause: cause,
		Usage: fwkrequest.Usage{PromptTokens: 300, CompletionTokens: 10},
	}, h.target(at))
}

// abandon runs RequestAbandoned: a request that left the path before reaching a pod.
func (h *goldenHarness) abandon(req *fwksched.InferenceRequest) {
	scopedReq, violations := datalayer.ScopeRequest(logr.Discard(), "RequestAbandoned", h.r, req)
	if err := violations.Write(); err != nil {
		h.t.Fatalf("RequestAbandoned wrote an undeclared attribute: %v", err)
	}
	h.r.RequestAbandoned(context.Background(), scopedReq)
}

// deletePod is the endpoint DELETE event: the engine's group is torn down.
// addPod brings a pod back under a name that was deleted: the readiness flap the reconciler makes
// on every not-ready state. The event carries no metrics (capacity is learned from the candidates),
// exactly as in production.
func (h *goldenHarness) addPod(id string, blocks int) {
	h.pods[id] = blocks
	if err := h.r.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: goldenDLEndpoint(id, 0, 0),
	}); err != nil {
		h.t.Fatalf("Extract add %s: %v", id, err)
	}
}

func (h *goldenHarness) deletePod(id string) {
	if err := h.r.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventDelete, Endpoint: goldenDLEndpoint(id, 0, 0),
	}); err != nil {
		h.t.Fatalf("Extract delete %s: %v", id, err)
	}
	delete(h.pods, id)
}

// request builds one inference request the way thread identity would leave it: the derived
// thread id in FairnessID and in its attribute, the arrival record, and the rendered tokens.
func (h *goldenHarness) request(turn int, promptLens ...int) *fwksched.InferenceRequest {
	h.seq++
	id := fmt.Sprintf("r%02d", h.seq)
	arrays := make([][]uint32, len(promptLens))
	for i, n := range promptLens {
		// a fresh seed per request, so two unrelated requests never share a prefix by accident
		arrays[i] = growTokens(h.seq*10000, n)
	}
	req := &fwksched.InferenceRequest{RequestID: id}
	req.Body = &fwkrequest.InferenceRequestBody{TokenizedRequest: fwkrequest.NewTokenizedRequest(arrays)}
	return req
}

// identify stamps the request with the thread identity the request header processor produces.
func (h *goldenHarness) identify(req *fwksched.InferenceRequest, thread string, turn int) {
	req.FairnessID = thread
	req.PutAttribute(threadidentity.ThreadIdentityKey, thread)
	outcome := threadidentity.OutcomeNew
	if turn > 1 {
		outcome = threadidentity.OutcomeMatched
	}
	req.PutAttribute(threadidentity.ThreadArrivalKey, threadidentity.ThreadArrival{Outcome: outcome, Turn: turn})
	// the blocks thread identity would publish for these tokens: the heads are charged from them
	if body := req.Body; body != nil && body.TokenizedRequest != nil {
		req.PutAttribute(threadidentity.ThreadBlocksKey, blocksOfRequest(req, h.r.cfg.BlockSizeTokens))
	}
}

// growTokens makes n tokens starting from a seed, so a later turn's prompt extends an earlier one.
func growTokens(seed, n int) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = uint32(seed + i)
	}
	return out
}

// digest is the per-decision state line: read straight from the scheduler's internals by ONE
// helper, so a later rename touches only this function.
func (h *goldenHarness) digest() {
	s := h.r.sched
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := make([]string, 0, len(s.groups))
	for id := range s.groups {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		g := s.groups[id]
		used, stay, exiting := 0, 0, 0
		var cold int64
		threads := make([]string, 0, len(g.threads))
		for b := range g.threads {
			threads = append(threads, b)
		}
		sort.Strings(threads)
		for _, b := range threads {
			if g.index.Exiting(b) {
				exiting++
			}
			cold += g.threads[b].coldEntry
		}
		used = g.index.UniqueBlocks()
		// THE BINDINGS INVARIANT, checked after every pass of every scenario: a thread is bound
		// exactly while it is seated, so an engine's bindings equal its seats.
		bound := 0
		for _, e := range s.bindings {
			if e == id {
				bound++
			}
		}
		if bound != len(g.threads) {
			h.t.Fatalf("%s: %d seats but %d bindings", id, len(g.threads), bound)
		}
		// ...and the BINDINGS are the only engine each thread names: a thread is bound exactly while
		// it is seated, so no binding may name a thread this engine does not hold.
		for b, eng := range s.bindings {
			if eng != id {
				continue
			}
			if _, seated := g.threads[b]; !seated {
				h.t.Fatalf("%s: binding for %q, which is not seated here", id, b)
			}
		}
		// RECORD ⇒ SEATED, checked after every pass of every scenario: a record exists only while
		// its thread holds a seat (a release requires one, and every unseat sweeps the records).
		for reqID, rec := range s.reqs {
			g2 := s.groupOf(rec.thread)
			if g2 == nil {
				h.t.Fatalf("record %q names thread %q, which belongs to no engine", reqID, rec.thread)
			}
			if _, seated := g2.threads[rec.thread]; !seated {
				h.t.Fatalf("record %q names thread %q, which holds no seat", reqID, rec.thread)
			}
		}
		stay = g.index.StayBlocks()
		cold = g.coldInFlight // the value the valve reads, not the sum of the threads' charges
		// THE ROTATION UNIT, per seated thread: completed busy periods are what the quantum rule
		// counts, and a trace that cannot see them cannot see the rule. Thread-keyed and sorted, so
		// the line stays deterministic.
		periods := make([]string, 0, len(threads))
		for _, b := range threads {
			periods = append(periods, fmt.Sprintf("%s:%d", b, g.threads[b].busyPeriods))
			// THE PENDING INVARIANT (B6): a thread's pending count IS the number of its records that
			// currently count. A count that drifts from the records is exactly how a dead window kept
			// a thread busy across passes.
			counted := 0
			for _, rec := range h.r.sched.reqs {
				if rec.thread == b && rec.counted {
					counted++
				}
			}
			if m := g.threads[b]; m.pending != counted {
				h.t.Fatalf("thread %s: pending=%d but %d of its records count — the count has drifted from the records", b, m.pending, counted)
			}
		}
		parts = append(parts, fmt.Sprintf("%s{seats=%d used=%d stay=%d exiting=%d cold=%d budget=%d periods=%s}",
			id, len(g.threads), used, stay, exiting, cold, g.budgetBlocks, strings.Join(periods, ",")))
	}
	kinds := make([]string, 0, len(s.counters.anomalies))
	for k := range s.counters.anomalies {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	anom := make([]string, 0, len(kinds))
	for _, k := range kinds {
		anom = append(anom, fmt.Sprintf("%s=%d", k, s.counters.anomalies[k]))
	}
	h.line("STATE %s waiters=%d recs=%d anomalies[%s]",
		strings.Join(parts, " "), len(s.waiters), len(s.reqs), strings.Join(anom, ","))
}

// trace is the accumulated decision trace.
func (h *goldenHarness) trace() string { return h.tr.String() }
