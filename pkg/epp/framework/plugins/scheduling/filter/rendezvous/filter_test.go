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

package rendezvous

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/jellydator/ttlcache/v3"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/endpointattribute"
	rendezvousutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/rendezvous"
	"github.com/llm-d/llm-d-router/test/utils"
)

var (
	t0    = time.Unix(1_800_000_000, 0)
	fresh = t0.Add(-time.Second) // metrics scraped a second ago
)

// cluster builds a cluster-endpoint with labels and, when updatedAt is not
// zero, a metrics-updated-at attribute.
func cluster(name string, labels map[string]string, updatedAt time.Time) scheduling.Endpoint {
	return clusterWith(name, labels, updatedAt, fwkdl.NewAttributes())
}

func clusterWith(name string, labels map[string]string, updatedAt time.Time, attrs fwkdl.AttributeMap) scheduling.Endpoint {
	if !updatedAt.IsZero() {
		attrs.Put(attrmetrics.MultiClusterMetricsUpdatedAtDataKey, attrmetrics.ScalarMetricValue(float64(updatedAt.Unix())))
	}
	return scheduling.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}, Labels: labels},
		&fwkdl.Metrics{}, attrs)
}

// shared builds a live cluster whose pool reports share ready pods ("" = not reported).
func shared(name, share string) scheduling.Endpoint {
	attrs := fwkdl.NewAttributes()
	if share != "" {
		ready, err := strconv.ParseFloat(share, 64)
		if err != nil {
			panic(err)
		}
		attrs.Put(attrmetrics.MultiClusterReadyEndpointsDataKey, attrmetrics.ScalarMetricValue(ready))
	}
	return clusterWith(name, nil, fresh, attrs)
}

// fullGuard keeps every endpoint not named in full.
type fullGuard struct{ full map[string]bool }

func (g *fullGuard) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "test-guard", Name: "test-guard"}
}

func (g *fullGuard) Filter(_ context.Context, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	var kept []scheduling.Endpoint
	for _, endpoint := range endpoints {
		if !g.full[endpoint.GetMetadata().ID.Name] {
			kept = append(kept, endpoint)
		}
	}
	return kept
}

func newGuard(full ...string) *fullGuard {
	g := &fullGuard{full: map[string]bool{}}
	for _, name := range full {
		g.full[name] = true
	}
	return g
}

func withHeaders(kv ...string) *scheduling.InferenceRequest {
	headers := map[string]string{}
	for i := 0; i+1 < len(kv); i += 2 {
		headers[kv[i]] = kv[i+1]
	}
	return &scheduling.InferenceRequest{Headers: headers}
}

func withSession(id string) *scheduling.InferenceRequest { return withHeaders("x-session-id", id) }

func chat(messages ...fwkrequest.Message) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		Headers: map[string]string{},
		Body:    &fwkrequest.InferenceRequestBody{ChatCompletions: &fwkrequest.ChatCompletionsRequest{Messages: messages}},
	}
}

func msg(role, text string) fwkrequest.Message {
	return fwkrequest.Message{Role: role, Content: fwkrequest.Content{Raw: text}}
}

func baseConfig() Config { return Config{} }

func noStick() Config {
	zero := 0.0
	return Config{StickTTLSeconds: &zero}
}

func newFilter(t *testing.T, config Config, guard scheduling.Filter) *Filter {
	t.Helper()
	f, err := New(t.Context(), "test", config, guard)
	if err != nil {
		t.Fatal(err)
	}
	f.now = func() time.Time { return t0 }
	return f
}

// pick runs the filter like the router does: a picker takes the first kept
// endpoint by name, then PreRequest runs (it records the pick of a spill).
func pick(t *testing.T, f *Filter, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) string {
	t.Helper()
	pickSeq++
	if request.RequestID == "" {
		request.RequestID = fmt.Sprintf("req-%d", pickSeq)
	}
	got := f.Filter(t.Context(), request, endpoints)
	if len(got) == 0 {
		t.Fatal("filter kept no endpoint")
	}
	sort.Slice(got, func(i, j int) bool { return got[i].GetMetadata().ID.Name < got[j].GetMetadata().ID.Name })
	if err := f.PreRequest(t.Context(), request, picked(got[0].GetMetadata().ID.Name)); err != nil {
		t.Fatal(err)
	}
	return got[0].GetMetadata().ID.Name
}

var pickSeq int

// hasPending reports whether a spill of requestID waits for its PreRequest.
func hasPending(f *Filter, requestID string) bool {
	_, err := plugin.ReadPluginStateKey[*spill](f.pending, requestID, spillStateKey)
	return err == nil
}

// sessionOnA returns a session id whose hash picks endpoint "a".
func sessionOnA(t *testing.T, f *Filter, endpoints []scheduling.Endpoint) string {
	t.Helper()
	for i := range 1000 {
		id := fmt.Sprintf("s-%d", i)
		if pick(t, f, withSession(id), endpoints) == "a" {
			return id
		}
	}
	t.Fatal("no session hashes to a")
	return ""
}

func TestSameSessionSameEndpoint(t *testing.T) {
	f := newFilter(t, baseConfig(), nil)
	endpoints := []scheduling.Endpoint{shared("main", "3"), shared("small", "1")}
	first := pick(t, f, withSession("s-1"), endpoints)
	for range 20 {
		if got := pick(t, f, withSession("s-1"), endpoints); got != first {
			t.Fatalf("session moved: %s then %s", first, got)
		}
	}
	// Another instance (hub replica, or after a restart) agrees.
	if got := pick(t, newFilter(t, baseConfig(), nil), withSession("s-1"), endpoints); got != first {
		t.Fatalf("second instance picked %s, first picked %s", got, first)
	}
}

func TestConversationTurnsStayTogether(t *testing.T) {
	f := newFilter(t, baseConfig(), nil)
	endpoints := []scheduling.Endpoint{shared("main", "1"), shared("small", "1")}
	turn1 := chat(msg("system", "sys"), msg("user", "task 42"))
	turn3 := chat(msg("system", "sys"), msg("user", "task 42"), msg("assistant", "ok"), msg("user", "next"))
	if a, b := pick(t, f, turn1, endpoints), pick(t, f, turn3, endpoints); a != b {
		t.Fatalf("turns of one conversation split: %s vs %s", a, b)
	}
	seen := map[string]bool{}
	for i := range 50 {
		seen[pick(t, f, chat(msg("system", "sys"), msg("user", fmt.Sprintf("task %d", i))), endpoints)] = true
	}
	if len(seen) != 2 {
		t.Fatalf("50 conversations with a shared system prompt all landed on %v", seen)
	}
}

// TestSharesHold: clusters take keys in proportion to their ready pods.
func TestSharesHold(t *testing.T) {
	f := newFilter(t, baseConfig(), nil)
	endpoints := []scheduling.Endpoint{shared("main", "3"), shared("small", "1")}
	const n = 20000
	onMain := 0
	for i := range n {
		if pick(t, f, withSession(fmt.Sprintf("s-%d", i)), endpoints) == "main" {
			onMain++
		}
	}
	if got := float64(onMain) / n; got < 0.73 || got > 0.77 {
		t.Fatalf("main share = %.3f, want ~0.75", got)
	}
}

func TestReadyPodsWeight(t *testing.T) {
	f := newFilter(t, baseConfig(), nil)
	// 0 ready pods excludes; a pool that does not report them weighs 1.
	endpoints := []scheduling.Endpoint{shared("zero", "0"), shared("missing", ""), shared("one", "1")}
	for i := range 500 {
		if got := pick(t, f, withSession(fmt.Sprintf("s-%d", i)), endpoints); got == "zero" {
			t.Fatal("endpoint with 0 ready pods was picked")
		}
	}
	// No ready pods anywhere: nothing to choose by, keep the candidates.
	if got := f.Filter(t.Context(), withSession("s"), []scheduling.Endpoint{shared("a", "0"), shared("b", "0")}); len(got) != 2 {
		t.Fatalf("all-zero weights filtered to %d endpoints", len(got))
	}
}

func TestFullClusterMovesOnlyItsKeys(t *testing.T) {
	guard := newGuard()
	f := newFilter(t, noStick(), guard) // no stick: keys return when room comes back
	all := []scheduling.Endpoint{shared("main", "6"), shared("small-a", "1"), shared("small-b", "1")}
	moved, stayed := 0, 0
	for i := range 5000 {
		r := withSession(fmt.Sprintf("s-%d", i))
		guard.full = map[string]bool{}
		before := pick(t, f, r, all)
		guard.full = map[string]bool{"small-a": true}
		after := pick(t, f, r, all)
		guard.full = map[string]bool{}
		back := pick(t, f, r, all)
		if before == "small-a" {
			moved++
			if after == "small-a" {
				t.Fatal("key stayed on a full endpoint")
			}
		} else {
			stayed++
			if after != before {
				t.Fatalf("key on %s moved to %s when only small-a was full", before, after)
			}
		}
		if back != before {
			t.Fatalf("without a TTL the key should return: %s then %s", before, back)
		}
	}
	if moved == 0 || stayed == 0 {
		t.Fatalf("degenerate split: moved=%d stayed=%d", moved, stayed)
	}
}

func TestPins(t *testing.T) {
	f := newFilter(t, baseConfig(), nil)
	endpoints := []scheduling.Endpoint{
		cluster("secondary", map[string]string{"pins": "tenant-c"}, time.Time{}),
		cluster("main", map[string]string{"pins": "tenant-a, tenant-b"}, time.Time{}),
	}
	for i := range 50 {
		session := fmt.Sprintf("s-%d", i)
		for tenant, want := range map[string]string{"tenant-a": "main", "tenant-b": "main", "tenant-c": "secondary"} {
			if got := pick(t, f, withHeaders("x-session-id", session, "x-llm-d-inference-fairness-id", tenant), endpoints); got != want {
				t.Fatalf("pinned tenant %s went to %s", tenant, got)
			}
		}
	}
	// The deprecated header name matches too; the objective does not pin.
	if got := pick(t, f, withHeaders("x-session-id", "s", "x-gateway-inference-fairness-id", "tenant-c"), endpoints); got != "secondary" {
		t.Fatalf("deprecated tenant header pin went to %s", got)
	}
	split := map[string]int{}
	for i := range 200 {
		split[pick(t, f, withHeaders("x-session-id", fmt.Sprintf("s-%d", i), "x-llm-d-inference-objective", "tenant-a"), endpoints)]++
	}
	if split["secondary"] == 0 || split["main"] == 0 {
		t.Fatalf("objective acted as a pin: %v", split)
	}
}

func TestPinnedTenantSpillsWhenFull(t *testing.T) {
	guard := newGuard("main")
	f := newFilter(t, baseConfig(), guard)
	endpoints := []scheduling.Endpoint{
		cluster("main", map[string]string{"pins": "tenant-a"}, fresh),
		cluster("secondary", map[string]string{}, fresh),
	}
	r := withHeaders("x-session-id", "conv-1", "x-llm-d-inference-fairness-id", "tenant-a")
	if got := pick(t, f, r, endpoints); got != "secondary" {
		t.Fatalf("pinned tenant with full main went to %s", got)
	}
	// main recovers: the spilled conversation stays, a new one follows the pin.
	guard.full = map[string]bool{}
	if got := pick(t, f, r, endpoints); got != "secondary" {
		t.Fatalf("spilled conversation went back to %s before its TTL", got)
	}
	if got := pick(t, f, withHeaders("x-session-id", "conv-2", "x-llm-d-inference-fairness-id", "tenant-a"), endpoints); got != "main" {
		t.Fatalf("new conversation of a pinned tenant went to %s", got)
	}
}

func TestAllFullKeepsDefault(t *testing.T) {
	f := newFilter(t, baseConfig(), newGuard("a", "b"))
	endpoints := []scheduling.Endpoint{shared("a", "1"), shared("b", "1")}
	for i := range 200 {
		s := fmt.Sprintf("s-%d", i)
		want := pick(t, newFilter(t, baseConfig(), nil), withSession(s), endpoints)
		if got := pick(t, f, withSession(s), endpoints); got != want {
			t.Fatalf("all full: %s moved from %s to %s", s, want, got)
		}
	}
	if n := countMoved(f); n != 0 {
		t.Fatalf("all full recorded %d moves", n)
	}
}

func TestDeadClusterFailsOver(t *testing.T) {
	config := baseConfig()
	stale := t0.Add(-5 * time.Minute)
	live := []scheduling.Endpoint{cluster("a", nil, fresh), cluster("b", nil, fresh)}
	s := sessionOnA(t, newFilter(t, config, newGuard()), live)

	// a stopped answering: its last metrics are old. Even with b full, b beats a dead cluster.
	f := newFilter(t, config, newGuard("b"))
	deadA := []scheduling.Endpoint{cluster("a", nil, stale), cluster("b", nil, fresh)}
	if got := pick(t, f, withSession(s), deadA); got != "b" {
		t.Fatalf("dead cluster kept traffic: %s", got)
	}
	// A cluster never scraped counts as dead too.
	never := []scheduling.Endpoint{cluster("a", nil, time.Time{}), cluster("b", nil, fresh)}
	if got := pick(t, newFilter(t, config, newGuard()), withSession(s), never); got != "b" {
		t.Fatalf("never-scraped cluster got traffic: %s", got)
	}
	// Nothing alive at all: keep the default rather than drop the request.
	allDead := []scheduling.Endpoint{cluster("a", nil, stale), cluster("b", nil, stale)}
	if got := pick(t, newFilter(t, config, newGuard()), withSession(s), allDead); got != "a" {
		t.Fatalf("all dead: went to %s, want default a", got)
	}
}

func TestMaxMetricsAge(t *testing.T) {
	config := baseConfig()
	live := []scheduling.Endpoint{cluster("a", nil, fresh), cluster("b", nil, fresh)}
	s := sessionOnA(t, newFilter(t, config, newGuard()), live)
	aged := []scheduling.Endpoint{cluster("a", nil, t0.Add(-15*time.Second)), cluster("b", nil, fresh)}
	// 15s old is alive under the 30s default, down under 10s.
	if got := pick(t, newFilter(t, config, newGuard()), withSession(s), aged); got != "a" {
		t.Fatalf("default age: went to %s, want a", got)
	}
	ten := 10.0
	config.MaxMetricsAgeSeconds = &ten
	if got := pick(t, newFilter(t, config, newGuard()), withSession(s), aged); got != "b" {
		t.Fatalf("maxMetricsAgeSeconds 10: went to %s, want b", got)
	}
}

func TestFactory(t *testing.T) {
	handle := utils.NewTestHandle(t.Context())
	handle.AddPlugin("not-overloaded", newGuard())

	build := func(params string) (*Filter, error) {
		p, err := MultiClusterFactory("hub", json.NewDecoder(strings.NewReader(params)), handle)
		if err != nil {
			return nil, err
		}
		return p.(*Filter), nil
	}
	f, err := build(`{"guardPluginName":"not-overloaded"}`)
	if err != nil {
		t.Fatal(err)
	}
	if f.TypedName().Type != MultiClusterFilterType || f.TypedName().Name != "hub" || f.guard == nil ||
		f.maxAge != defaultMaxMetricsAgeSeconds || stickTTL(f) != defaultStickTTLSeconds*time.Second {
		t.Fatalf("defaults with a guard: %+v", f)
	}
	if _, ok := f.Consumes().Required[attrmetrics.MultiClusterMetricsUpdatedAtDataKey]; !ok {
		t.Fatal("staleness on but metrics-updated-at not required")
	}
	if f, err = build(`{"guardPluginName":"not-overloaded","stickTtlSeconds":600}`); err != nil || stickTTL(f) != 600*time.Second {
		t.Fatalf("stickTtlSeconds 600: %v %v", f, err)
	}
	if f, err = build(`{"guardPluginName":"not-overloaded","stickTtlSeconds":0}`); err != nil || stickTTL(f) != 0 {
		t.Fatalf("stickTtlSeconds 0: %v %v", f, err)
	}
	if f, err = build(`{"maxMetricsAgeSeconds":10}`); err != nil || f.maxAge != 10 {
		t.Fatalf("maxMetricsAgeSeconds 10: %v %v", f, err)
	}
	// No parameters: the built-in queue guard, staleness and spill memory are on.
	if f, err = build(`{}`); err != nil || f.guard == nil || f.guard.TypedName().Name != defaultGuardName ||
		f.maxAge != defaultMaxMetricsAgeSeconds || stickTTL(f) != defaultStickTTLSeconds*time.Second {
		t.Fatalf("empty config: %+v %v", f, err)
	}
	if _, ok := f.Consumes().Optional[attrmetrics.MultiClusterQueueSizeDataKey]; !ok {
		t.Fatal("default guard's queue size not consumed")
	}
	for name, bad := range map[string]string{
		"limits with a custom guard": `{"guardPluginName":"not-overloaded","maxWaitSeconds":5}`,
		"negative maxQueue":          `{"maxQueue":-1}`,
		"negative maxWaitSeconds":    `{"maxWaitSeconds":-1}`,
		"unknown guard":              `{"guardPluginName":"missing"}`,
		"negative stick TTL":         `{"guardPluginName":"not-overloaded","stickTtlSeconds":-1}`,
		"stick TTL past a year":      `{"guardPluginName":"not-overloaded","stickTtlSeconds":1e12}`,
		"maxQueue 0, always full":    `{"maxQueue":0}`,
		"maxWaitSeconds 0":           `{"maxWaitSeconds":0}`,
		"wrong type stick TTL":       `{"stickTtlSeconds":"long"}`,
		"maxMetricsAgeSeconds 0":     `{"maxMetricsAgeSeconds":0}`,
		"negative maxMetricsAge":     `{"maxMetricsAgeSeconds":-1}`,
	} {
		if _, err := build(bad); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func countMoved(f *Filter) int {
	if f.moved == nil {
		return 0
	}
	return f.moved.Len()
}

// stickTTL reads the moved table's TTL (0 when spilled keys are not remembered).
func stickTTL(f *Filter) time.Duration {
	if f.moved == nil {
		return 0
	}
	probe := ^uint64(0)
	ttl := f.moved.Set(probe, "", ttlcache.DefaultTTL).TTL()
	f.moved.Delete(probe)
	return ttl
}

func names(endpoints []scheduling.Endpoint) []string {
	out := make([]string, 0, len(endpoints))
	for _, endpoint := range endpoints {
		out = append(out, endpoint.GetMetadata().ID.Name)
	}
	return out
}

// TestRealGuardOnScopedEndpoints runs the filter the way the scheduler does: on
// endpoints scoped to the filter's declared keys, with the real queue guard.
// The guard reads the queue size through that scope, so the filter must declare
// what the guard consumes; otherwise every cluster looks full and nothing spills.
func TestRealGuardOnScopedEndpoints(t *testing.T) {
	guard, err := endpointattribute.EndpointAttributeFilterFactory("not-overloaded", json.NewDecoder(strings.NewReader(
		`{"attribute":"llm-d.ai/multicluster-queue-size","producer":"","onMissing":"Fail",
		  "algorithm":{"type":"threshold","threshold":{"operator":"LessThan","value":4}}}`)), nil)
	if err != nil {
		t.Fatal(err)
	}
	f := newFilter(t, baseConfig(), guard.(scheduling.Filter))
	withQueue := func(name string, queue float64) scheduling.Endpoint {
		e := cluster(name, nil, t0.Add(-time.Second))
		e.Put(attrmetrics.MultiClusterQueueSizeDataKey, attrmetrics.ScalarMetricValue(queue))
		return e
	}
	idle := []scheduling.Endpoint{withQueue("a", 0), withQueue("b", 0)}
	s := sessionOnA(t, f, idle)
	busyA := []scheduling.Endpoint{withQueue("a", 30), withQueue("b", 0)}
	scoped, _ := datalayer.Scope(logr.Discard(), "Filter", f, busyA)
	got := f.Filter(t.Context(), withSession(s), scoped)
	if len(got) != 1 || got[0].GetMetadata().ID.Name != "b" {
		t.Fatalf("session on overloaded a went to %v, want b", names(got))
	}
}

// TestRoomCheckLimits covers the built-in room check: the vLLM queue for a
// cluster without flow control, and per-priority flow-control waits otherwise.
func TestRoomCheckLimits(t *testing.T) {
	type band = attrmetrics.FlowControlBand
	now := float64(t0.Unix())
	ok := band{Held: 2, RecentWait: 1, LastProgress: now - 2}         // moving, short waits
	slow := band{Held: 30, RecentWait: 12, LastProgress: now - 1}     // waits over 10 s
	stalled := band{Held: 200, RecentWait: 0, LastProgress: now - 60} // nothing left for a minute
	endpoint := func(name string, queue float64, bands attrmetrics.MultiClusterFlowControl) scheduling.Endpoint {
		e := cluster(name, nil, fresh)
		if queue >= 0 { // -1: no queue metric
			e.Put(attrmetrics.MultiClusterQueueSizeDataKey, attrmetrics.ScalarMetricValue(queue))
		}
		if bands != nil {
			e.Put(attrmetrics.MultiClusterFlowControlDataKey, bands)
		}
		return e
	}
	for name, tc := range map[string]struct {
		params   string
		priority int
		queue    float64
		bands    attrmetrics.MultiClusterFlowControl
		want     string
	}{
		"no flow control, idle":                {`{}`, 60, 0, nil, "a"},
		"no flow control, below limit":         {`{}`, 60, defaultMaxQueue - 1, nil, "a"},
		"no flow control, queue at limit":      {`{}`, 60, defaultMaxQueue, nil, "b"},
		"no flow control, no queue metric":     {`{}`, 60, -1, nil, "b"},
		"maxQueue raised":                      {`{"maxQueue":6}`, 60, 5, nil, "a"},
		"maxQueue lowered":                     {`{"maxQueue":2}`, 60, 2, nil, "b"},
		"flow control healthy":                 {`{}`, 60, 0, attrmetrics.MultiClusterFlowControl{60: ok, 40: ok}, "a"},
		"low band stalled, high request":       {`{}`, 60, 0, attrmetrics.MultiClusterFlowControl{60: ok, 40: stalled}, "a"},
		"low band stalled, low request":        {`{}`, 40, 0, attrmetrics.MultiClusterFlowControl{60: ok, 40: stalled}, "b"},
		"lower band ignored for mid request":   {`{}`, 40, 0, attrmetrics.MultiClusterFlowControl{60: ok, 20: stalled}, "a"},
		"own band waits too long":              {`{}`, 60, 0, attrmetrics.MultiClusterFlowControl{60: slow}, "b"},
		"higher band backs up a low request":   {`{}`, 20, 0, attrmetrics.MultiClusterFlowControl{60: slow, 20: ok}, "b"},
		"own band stalled":                     {`{}`, 60, 0, attrmetrics.MultiClusterFlowControl{60: {Held: 5, LastProgress: now - 20}}, "b"},
		"maxWaitSeconds raised":                {`{"maxWaitSeconds":30}`, 60, 0, attrmetrics.MultiClusterFlowControl{60: slow}, "a"},
		"vLLM queue ignored with flow control": {`{}`, 60, 9, attrmetrics.MultiClusterFlowControl{60: ok}, "a"},
	} {
		t.Run(name, func(t *testing.T) {
			p, err := MultiClusterFactory("hub", json.NewDecoder(strings.NewReader(tc.params)), nil)
			if err != nil {
				t.Fatal(err)
			}
			f := p.(*Filter)
			f.now = func() time.Time { return t0 }
			f.guard.(*roomCheck).now = func() time.Time { return t0 }
			idle := []scheduling.Endpoint{endpoint("a", 0, nil), endpoint("b", 0, nil)}
			s := sessionOnA(t, f, idle)
			f.moved.DeleteAll()
			busy := []scheduling.Endpoint{endpoint("a", tc.queue, tc.bands), endpoint("b", 0, nil)}
			scoped, _ := datalayer.Scope(logr.Discard(), "Filter", f, busy)
			request := withSession(s)
			request.Objectives.Priority = tc.priority
			got := f.Filter(t.Context(), request, scoped)
			if len(got) != 1 || got[0].GetMetadata().ID.Name != tc.want {
				t.Fatalf("got %v, want %s", names(got), tc.want)
			}
		})
	}
}

// picked is a scheduling result whose primary profile picked name.
func picked(name string) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{PrimaryProfileName: "default", ProfileResults: map[string]*scheduling.ProfileRunResult{
		"default": {TargetEndpoints: []scheduling.Endpoint{cluster(name, nil, fresh)}}}}
}

// TestSpillPickedByScorers: the scorers pick the spill cluster, PreRequest remembers it.
func TestSpillPickedByScorers(t *testing.T) {
	guard := newGuard()
	f := newFilter(t, baseConfig(), guard)
	endpoints := []scheduling.Endpoint{shared("a", "1"), shared("b", "1"), shared("c", "1")}
	s := sessionOnA(t, f, endpoints)
	turn := func(id string) *scheduling.InferenceRequest {
		r := withSession(s)
		r.RequestID = id
		return r
	}

	// a has room: only a, nothing pending.
	if got := pick(t, f, turn("r0"), endpoints); got != "a" {
		t.Fatalf("home with room went to %s", got)
	}
	if hasPending(f, "r0") {
		t.Fatal("a request kept home left a pending spill")
	}

	// a full: the scorers get b and c; the picker's choice is remembered.
	guard.full = map[string]bool{"a": true}
	if got := names(f.Filter(t.Context(), turn("r1"), endpoints)); !reflect.DeepEqual(got, []string{"b", "c"}) {
		t.Fatalf("spill by score kept %v, want [b c]", got)
	}
	if err := f.PreRequest(t.Context(), turn("r1"), picked("c")); err != nil {
		t.Fatal(err)
	}
	if hasPending(f, "r1") || countMoved(f) != 1 {
		t.Fatalf("after PreRequest: pending %v, %d moved", hasPending(f, "r1"), countMoved(f))
	}
	// Later turns stay on c, also after a recovers.
	for i, full := range []map[string]bool{{"a": true}, {}} {
		guard.full = full
		if got := pick(t, f, turn(fmt.Sprintf("r2-%d", i)), endpoints); got != "c" {
			t.Fatalf("turn %d went to %s, want c", i, got)
		}
	}

	// A PreRequest without a pending spill changes nothing.
	if err := f.PreRequest(t.Context(), turn("unknown"), picked("b")); err != nil {
		t.Fatal(err)
	}
	guard.full = map[string]bool{}
	if got := pick(t, f, turn("r3"), endpoints); got != "c" {
		t.Fatalf("unknown request moved the key to %s", got)
	}

	// No cluster has room: keep the hashed cluster, as with spill by hash.
	guard.full = map[string]bool{"a": true, "b": true, "c": true}
	if got := pick(t, f, withSession("other"), endpoints); got == "" {
		t.Fatal("no endpoint with every cluster full")
	}
}

// TestNoKey: a request with nothing to key on (the tenant header alone does not
// key it) goes to its pinned cluster while that has room, else keeps the
// clusters with room, and leaves no pending spill.
func TestNoKey(t *testing.T) {
	const fairness = "x-llm-d-inference-fairness-id"
	for name, tc := range map[string]struct {
		request *scheduling.InferenceRequest
		full    []string
		want    []string
	}{
		"keeps clusters with room":  {withHeaders(), []string{"a"}, []string{"b", "c"}},
		"all full keeps all":        {withHeaders(), []string{"a", "b", "c"}, []string{"a", "b", "c"}},
		"tenant alone does not key": {withHeaders(fairness, "tenant-b"), []string{"a"}, []string{"b", "c"}},
		"pin with room":             {withHeaders(fairness, "tenant-a"), nil, []string{"a"}},
		"full pin":                  {withHeaders(fairness, "tenant-a"), []string{"a"}, []string{"b", "c"}},
	} {
		t.Run(name, func(t *testing.T) {
			f := newFilter(t, baseConfig(), newGuard(tc.full...))
			endpoints := []scheduling.Endpoint{cluster("a", map[string]string{"pins": "tenant-a"}, fresh), shared("b", "1"), shared("c", "1")}
			tc.request.RequestID = "no-key"
			got := names(f.Filter(t.Context(), tc.request, endpoints))
			sort.Strings(got)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("kept %v, want %v", got, tc.want)
			}
			if hasPending(f, "no-key") {
				t.Fatal("keyless request left a pending spill")
			}
		})
	}
}

// TestZeroReadyPodsHasNoRoom: a cluster with no ready pods takes no pinned and
// no spilled conversations, even when its stale queue metric reads as room, and
// also when no cluster has room.
func TestZeroReadyPodsHasNoRoom(t *testing.T) {
	for name, full := range map[string][]string{"room elsewhere": nil, "no room anywhere": {"main", "secondary"}} {
		t.Run(name, func(t *testing.T) {
			f := newFilter(t, baseConfig(), newGuard(full...))
			zero := shared("main", "0")
			zero.GetMetadata().Labels = map[string]string{"pins": "tenant-a"}
			endpoints := []scheduling.Endpoint{zero, shared("secondary", "2")}
			if got := pick(t, f, withHeaders("x-session-id", "s", "x-llm-d-inference-fairness-id", "tenant-a"), endpoints); got != "secondary" {
				t.Fatalf("pinned tenant went to a cluster with no ready pods: %s", got)
			}
			key, _ := rendezvousutil.Key(t.Context(), withSession("s2"), sessionSources)
			f.moved.Set(key, "/main", ttlcache.DefaultTTL)
			if got := pick(t, f, withSession("s2"), endpoints); got != "secondary" {
				t.Fatalf("spilled key went to a cluster with no ready pods: %s", got)
			}
		})
	}
}

// TestSpillLifecycle follows one session through spill, stay, no room anywhere,
// return home and expiry. Each step sets the full clusters, then picks.
func TestSpillLifecycle(t *testing.T) {
	guard := newGuard()
	second := 1.0
	f := newFilter(t, Config{StickTTLSeconds: &second}, guard)
	endpoints := []scheduling.Endpoint{shared("a", "1"), shared("b", "1")}
	s := sessionOnA(t, f, endpoints)
	for i, step := range []struct {
		full []string
		want string
		why  string
	}{
		{[]string{"a"}, "b", "a full: spill to b, remembered"},
		{nil, "b", "a recovers: stays on b"},
		{[]string{"a", "b"}, "b", "no room anywhere: stays on b"},
		{[]string{"a"}, "b", "b has room again: entry kept"},
		{[]string{"b"}, "a", "b full, a has room: home, entry dropped"},
		{nil, "a", "entry dropped"},
		{[]string{"a"}, "b", "spill again"},
	} {
		guard.full = newGuard(step.full...).full
		if got := pick(t, f, withSession(s), endpoints); got != step.want {
			t.Fatalf("step %d (%s): went to %s, want %s", i, step.why, got, step.want)
		}
	}
	guard.full = map[string]bool{}
	// Wait for the entry to expire, reading it without refreshing its TTL.
	key, _ := rendezvousutil.Key(t.Context(), withSession(s), sessionSources)
	expired := func() bool {
		return f.moved.Get(key, ttlcache.WithDisableTouchOnHit[uint64, string]()) == nil
	}
	for deadline := time.Now().Add(5 * time.Second); !expired(); time.Sleep(50 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("moved entry never expired")
		}
	}
	if got := pick(t, f, withSession(s), endpoints); got != "a" {
		t.Fatalf("expired entry still applied: went to %s", got)
	}
}

// TestReadyNotReportedWeighsDefault: a pool that stopped reporting ready pods
// (the extractor writes -1) weighs defaultWeight and still takes its share.
func TestReadyNotReportedWeighsDefault(t *testing.T) {
	f := newFilter(t, baseConfig(), nil)
	endpoints := []scheduling.Endpoint{shared("unreported", "-1"), shared("one", "1")}
	const n = 4000
	got := 0
	for i := range n {
		if pick(t, f, withSession(fmt.Sprintf("s-%d", i)), endpoints) == "unreported" {
			got++
		}
	}
	if share := float64(got) / n; share < 0.45 || share > 0.55 {
		t.Fatalf("unreported share = %.3f, want ~0.5", share)
	}
}

// TestSessionKeySources: agent-identity's attribute keys the conversation before
// x-session-id; without either, its prefix does. The attribute name matches the
// plugin's.
func TestSessionKeySources(t *testing.T) {
	if plugin.NewDataKey(agentIdentityAttribute, "") != agentidentity.AgentIdentityKey {
		t.Fatalf("attribute %q is not the key agent-identity publishes", agentIdentityAttribute)
	}
	key := func(r *scheduling.InferenceRequest) (uint64, rendezvousutil.Source) {
		return rendezvousutil.Key(t.Context(), r, sessionSources)
	}
	withAgent := func(id string, kv ...string) *scheduling.InferenceRequest {
		r := withHeaders(kv...)
		r.PutAttribute(agentidentity.AgentIdentityKey, id)
		return r
	}
	both, src := key(withAgent("A", "x-session-id", "D"))
	if src != rendezvousutil.SourceSession {
		t.Fatalf("source %q, want session", src)
	}
	if agentOnly, _ := key(withAgent("A")); both != agentOnly {
		t.Fatal("x-session-id keyed the conversation over agent-identity")
	}
	if headerOnly, _ := key(withSession("D")); both == headerOnly {
		t.Fatal("agent-identity did not key the conversation")
	}
	if _, src := key(withHeaders("x-claude-code-session-id", "A")); src == rendezvousutil.SourceSession {
		t.Fatal("a raw agent header keyed the conversation without the agent-identity plugin")
	}
}

// TestSpillStaysWhileUsed: each turn refreshes the moved entry, so a
// conversation used more often than stickTtlSeconds stays on its spill
// cluster well past that TTL, also after home has room again.
func TestSpillStaysWhileUsed(t *testing.T) {
	guard := newGuard()
	ttl := 0.5
	f := newFilter(t, Config{StickTTLSeconds: &ttl}, guard)
	endpoints := []scheduling.Endpoint{shared("a", "1"), shared("b", "1")}
	s := sessionOnA(t, f, endpoints)
	guard.full = map[string]bool{"a": true}
	if got := pick(t, f, withSession(s), endpoints); got != "b" {
		t.Fatalf("spill went to %s", got)
	}
	guard.full = map[string]bool{}
	for deadline := time.Now().Add(4 * time.Duration(ttl*float64(time.Second))); time.Now().Before(deadline); time.Sleep(100 * time.Millisecond) {
		if got := pick(t, f, withSession(s), endpoints); got != "b" {
			t.Fatalf("a used conversation left its spill cluster for %s", got)
		}
	}
}

// TestPreRequestWithoutPick: when no endpoint was picked (a later filter
// emptied the candidates), PreRequest leaves the moved table and the pending
// spill as they are.
func TestPreRequestWithoutPick(t *testing.T) {
	guard := newGuard()
	f := newFilter(t, baseConfig(), guard)
	endpoints := []scheduling.Endpoint{shared("a", "1"), shared("b", "1"), shared("c", "1")}
	s := sessionOnA(t, f, endpoints)
	guard.full = map[string]bool{"a": true, "b": true}
	if got := pick(t, f, withSession(s), endpoints); got != "c" {
		t.Fatalf("spill went to %s", got)
	}
	guard.full = map[string]bool{"a": true, "c": true}
	r := withSession(s)
	r.RequestID = "no-pick"
	if got := names(f.Filter(t.Context(), r, endpoints)); !reflect.DeepEqual(got, []string{"b"}) {
		t.Fatalf("spill candidates %v, want [b]", got)
	}
	for _, result := range []*scheduling.SchedulingResult{nil, {}} {
		if err := f.PreRequest(t.Context(), r, result); err != nil {
			t.Fatal(err)
		}
	}
	if !hasPending(f, "no-pick") || countMoved(f) != 1 {
		t.Fatalf("pending %v, %d moved; want the spill kept", hasPending(f, "no-pick"), countMoved(f))
	}
	guard.full = map[string]bool{}
	if got := pick(t, f, withSession(s), endpoints); got != "c" {
		t.Fatalf("conversation lost its spill cluster: went to %s", got)
	}
}
