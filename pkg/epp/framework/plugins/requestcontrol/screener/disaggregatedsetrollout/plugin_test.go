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

package disaggregatedsetrollout

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	testRevLabel  = "disaggregatedset.x-k8s.io/revision"
	testRoleLabel = "disaggregatedset.x-k8s.io/role"
	testNS        = "default"
	testSelector  = "disaggregatedset.x-k8s.io/name=my-set"
)

func readyPod(name, revision, role string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: testNS,
			Labels: map[string]string{
				testRevLabel:                     revision,
				testRoleLabel:                    role,
				"disaggregatedset.x-k8s.io/name": "my-set",
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionTrue},
			},
		},
	}
}

func newTestScreener(config Config) *Screener {
	if err := config.Validate(); err != nil {
		panic(err)
	}
	scope, err := labels.Parse(config.Scope.LabelSelector)
	if err != nil {
		panic(err)
	}
	return newScreener("test-screener", config, scope)
}

func endpoint(name string, endpointLabels map[string]string) fwksched.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		ID:     types.NamespacedName{Namespace: testNS, Name: name},
		Name:   name,
		Labels: endpointLabels,
	}
	return fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, nil)
}

func revLabels(revision string) map[string]string {
	return map[string]string{testRevLabel: revision, testRoleLabel: "prefill"}
}

func podEvent(t *testing.T, pod *corev1.Pod, eventType fwkdl.EventType) fwkdl.NotificationEvent {
	t.Helper()
	object, err := runtime.DefaultUnstructuredConverter.ToUnstructured(pod)
	if err != nil {
		t.Fatalf("convert Pod: %v", err)
	}
	result := &fwkdl.NotificationEvent{Type: eventType}
	result.Object = &unstructured.Unstructured{Object: object}
	result.Object.SetGroupVersionKind(podGVK)
	return *result
}

func seedPods(t *testing.T, screener *Screener, pods ...*corev1.Pod) {
	t.Helper()
	handler := &podNotificationHandler{screener: screener}
	for _, pod := range pods {
		if err := handler.Extract(context.Background(), podEvent(t, pod, fwkdl.EventAddOrUpdate)); err != nil {
			t.Fatalf("seed pod %s: %v", pod.Name, err)
		}
	}
}

func seedCounts(t *testing.T, screener *Screener, counts map[string]map[string]int) {
	t.Helper()
	index := 0
	for revision, roles := range counts {
		for role, count := range roles {
			for range count {
				index++
				seedPods(t, screener, readyPod(fmt.Sprintf("%s-%s-%d", revision, role, index), revision, role))
			}
		}
	}
}

func screenCandidates(t *testing.T, screener *Screener, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	t.Helper()
	return screener.Screen(context.Background(), request, endpoints)
}

func TestScreenerFactoryAndPodDependency(t *testing.T) {
	raw, err := json.Marshal(validConfig())
	if err != nil {
		t.Fatal(err)
	}
	plugin, err := Factory("rollout-screener", fwkplugin.StrictDecoder(raw), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	screener := plugin.(*Screener)
	var _ fwkrc.Screener = screener
	if screener.TypedName() != (fwkplugin.TypedName{Type: PluginType, Name: "rollout-screener"}) {
		t.Fatalf("unexpected typed name: %v", screener.TypedName())
	}
	registrar := &captureRegistrar{}
	if err := screener.RegisterDependencies(registrar); err != nil {
		t.Fatalf("RegisterDependencies: %v", err)
	}
	if len(registrar.registrations) != 1 {
		t.Fatalf("want one Pod dependency, got %d", len(registrar.registrations))
	}
	source, ok := registrar.registrations[0].DefaultSource.(fwkdl.NotificationSource)
	if !ok || source.GVK() != podGVK {
		t.Fatalf("dependency is not a Pod notification source: %#v", registrar.registrations[0])
	}
}

func TestScreenerWithDisabledGatingDoesNotRegisterPodDependency(t *testing.T) {
	config := validConfig()
	config.RevisionGating = &RevisionGating{Mode: GatingModeDisabled}
	screener := newTestScreener(config)
	registrar := &captureRegistrar{}
	if err := screener.RegisterDependencies(registrar); err != nil {
		t.Fatalf("RegisterDependencies: %v", err)
	}
	if len(registrar.registrations) != 0 {
		t.Fatalf("want no Pod dependencies, got %d", len(registrar.registrations))
	}
}

type captureRegistrar struct {
	registrations []fwkdl.PendingRegistration
}

func (r *captureRegistrar) Register(registration fwkdl.PendingRegistration) error {
	r.registrations = append(r.registrations, registration)
	return nil
}

func TestPodNotificationsTrackOnlyReadyPodsInScope(t *testing.T) {
	screener := newTestScreener(validConfig())
	handler := &podNotificationHandler{screener: screener}
	inScope := readyPod("p1", "v1", "prefill")
	outOfScope := readyPod("p2", "v2", "decode")
	outOfScope.Labels["disaggregatedset.x-k8s.io/name"] = "other"
	notReady := readyPod("p3", "v1", "decode")
	notReady.Status.Conditions[0].Status = corev1.ConditionFalse
	for _, pod := range []*corev1.Pod{inScope, outOfScope, notReady} {
		if err := handler.Extract(context.Background(), podEvent(t, pod, fwkdl.EventAddOrUpdate)); err != nil {
			t.Fatal(err)
		}
	}
	counts := screener.distributionSnapshot().roleCounts
	if counts["v1"]["prefill"] != 1 || len(counts["v1"]) != 1 || len(counts) != 1 {
		t.Fatalf("unexpected counts: %#v", counts)
	}

	if err := handler.Extract(context.Background(), podEvent(t, inScope, fwkdl.EventDelete)); err != nil {
		t.Fatal(err)
	}
	if got := screener.distributionSnapshot().roleCounts; len(got) != 0 {
		t.Fatalf("delete did not remove Pod: %#v", got)
	}
}

func TestPodNotificationsRefreshCachedRevisionShares(t *testing.T) {
	screener := newTestScreener(validConfig())
	v1Decode := readyPod("v1-decode-1", "v1", "decode")
	seedPods(t, screener,
		readyPod("v1-prefill-1", "v1", "prefill"),
		readyPod("v1-prefill-2", "v1", "prefill"),
		v1Decode,
		readyPod("v1-decode-2", "v1", "decode"),
		readyPod("v2-prefill-1", "v2", "prefill"),
		readyPod("v2-decode-1", "v2", "decode"),
	)

	distribution := screener.distributionSnapshot()
	if math.Abs(distribution.shares["v1"]-2.0/3.0) > 1e-9 || math.Abs(distribution.shares["v2"]-1.0/3.0) > 1e-9 {
		t.Fatalf("unexpected initial shares: %#v", distribution.shares)
	}

	handler := &podNotificationHandler{screener: screener}
	movedToV2 := v1Decode.DeepCopy()
	movedToV2.Labels[testRevLabel] = "v2"
	if err := handler.Extract(context.Background(), podEvent(t, movedToV2, fwkdl.EventAddOrUpdate)); err != nil {
		t.Fatal(err)
	}
	distribution = screener.distributionSnapshot()
	if math.Abs(distribution.shares["v1"]-1.0/2.0) > 1e-9 || math.Abs(distribution.shares["v2"]-1.0/2.0) > 1e-9 {
		t.Fatalf("shares retained stale Pod labels: %#v", distribution.shares)
	}

	notReady := movedToV2.DeepCopy()
	notReady.Status.Conditions[0].Status = corev1.ConditionFalse
	if err := handler.Extract(context.Background(), podEvent(t, notReady, fwkdl.EventAddOrUpdate)); err != nil {
		t.Fatal(err)
	}
	distribution = screener.distributionSnapshot()
	if math.Abs(distribution.shares["v1"]-3.0/5.0) > 1e-9 || math.Abs(distribution.shares["v2"]-2.0/5.0) > 1e-9 {
		t.Fatalf("shares were not refreshed: %#v", distribution.shares)
	}
}

func TestRevisionSumWeightAndCoverage(t *testing.T) {
	tests := []struct {
		name    string
		counts  map[string]int
		want    int
		covered bool
	}{
		{name: "covered", counts: map[string]int{"prefill": 2, "decode": 8}, want: 10, covered: true},
		{name: "missing role", counts: map[string]int{"decode": 8}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, covered := revisionSumWeight(test.counts, []string{"prefill", "decode"})
			if got != test.want || covered != test.covered {
				t.Fatalf("revisionSumWeight() = (%d, %t), want (%d, %t)", got, covered, test.want, test.covered)
			}
		})
	}
}

func TestDominantRoleUsesCountsAcrossCoveredRevisions(t *testing.T) {
	const (
		prefillRole = "prefill"
		decodeRole  = "decode"
	)
	covered := map[string]map[string]int{
		"v1": {prefillRole: 8, decodeRole: 1},
		"v2": {prefillRole: 2, decodeRole: 4},
	}
	if got := dominantRole(covered, []string{prefillRole, decodeRole}); got != prefillRole {
		t.Fatalf("dominantRole() = %q, want prefill", got)
	}
	if got := dominantRole(covered, []string{decodeRole, prefillRole}); got != prefillRole {
		t.Fatalf("dominantRole() = %q, want prefill", got)
	}

	tied := map[string]map[string]int{
		"v1": {prefillRole: 8, decodeRole: 2},
		"v2": {prefillRole: 2, decodeRole: 8},
	}
	if got := dominantRole(tied, []string{prefillRole, decodeRole}); got != prefillRole {
		t.Fatalf("tied dominantRole() = %q, want first required role prefill", got)
	}
}

func TestRevisionModesCacheExpectedShares(t *testing.T) {
	tests := []struct {
		name   string
		mode   GatingMode
		counts map[string]map[string]int
		wantV2 float64
	}{
		{
			name: "2p10d sum",
			mode: GatingModeSum,
			counts: map[string]map[string]int{
				"v1": {"prefill": 2, "decode": 8},
				"v2": {"prefill": 1, "decode": 2},
			},
			wantV2: 3.0 / 13.0,
		},
		{
			name: "2p10d max role",
			mode: GatingModeMaxRole,
			counts: map[string]map[string]int{
				"v1": {"prefill": 2, "decode": 8},
				"v2": {"prefill": 1, "decode": 2},
			},
			wantV2: 2.0 / 10.0,
		},
		{
			name: "10p10d sum",
			mode: GatingModeSum,
			counts: map[string]map[string]int{
				"v1": {"prefill": 8, "decode": 8},
				"v2": {"prefill": 2, "decode": 2},
			},
			wantV2: 2.0 / 10.0,
		},
		{
			name: "10p10d max role",
			mode: GatingModeMaxRole,
			counts: map[string]map[string]int{
				"v1": {"prefill": 8, "decode": 8},
				"v2": {"prefill": 2, "decode": 2},
			},
			wantV2: 2.0 / 10.0,
		},
		{
			name: "10p2d sum",
			mode: GatingModeSum,
			counts: map[string]map[string]int{
				"v1": {"prefill": 8, "decode": 2},
				"v2": {"prefill": 2, "decode": 1},
			},
			wantV2: 3.0 / 13.0,
		},
		{
			name: "10p2d max role",
			mode: GatingModeMaxRole,
			counts: map[string]map[string]int{
				"v1": {"prefill": 8, "decode": 2},
				"v2": {"prefill": 2, "decode": 1},
			},
			wantV2: 2.0 / 10.0,
		},
		{
			name: "max role is shared across revisions",
			mode: GatingModeMaxRole,
			counts: map[string]map[string]int{
				"v1": {"prefill": 8, "decode": 1},
				"v2": {"prefill": 2, "decode": 4},
			},
			wantV2: 2.0 / 10.0,
		},
		{
			name: "max role ignores incomplete revisions",
			mode: GatingModeMaxRole,
			counts: map[string]map[string]int{
				"v1": {"prefill": 2, "decode": 8},
				"v2": {"prefill": 1, "decode": 2},
				"v3": {"prefill": 100},
			},
			wantV2: 2.0 / 10.0,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := validConfig()
			config.RevisionGating.Mode = test.mode
			screener := newTestScreener(config)
			seedCounts(t, screener, test.counts)

			shares := screener.distributionSnapshot().shares
			if math.Abs(shares["v2"]-test.wantV2) > 1e-9 || math.Abs(shares["v1"]-(1-test.wantV2)) > 1e-9 {
				t.Fatalf("unexpected shares: %#v", shares)
			}
		})
	}
}

func TestPickWeightedRevision(t *testing.T) {
	shares := map[string]float64{"v1": 0.75, "v2": 0.25}
	if got := pickWeightedRevision(shares, 0); got != "v1" {
		t.Fatalf("draw 0 selected %q, want v1", got)
	}
	if got := pickWeightedRevision(shares, 0.9); got != "v2" {
		t.Fatalf("draw 0.9 selected %q, want v2", got)
	}
}

func TestScreenerPinnedUncoveredRevisionFailsClosed(t *testing.T) {
	screener := newTestScreener(validConfig())
	seedCounts(t, screener, map[string]map[string]int{
		"v1": {"prefill": 3},
		"v2": {"prefill": 1, "decode": 1},
	})
	candidates := []fwksched.Endpoint{
		endpoint("v1-p", revLabels("v1")),
		endpoint("v2-p", revLabels("v2")),
	}
	request := &fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v1"}}
	if got := screenCandidates(t, screener, request, candidates); len(got) != 0 {
		t.Fatalf("uncovered pinned revision must fail closed, got %v", got)
	}
}

func TestScreenerSingleRevisionStillChecksCrossRoleCoverage(t *testing.T) {
	screener := newTestScreener(validConfig())
	seedCounts(t, screener, map[string]map[string]int{"v1": {"prefill": 2}})
	request := &fwksched.InferenceRequest{Headers: map[string]string{}}
	candidates := []fwksched.Endpoint{endpoint("v1-p", revLabels("v1"))}
	if got := screenCandidates(t, screener, request, candidates); len(got) != 0 {
		t.Fatalf("single uncovered revision must be gated, got %v", got)
	}
}

func TestScreenerFailsClosedUntilPodNotificationsArrive(t *testing.T) {
	screener := newTestScreener(validConfig())
	request := &fwksched.InferenceRequest{Headers: map[string]string{}}
	candidates := []fwksched.Endpoint{endpoint("v1-p", revLabels("v1"))}
	if got := screenCandidates(t, screener, request, candidates); len(got) != 0 {
		t.Fatalf("empty notification state must fail closed, got %v", got)
	}
}

func TestScreenerWeightedDistribution(t *testing.T) {
	tests := []struct {
		name      string
		mode      GatingMode
		counts    map[string]map[string]int
		wantShare float64
	}{
		{"2p10d sum", GatingModeSum, map[string]map[string]int{"v1": {"prefill": 2, "decode": 8}, "v2": {"prefill": 1, "decode": 2}}, 3.0 / 13.0},
		{"2p10d max role", GatingModeMaxRole, map[string]map[string]int{"v1": {"prefill": 2, "decode": 8}, "v2": {"prefill": 1, "decode": 2}}, 2.0 / 10.0},
		{"10p10d sum", GatingModeSum, map[string]map[string]int{"v1": {"prefill": 8, "decode": 8}, "v2": {"prefill": 2, "decode": 2}}, 2.0 / 10.0},
		{"10p10d max role", GatingModeMaxRole, map[string]map[string]int{"v1": {"prefill": 8, "decode": 8}, "v2": {"prefill": 2, "decode": 2}}, 2.0 / 10.0},
		{"10p2d sum", GatingModeSum, map[string]map[string]int{"v1": {"prefill": 8, "decode": 2}, "v2": {"prefill": 2, "decode": 1}}, 3.0 / 13.0},
		{"10p2d max role", GatingModeMaxRole, map[string]map[string]int{"v1": {"prefill": 8, "decode": 2}, "v2": {"prefill": 2, "decode": 1}}, 2.0 / 10.0},
	}
	const iterations = 10000
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := validConfig()
			config.RevisionGating.Mode = test.mode
			screener := newTestScreener(config)
			seedCounts(t, screener, test.counts)
			candidates := candidatePool(test.counts["v1"]["prefill"], test.counts["v2"]["prefill"])
			v2 := 0
			for range iterations {
				request := &fwksched.InferenceRequest{Headers: map[string]string{}}
				got := screenCandidates(t, screener, request, candidates)
				if len(got) == 0 {
					t.Fatal("empty survivor set")
				}
				if got[0].GetMetadata().Labels[testRevLabel] == "v2" {
					v2++
				}
			}
			share := float64(v2) / iterations
			if share < test.wantShare-0.03 || share > test.wantShare+0.03 {
				t.Fatalf("want v2 share %.3f +/- .03, got %.3f", test.wantShare, share)
			}
		})
	}
}

func candidatePool(v1, v2 int) []fwksched.Endpoint {
	result := make([]fwksched.Endpoint, 0, v1+v2)
	for i := range v1 {
		result = append(result, endpoint("v1-"+strconv.Itoa(i), revLabels("v1")))
	}
	for i := range v2 {
		result = append(result, endpoint("v2-"+strconv.Itoa(i), revLabels("v2")))
	}
	return result
}

func TestStrictSelectors(t *testing.T) {
	config := validConfig()
	config.RevisionGating = &RevisionGating{Mode: GatingModeDisabled}
	screener := newTestScreener(config)
	candidates := []fwksched.Endpoint{endpoint("v1", revLabels("v1")), endpoint("v2", revLabels("v2"))}
	request := &fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v2"}}
	got := screenCandidates(t, screener, request, candidates)
	if len(got) != 1 || got[0].GetMetadata().Name != "v2" {
		t.Fatalf("strict screening mismatch: %v", got)
	}
	request.Headers["x-disagg-revision"] = "missing"
	if got := screenCandidates(t, screener, request, candidates); len(got) != 0 {
		t.Fatalf("strict no-match must be empty: %v", got)
	}
}

func TestResponseHeaderStampsSelectors(t *testing.T) {
	screener := newTestScreener(validConfig())
	response := &fwkrc.Response{Headers: map[string]string{}}
	screener.ResponseHeader(context.Background(), nil, response, &fwkdl.EndpointMetadata{Labels: revLabels("v1")})
	if response.Headers["x-disagg-revision"] != "v1" {
		t.Fatalf("revision header not stamped: %#v", response.Headers)
	}
}

func TestResponseHeaderStampsSelectorsWithGatingDisabled(t *testing.T) {
	config := validConfig()
	config.RevisionGating = &RevisionGating{Mode: GatingModeDisabled}
	screener := newTestScreener(config)
	response := &fwkrc.Response{Headers: map[string]string{}}
	screener.ResponseHeader(context.Background(), nil, response, &fwkdl.EndpointMetadata{Labels: revLabels("v1")})
	if response.Headers["x-disagg-revision"] != "v1" {
		t.Fatalf("revision header not stamped with gating disabled: %#v", response.Headers)
	}
}
