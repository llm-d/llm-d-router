package disaggregation

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

func pendingPod(name, revision, role string) *corev1.Pod {
	p := readyPod(name, revision, role)
	p.Status.Phase = corev1.PodPending
	p.Status.Conditions[0].Status = corev1.ConditionFalse
	return p
}

// seedCache builds a PodCache without an informer and feeds it a fixed set of
// pods via the same upsert path an informer would use. Any test that needs a
// populated PodCache uses this — no fake client, no informer sync wait.
func seedCache(t *testing.T, pods ...*corev1.Pod) *PodCache {
	t.Helper()
	pc, err := newPodCache(testNS, testSelector, testRevLabel, testRoleLabel)
	if err != nil {
		t.Fatalf("newPodCache: %v", err)
	}
	for _, pod := range pods {
		pc.upsert(pod)
	}
	return pc
}

func TestPodCache_CountsReadyPods(t *testing.T) {
	pc := seedCache(t,
		readyPod("p1", "v1", "prefill"),
		readyPod("p2", "v1", "prefill"),
		readyPod("p3", "v1", "prefill"),
		readyPod("d1", "v1", "decode"),
		readyPod("d2", "v1", "decode"),
		readyPod("d3", "v1", "decode"),
		readyPod("p4", "v2", "prefill"),
		readyPod("d4", "v2", "decode"),
	)

	if got := pc.Count("v1", "prefill"); got != 3 {
		t.Errorf("v1 prefill: want 3, got %d", got)
	}
	if got := pc.Count("v1", "decode"); got != 3 {
		t.Errorf("v1 decode: want 3, got %d", got)
	}
	if got := pc.Count("v2", "prefill"); got != 1 {
		t.Errorf("v2 prefill: want 1, got %d", got)
	}
	if got := pc.Count("v2", "decode"); got != 1 {
		t.Errorf("v2 decode: want 1, got %d", got)
	}
	if got := pc.Count("v3", "prefill"); got != 0 {
		t.Errorf("unknown revision: want 0, got %d", got)
	}
}

func TestPodCache_SkipsNotReadyPods(t *testing.T) {
	pc := seedCache(t,
		readyPod("p1", "v1", "prefill"),
		pendingPod("p2", "v1", "prefill"),
	)
	if got := pc.Count("v1", "prefill"); got != 1 {
		t.Errorf("only ready pods count: want 1, got %d", got)
	}
}

func TestPodCache_SkipsPodsOutsideSelector(t *testing.T) {
	// A pod whose labels do not satisfy the scope selector must be filtered
	// out at upsert time — the Manager's shared informer sees every pod in
	// the namespace, so we cannot rely on the informer for filtering.
	stray := readyPod("stray", "v1", "prefill")
	delete(stray.Labels, "disaggregatedset.x-k8s.io/name")
	pc := seedCache(t,
		readyPod("p1", "v1", "prefill"),
		stray,
	)
	if got := pc.Count("v1", "prefill"); got != 1 {
		t.Errorf("out-of-selector pod must be dropped: want 1, got %d", got)
	}
}

func TestPodCache_SkipsUnlabelledPods(t *testing.T) {
	unlabelled := readyPod("stray", "", "")
	unlabelled.Labels = map[string]string{"disaggregatedset.x-k8s.io/name": "my-set"}
	pc := seedCache(t,
		readyPod("p1", "v1", "prefill"),
		unlabelled,
	)
	if got := pc.Count("v1", "prefill"); got != 1 {
		t.Errorf("labelled only: want 1, got %d", got)
	}
	if got := pc.Count("", ""); got != 0 {
		t.Errorf("empty label pod must not accrue count, got %d", got)
	}
}

func TestPodCache_HasRoleForRevision(t *testing.T) {
	pc := seedCache(t, readyPod("p1", "v1", "prefill"))
	if !pc.HasRoleForRevision("v1", "prefill") {
		t.Errorf("v1 prefill should be present")
	}
	if pc.HasRoleForRevision("v1", "decode") {
		t.Errorf("v1 decode should be absent")
	}
	if pc.HasRoleForRevision("v2", "prefill") {
		t.Errorf("v2 should be absent")
	}
}

func TestPodCache_Revisions(t *testing.T) {
	pc := seedCache(t,
		readyPod("p1", "v1", "prefill"),
		readyPod("d1", "v1", "decode"),
		readyPod("p2", "v2", "prefill"),
	)
	revs := pc.Revisions()
	if len(revs) != 2 {
		t.Errorf("want 2 revisions, got %d: %v", len(revs), revs)
	}
	if pc.Count("v1", "prefill") != 1 || pc.Count("v1", "decode") != 1 {
		t.Errorf("v1 role counts unexpected: prefill=%d decode=%d", pc.Count("v1", "prefill"), pc.Count("v1", "decode"))
	}
}

func TestPodCache_UpsertThenRemove(t *testing.T) {
	pc := seedCache(t, readyPod("p1", "v1", "prefill"))
	if got := pc.Count("v1", "prefill"); got != 1 {
		t.Fatalf("setup: want 1, got %d", got)
	}
	pc.remove(readyPod("p1", "v1", "prefill"))
	if got := pc.Count("v1", "prefill"); got != 0 {
		t.Fatalf("after remove: want 0, got %d", got)
	}
}

func TestPodCache_UpsertReplacesPreviousPodState(t *testing.T) {
	// A pod flipping from Ready to NotReady (informer UpdateFunc) must
	// drop out of the count immediately.
	pc := seedCache(t, readyPod("p1", "v1", "prefill"))
	if got := pc.Count("v1", "prefill"); got != 1 {
		t.Fatalf("setup: want 1, got %d", got)
	}
	pc.upsert(pendingPod("p1", "v1", "prefill"))
	if got := pc.Count("v1", "prefill"); got != 0 {
		t.Fatalf("after flip to NotReady: want 0, got %d", got)
	}
}
