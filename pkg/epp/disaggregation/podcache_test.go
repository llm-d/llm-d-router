package disaggregation

import (
	"context"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

const (
	testRevLabel  = "disaggregatedset.x-k8s.io/revision"
	testRoleLabel = "disaggregatedset.x-k8s.io/role"
	testNS        = "default"
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

func startCache(t *testing.T, pods ...*corev1.Pod) (*PodCache, context.CancelFunc) {
	t.Helper()
	client := fake.NewSimpleClientset()
	for _, p := range pods {
		if _, err := client.CoreV1().Pods(testNS).Create(context.Background(), p, metav1.CreateOptions{}); err != nil {
			t.Fatalf("seed pod: %v", err)
		}
	}
	pc, err := NewPodCache(client, testNS, "disaggregatedset.x-k8s.io/name=my-set", testRevLabel, testRoleLabel)
	if err != nil {
		t.Fatalf("new cache: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	pc.Start(ctx)
	if !pc.WaitForCacheSync(ctx) {
		cancel()
		t.Fatalf("cache did not sync")
	}
	return pc, cancel
}

func TestPodCache_CountsReadyPods(t *testing.T) {
	pc, cancel := startCache(t,
		readyPod("p1", "v1", "prefill"),
		readyPod("p2", "v1", "prefill"),
		readyPod("p3", "v1", "prefill"),
		readyPod("d1", "v1", "decode"),
		readyPod("d2", "v1", "decode"),
		readyPod("d3", "v1", "decode"),
		readyPod("p4", "v2", "prefill"),
		readyPod("d4", "v2", "decode"),
	)
	defer cancel()

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
	pc, cancel := startCache(t,
		readyPod("p1", "v1", "prefill"),
		pendingPod("p2", "v1", "prefill"),
	)
	defer cancel()
	if got := pc.Count("v1", "prefill"); got != 1 {
		t.Errorf("only ready pods count: want 1, got %d", got)
	}
}

func TestPodCache_SkipsUnlabelledPods(t *testing.T) {
	unlabelled := readyPod("stray", "", "")
	unlabelled.Labels = map[string]string{"disaggregatedset.x-k8s.io/name": "my-set"}
	pc, cancel := startCache(t,
		readyPod("p1", "v1", "prefill"),
		unlabelled,
	)
	defer cancel()
	if got := pc.Count("v1", "prefill"); got != 1 {
		t.Errorf("labelled only: want 1, got %d", got)
	}
	if got := pc.Count("", ""); got != 0 {
		t.Errorf("empty label pod must not accrue count, got %d", got)
	}
}

func TestPodCache_HasRoleForRevision(t *testing.T) {
	pc, cancel := startCache(t,
		readyPod("p1", "v1", "prefill"),
	)
	defer cancel()
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

func TestPodCache_RevisionsAndRoles(t *testing.T) {
	pc, cancel := startCache(t,
		readyPod("p1", "v1", "prefill"),
		readyPod("d1", "v1", "decode"),
		readyPod("p2", "v2", "prefill"),
	)
	defer cancel()

	revs := pc.Revisions()
	if len(revs) != 2 {
		t.Errorf("want 2 revisions, got %d: %v", len(revs), revs)
	}

	roles := pc.rolesForRevision("v1")
	if roles["prefill"] != 1 || roles["decode"] != 1 {
		t.Errorf("v1 roles: %v", roles)
	}
	if pc.rolesForRevision("nonexistent") != nil {
		t.Errorf("missing revision should return nil")
	}
}

func TestPodCache_UpdatesOnPodCreate(t *testing.T) {
	client := fake.NewSimpleClientset()
	pc, err := NewPodCache(client, testNS, "disaggregatedset.x-k8s.io/name=my-set", testRevLabel, testRoleLabel)
	if err != nil {
		t.Fatalf("new cache: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	pc.Start(ctx)
	if !pc.WaitForCacheSync(ctx) {
		t.Fatalf("cache did not sync")
	}
	if _, err := client.CoreV1().Pods(testNS).Create(ctx, readyPod("p2", "v1", "prefill"), metav1.CreateOptions{}); err != nil {
		t.Fatalf("create pod: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if pc.Count("v1", "prefill") == 1 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("cache did not observe pod add within timeout: got %d", pc.Count("v1", "prefill"))
}
