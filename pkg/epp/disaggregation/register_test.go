package disaggregation

import (
	"context"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func newSeededClient(t *testing.T, pods ...*corev1.Pod) *fake.Clientset {
	t.Helper()
	client := fake.NewSimpleClientset()
	for _, p := range pods {
		if _, err := client.CoreV1().Pods(testNS).Create(context.Background(), p, metav1.CreateOptions{}); err != nil {
			t.Fatalf("seed pod: %v", err)
		}
	}
	return client
}

func registerCtx(t *testing.T) (context.Context, context.CancelFunc) {
	t.Helper()
	return context.WithTimeout(context.Background(), 5*time.Second)
}

func TestRegister_InvalidConfigReturnsError(t *testing.T) {
	client := fake.NewSimpleClientset()
	ctx, cancel := registerCtx(t)
	defer cancel()
	cfg := validConfig()
	cfg.Scope.LabelSelector = ""
	if _, err := Register(ctx, client, testNS, cfg); err == nil {
		t.Fatalf("expected validation error")
	}
}

func TestRegister_HappyPath(t *testing.T) {
	client := newSeededClient(t,
		readyPod("p1", "v1", "prefill"),
		readyPod("d1", "v1", "decode"),
	)
	ctx, cancel := registerCtx(t)
	defer cancel()
	c, err := Register(ctx, client, testNS, validConfig())
	if err != nil || c == nil {
		t.Fatalf("register happy path: got (%v, %v)", c, err)
	}
}

func TestRegister_FailsIfRoleNeverObserved(t *testing.T) {
	// prefill pods only — decode role listed in requireRoles but zero pods.
	client := newSeededClient(t,
		readyPod("p1", "v1", "prefill"),
	)
	ctx, cancel := registerCtx(t)
	defer cancel()
	_, err := Register(ctx, client, testNS, validConfig())
	if err == nil || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("expected error naming decode role, got %v", err)
	}
}

func TestRegister_FailsIfNoRevisionsObserved(t *testing.T) {
	client := fake.NewSimpleClientset()
	ctx, cancel := registerCtx(t)
	defer cancel()
	_, err := Register(ctx, client, testNS, validConfig())
	if err == nil || !strings.Contains(err.Error(), "no revisions observed") {
		t.Fatalf("expected no-revisions error, got %v", err)
	}
}

func TestRegister_NoGatingSkipsGateCheck(t *testing.T) {
	// gating absent → gate is not wired → role-observation check should not
	// fire even when the DisaggregatedSet is missing pods.
	cfg := validConfig()
	cfg.Gating = nil
	client := newSeededClient(t, readyPod("p1", "v1", "prefill"))
	ctx, cancel := registerCtx(t)
	defer cancel()
	c, err := Register(ctx, client, testNS, cfg)
	if err != nil || c == nil {
		t.Fatalf("no-gating config should register: got (%v, %v)", c, err)
	}
}

func TestRegister_FailsIfNoRevisionHasAllRequiredRoles(t *testing.T) {
	// Each role has pods somewhere but never both on the same revision.
	// validateRolesObserved's per-role check would say "OK" for both;
	// the compound check must catch it.
	client := newSeededClient(t,
		readyPod("p1", "v1", "prefill"),
		readyPod("d1", "v2", "decode"),
	)
	ctx, cancel := registerCtx(t)
	defer cancel()
	_, err := Register(ctx, client, testNS, validConfig())
	if err == nil || !strings.Contains(err.Error(), "no observed revision") {
		t.Fatalf("expected compound-gate failure, got %v", err)
	}
}

func TestRegister_ContextCancelledDuringSync(t *testing.T) {
	client := newSeededClient(t, readyPod("p1", "v1", "prefill"))
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately
	_, err := Register(ctx, client, testNS, validConfig())
	if err == nil {
		t.Fatalf("expected error on cancelled context")
	}
}
