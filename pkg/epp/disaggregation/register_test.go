package disaggregation

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// Register's k8s wiring (attach event handler, register a boot-validation
// runnable) needs a real controller-runtime Manager, which is exercised in
// the integration tests via envtest. Here we cover Register's pure branches
// (disabled short-circuit + config validation) and the boot-validator
// (validateRolesObserved) end of the story by feeding a PodCache directly.

func TestRegister_DisabledReturnsErrDisabled(t *testing.T) {
	_, err := Register(context.Background(), nil, testNS, Config{Enabled: false})
	if !errors.Is(err, ErrDisabled) {
		t.Fatalf("want ErrDisabled, got %v", err)
	}
}

func TestRegister_InvalidConfigReturnsError(t *testing.T) {
	cfg := validConfig()
	cfg.Scope.LabelSelector = ""
	_, err := Register(context.Background(), nil, testNS, cfg)
	if err == nil || errors.Is(err, ErrDisabled) {
		t.Fatalf("want validation error, got %v", err)
	}
}

// --- validateRolesObserved (the body of the boot-time Runnable) ---

func TestValidateRolesObserved_HappyPath(t *testing.T) {
	pc := seedCache(t,
		readyPod("p1", "v1", "prefill"),
		readyPod("d1", "v1", "decode"),
	)
	if err := validateRolesObserved(pc, []string{"prefill", "decode"}); err != nil {
		t.Fatalf("valid coverage should pass: %v", err)
	}
}

func TestValidateRolesObserved_FailsIfNoRevisions(t *testing.T) {
	pc := seedCache(t)
	err := validateRolesObserved(pc, []string{"prefill", "decode"})
	if err == nil || !strings.Contains(err.Error(), "no revisions observed") {
		t.Fatalf("want no-revisions error, got %v", err)
	}
}

func TestValidateRolesObserved_FailsIfRoleNeverObserved(t *testing.T) {
	pc := seedCache(t, readyPod("p1", "v1", "prefill"))
	err := validateRolesObserved(pc, []string{"prefill", "decode"})
	if err == nil || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("want error naming decode, got %v", err)
	}
}

func TestValidateRolesObserved_FailsIfNoRevisionHasAllRoles(t *testing.T) {
	// Each role has pods somewhere but never both on the same revision.
	// Per-role check would pass; the compound check must catch it.
	pc := seedCache(t,
		readyPod("p1", "v1", "prefill"),
		readyPod("d1", "v2", "decode"),
	)
	err := validateRolesObserved(pc, []string{"prefill", "decode"})
	if err == nil || !strings.Contains(err.Error(), "no observed revision") {
		t.Fatalf("want compound-gate failure, got %v", err)
	}
}
