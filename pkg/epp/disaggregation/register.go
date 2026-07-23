package disaggregation

import (
	"context"
	"errors"
	"fmt"

	"k8s.io/client-go/kubernetes"
	ctrllog "sigs.k8s.io/controller-runtime/pkg/log"
)

// ErrDisabled is returned by Register when called with a disabled config.
// Callers should gate on config.Enabled before calling and treat this error
// as a no-op signal rather than a failure.
var ErrDisabled = errors.New("disaggregation is disabled")

// Register validates config, starts the informer-backed pod cache, waits for
// cache sync, and returns a ready controller. Boot fails when:
//
//   - config is disabled (returns ErrDisabled — caller-side gate)
//   - config is invalid (see Config.Validate)
//   - the pod cache cannot be built or does not sync within ctx
//   - Gating is set and any listed role has zero observed Ready pods,
//     or no observed revision has Ready pods for every listed role
//     simultaneously
//
// Fail-fast keeps a wrongly-labelled deployment from starting the EPP with
// silent misdirected routing.
func Register(ctx context.Context, client kubernetes.Interface, namespace string, config Config) (*Controller, error) {
	if !config.Enabled {
		return nil, ErrDisabled
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}

	revisionLabelKey := config.Selectors[0].LabelKey
	roleLabelKey := ""
	if config.Gating.Active() {
		roleLabelKey = config.Gating.RequireRoles.LabelKey
	}
	// roleLabelKey may be empty — PodCache drops the role dimension in that
	// case and stores per-revision counts only. Empty is a legitimate config
	// (gating absent → no cross-role check needed).

	// scope.namespace overrides the caller-provided namespace when set. The
	// runner passes the InferencePool's namespace by default; explicit
	// config wins when the operator wants to target a different namespace
	// (e.g. when the EPP itself lives in an ops namespace but observes
	// workloads elsewhere).
	watchNamespace := config.Scope.Namespace
	if watchNamespace == "" {
		watchNamespace = namespace
	}

	registerMetrics()
	podCache, err := NewPodCache(client, watchNamespace, config.Scope.LabelSelector, revisionLabelKey, roleLabelKey)
	if err != nil {
		return nil, fmt.Errorf("build pod cache: %w", err)
	}
	podCache.Start(ctx)
	if !podCache.WaitForCacheSync(ctx) {
		return nil, errors.New("pod cache did not sync before context expired")
	}

	if config.Gating.Active() {
		if err := validateRolesObserved(podCache, config.Gating.RequireRoles.Values); err != nil {
			return nil, err
		}
	}

	controller := NewController(config, podCache)
	ctrllog.FromContext(ctx).Info("disaggregation controller registered",
		"namespace", watchNamespace,
		"scope", config.Scope.LabelSelector,
		"selectors", len(config.Selectors),
		"gating", gatingForLog(config.Gating),
	)
	return controller, nil
}

func gatingForLog(g *Gating) string {
	if g == nil {
		return "off"
	}
	return string(g.Mode)
}

// validateRolesObserved fails at boot when the required-roles gate would
// drop every observed revision. Two checks:
//
//   - Per-role liveness: each listed role must have ≥1 Ready pod for at
//     least one revision. Catches typos in role names.
//   - Compound liveness: at least one revision must have ≥1 Ready pod for
//     EVERY listed role simultaneously. Catches the subtler misconfig
//     where each role has pods somewhere but they never overlap on a
//     single revision — the filter would otherwise silently drop every
//     endpoint at request time.
func validateRolesObserved(podCache *PodCache, roles []string) error {
	revisions := podCache.Revisions()
	if len(revisions) == 0 {
		return errors.New("no revisions observed in scope after cache sync")
	}
	for _, role := range roles {
		observed := false
		for _, revision := range revisions {
			if podCache.HasRoleForRevision(revision, role) {
				observed = true
				break
			}
		}
		if !observed {
			return fmt.Errorf("required role %q has zero Ready pods across all observed revisions", role)
		}
	}
	for _, revision := range revisions {
		allRolesLive := true
		for _, role := range roles {
			if !podCache.HasRoleForRevision(revision, role) {
				allRolesLive = false
				break
			}
		}
		if allRolesLive {
			return nil
		}
	}
	return fmt.Errorf("no observed revision has Ready pods for every role in gating.requireRoles.values=%v; every endpoint would be gated out", roles)
}
