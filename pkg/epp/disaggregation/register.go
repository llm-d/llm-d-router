package disaggregation

import (
	"context"
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/labels"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrllog "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

// ErrDisabled is returned by Register when called with a disabled config.
// Callers should gate on config.Enabled before calling and treat this error
// as a no-op signal rather than a failure.
var ErrDisabled = errors.New("disaggregation is disabled")

// Register validates config and returns a Controller backed by the Manager's
// shared cache. When Gating is active, boot-time role coverage validation
// runs as a Manager Runnable — it fails mgr.Start with a clear error when
// the observed cluster state would gate out every endpoint. Boot fails when:
//
//   - config is disabled (returns ErrDisabled — caller-side gate)
//   - config is invalid (see Config.Validate)
//   - Gating is set and, after cache sync, any listed role has zero observed
//     Ready pods, or no observed revision has Ready pods for every listed
//     role simultaneously (surfaced as an mgr.Start error)
//
// Fail-fast keeps a wrongly-labelled deployment from starting the EPP with
// silent misdirected routing.
func Register(ctx context.Context, mgr ctrl.Manager, namespace string, config Config) (*Controller, error) {
	if !config.Enabled {
		return nil, ErrDisabled
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}

	scope, err := labels.Parse(config.Scope.LabelSelector)
	if err != nil {
		return nil, fmt.Errorf("parse scope selector %q: %w", config.Scope.LabelSelector, err)
	}

	// scope.namespace overrides the caller-provided namespace when set. The
	// runner passes the InferencePool's namespace by default; explicit
	// config wins when the operator wants to target a different namespace.
	watchNamespace := config.Scope.Namespace
	if watchNamespace == "" {
		watchNamespace = namespace
	}

	registerMetrics()
	controller := newController(config, mgr.GetCache(), watchNamespace, scope)

	if config.Gating.Active() {
		roles := config.Gating.RequireRoles.Values
		if err := mgr.Add(manager.RunnableFunc(func(ctx context.Context) error {
			if !mgr.GetCache().WaitForCacheSync(ctx) {
				return errors.New("pod cache did not sync before context expired")
			}
			return validateRolesObserved(ctx, controller, roles)
		})); err != nil {
			return nil, fmt.Errorf("register boot-validation runnable: %w", err)
		}
	}

	gatingLog := "off"
	if config.Gating != nil {
		gatingLog = string(config.Gating.Mode)
	}
	ctrllog.FromContext(ctx).Info("disaggregation controller registered",
		"namespace", watchNamespace,
		"scope", config.Scope.LabelSelector,
		"selectors", len(config.Selectors),
		"gating", gatingLog,
	)
	return controller, nil
}

// validateRolesObserved fails when the required-roles gate would drop every
// observed revision. Two checks:
//
//   - Per-role liveness: each listed role must have ≥1 Ready pod for at
//     least one revision. Catches typos in role names.
//   - Compound liveness: at least one revision must have ≥1 Ready pod for
//     EVERY listed role simultaneously. Catches the subtler misconfig
//     where each role has pods somewhere but they never overlap on a
//     single revision — the filter would otherwise silently drop every
//     endpoint at request time.
func validateRolesObserved(ctx context.Context, c *Controller, roles []string) error {
	revisions, roleCounts, err := c.scanCoverage(ctx)
	if err != nil {
		return err
	}
	if len(revisions) == 0 {
		return errors.New("no revisions observed in scope after cache sync")
	}
	for _, role := range roles {
		observed := false
		for revision := range revisions {
			if roleCounts[revision][role] > 0 {
				observed = true
				break
			}
		}
		if !observed {
			return fmt.Errorf("required role %q has zero Ready pods across all observed revisions", role)
		}
	}
	for revision := range revisions {
		if crossRoleWeight(roleCounts[revision], roles) > 0 {
			return nil
		}
	}
	return fmt.Errorf("no observed revision has Ready pods for every role in gating.requireRoles.values=%v; every endpoint would be gated out", roles)
}
