package disaggregation

import (
	"github.com/llm-d/llm-d-router/pkg/epp/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

// WireInto attaches a Controller to a SchedulerConfig and a request-control
// Config. The scheduler filter chain, in execution order per request:
//
//	gating → strict → operator filters → prefer
//
// The revision axis is decided at the head of the chain — either by the
// gating filter (stochastic, cache-driven weighted pick, fires when no
// client header is present) or by the strict filter (deterministic, from
// the request header, fires when the header is set). Only one of the two
// makes the revision decision on any given request; the other detects the
// situation and fast-paths.
//
// Everything downstream of the head therefore sees a candidate pool
// already narrowed to a single revision. Operator-declared filters run in
// the middle. The prefer filter is appended at the tail so that its
// "keep-if-empty" fallback captures the fully-narrowed pool — an operator
// filter that trims after prefer would violate prefer's fallback set.
//
// The response-header processor is attached to the request-control
// pipeline. No scorer is registered: pick behaviour is left to the
// operator's YAML picker choice.
func WireInto(schedulerConfig *scheduling.SchedulerConfig, requestControlConfig *requestcontrol.Config, controller *Controller) error {
	if controller == nil {
		return nil
	}

	strictFilter := newModeSelectorsFilter(controller, ModeStrict, strictFilterType)
	preferFilter := newModeSelectorsFilter(controller, ModePrefer, preferFilterType)

	hasStrict := controller.config.HasSelectorsInMode(ModeStrict)
	hasPrefer := controller.config.HasSelectorsInMode(ModePrefer)
	gatingActive := controller.config.Gating.Active()

	var gating *gatingFilter
	if gatingActive {
		gating = newGatingFilter(controller)
	}

	for _, profile := range schedulerConfig.Profiles() {
		// Prepend order matters: the LAST PrependFilter call ends up at
		// position 0 of the chain. Strict is prepended first so gating
		// ends up in front of it — gating is the first thing to run.
		if hasStrict {
			profile.PrependFilter(strictFilter)
		}
		if gatingActive {
			profile.PrependFilter(gating)
		}
		if hasPrefer {
			profile.AppendFilter(preferFilter)
		}
	}
	requestControlConfig.AddPlugins(controller)
	return nil
}
