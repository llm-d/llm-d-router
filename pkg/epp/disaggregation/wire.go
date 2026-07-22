package disaggregation

import (
	"github.com/llm-d/llm-d-router/pkg/epp/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

// WireInto attaches a Controller to a SchedulerConfig and a request-control
// Config in the correct positions:
//
//   - strict-selector filter prepended to every scheduler profile's filter
//     chain, so no downstream filter ever sees a wrong-revision candidate.
//   - prefer-selector filter appended to every profile's filter chain, so
//     its fallback captures the fully-narrowed pool.
//   - gating filter appended after the two selector wrappers (only when the
//     config declares an active Gating block), so revisions that fail the
//     gating check are dropped before the picker sees them.
//   - response-header processor attached to the request-control pipeline.
//
// No scorer is registered: pick behaviour is left to the operator's YAML
// picker choice. The gating filter handles cross-role liveness; any
// traffic-shaping across surviving revisions is a follow-on concern.
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
		if hasStrict {
			profile.PrependFilter(strictFilter)
		}
		if hasPrefer {
			profile.AppendFilter(preferFilter)
		}
		if gatingActive {
			profile.AppendFilter(gating)
		}
	}
	requestControlConfig.AddPlugins(controller)
	return nil
}
