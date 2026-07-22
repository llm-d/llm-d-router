package disaggregation

import (
	"context"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// ControllerType is the plugin-framework type identifier for this
// controller. It is required by the plugin.Plugin interface; the controller
// is never registered in the plugin registry.
const ControllerType = "disaggregation"

// Controller owns the per-request work: header stamping on responses, and
// (via the wrappers WireInto constructs) filtering on requests. It holds
// the parsed config and a shared PodCache.
type Controller struct {
	config    Config
	podCache  *PodCache
	typedName fwkplugin.TypedName
}

var (
	_ fwkplugin.Plugin              = (*Controller)(nil)
	_ fwkrc.ResponseHeaderProcessor = (*Controller)(nil)
)

// NewController builds a controller from an already-validated config and an
// already-started PodCache. Prefer Register() from an EPP boot path;
// NewController is the low-level constructor for tests.
func NewController(config Config, podCache *PodCache) *Controller {
	return &Controller{
		config:    config,
		podCache:  podCache,
		typedName: fwkplugin.TypedName{Type: ControllerType, Name: ControllerType},
	}
}

// TypedName implements plugin.Plugin.
func (controller *Controller) TypedName() fwkplugin.TypedName { return controller.typedName }

// Filter is a test-only entry point that applies every configured selector
// in declaration order. Production wiring uses the mode-specific wrappers
// installed by WireInto — never the Controller itself as a Filter.
//
// The returned slice is always a fresh allocation — matches the convention
// followed by every other scheduler filter and lets callers treat their
// input as read-only regardless of what happens downstream.
func (controller *Controller) Filter(ctx context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	return controller.filterSelectors(ctx, request, pods, func(Selector) bool { return true })
}

// filterSelectors is the shared engine used by the strict and prefer
// wrappers. keepSelector decides which configured selectors participate in
// this pass.
func (controller *Controller) filterSelectors(_ context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint, keepSelector func(Selector) bool) []fwksched.Endpoint {
	current := append(make([]fwksched.Endpoint, 0, len(pods)), pods...)
	if request == nil || len(current) == 0 {
		return current
	}
	for _, selector := range controller.config.Selectors {
		if !keepSelector(selector) {
			continue
		}
		requested, present := request.Headers[selector.HeaderName]
		if !present || requested == "" {
			continue
		}
		matched := make([]fwksched.Endpoint, 0, len(current))
		for _, endpoint := range current {
			if endpoint == nil || endpoint.GetMetadata() == nil {
				continue
			}
			if endpoint.GetMetadata().Labels[selector.LabelKey] == requested {
				matched = append(matched, endpoint)
			}
		}
		switch selector.Mode {
		case ModeStrict:
			current = matched
			if len(matched) == 0 {
				recordFilterOutcome(selector.Name, selector.Mode, filterOutcomeNoMatchStrict)
			} else {
				recordFilterOutcome(selector.Name, selector.Mode, filterOutcomeMatched)
			}
		case ModePrefer:
			if len(matched) > 0 {
				current = matched
				recordFilterOutcome(selector.Name, selector.Mode, filterOutcomeMatched)
			} else {
				recordFilterOutcome(selector.Name, selector.Mode, filterOutcomeNoMatchPreferFallback)
			}
		}
		if len(current) == 0 {
			return current
		}
	}
	return current
}

// ResponseHeader stamps each selector's headerName onto the response with
// the serving endpoint's labelKey value. Skips silently on empty labels or
// a nil endpoint/response.
func (controller *Controller) ResponseHeader(_ context.Context, _ *fwksched.InferenceRequest, response *fwkrc.Response, endpoint *fwkdl.EndpointMetadata) {
	if endpoint == nil || response == nil || response.Headers == nil {
		return
	}
	for _, selector := range controller.config.Selectors {
		if value := endpoint.Labels[selector.LabelKey]; value != "" {
			response.Headers[selector.HeaderName] = value
			recordHeaderStamped(selector.Name)
		}
	}
}
