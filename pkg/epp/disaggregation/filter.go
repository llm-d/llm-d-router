package disaggregation

import (
	"context"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// modeSelectorsFilter is a thin Filter wrapper that visits only the
// controller's selectors matching one mode. Registered at head for strict,
// tail for prefer, so downstream filters never see wrong-revision
// candidates and prefer's fallback captures the fully-narrowed pool.
type modeSelectorsFilter struct {
	controller *Controller
	mode       SelectorMode
	typedName  fwkplugin.TypedName
	keepFn     func(Selector) bool
}

var (
	_ fwkplugin.Plugin = (*modeSelectorsFilter)(nil)
	_ fwksched.Filter  = (*modeSelectorsFilter)(nil)
)

const (
	strictFilterType = "disaggregation-strict-filter"
	preferFilterType = "disaggregation-prefer-filter"
	gatingFilterType = "disaggregation-gating-filter"
)

func newModeSelectorsFilter(controller *Controller, mode SelectorMode, typeName string) *modeSelectorsFilter {
	// Cache the closure at construction time so per-request Filter calls
	// don't allocate a new one.
	return &modeSelectorsFilter{
		controller: controller,
		mode:       mode,
		typedName:  fwkplugin.TypedName{Type: typeName, Name: typeName},
		keepFn: func(selector Selector) bool {
			return selector.Mode == mode
		},
	}
}

func (f *modeSelectorsFilter) TypedName() fwkplugin.TypedName { return f.typedName }

func (f *modeSelectorsFilter) Filter(ctx context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	return f.controller.filterSelectors(ctx, request, pods, f.keepFn)
}

// gatingFilter drops candidates whose revision fails the gating check.
// Appended after the strict and prefer selector wrappers so revision selection
// has already narrowed the pool. Runs even when no selector header is set:
// this filter is the safety gate against rollout drift, independent of client
// behaviour.
type gatingFilter struct {
	controller       *Controller
	revisionLabelKey string
	typedName        fwkplugin.TypedName
}

var (
	_ fwkplugin.Plugin = (*gatingFilter)(nil)
	_ fwksched.Filter  = (*gatingFilter)(nil)
)

func newGatingFilter(controller *Controller) *gatingFilter {
	// Group by the first selector's label key (typically "revision"), same
	// as elsewhere in the controller.
	revisionLabelKey := ""
	if len(controller.config.Selectors) > 0 {
		revisionLabelKey = controller.config.Selectors[0].LabelKey
	}
	return &gatingFilter{
		controller:       controller,
		revisionLabelKey: revisionLabelKey,
		typedName:        fwkplugin.TypedName{Type: gatingFilterType, Name: gatingFilterType},
	}
}

func (f *gatingFilter) TypedName() fwkplugin.TypedName { return f.typedName }

func (f *gatingFilter) Filter(_ context.Context, _ *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	gating := f.controller.config.Gating
	if !gating.Active() || f.revisionLabelKey == "" {
		return append(make([]fwksched.Endpoint, 0, len(pods)), pods...)
	}
	requiredRoles := gating.RequireRoles.Values

	survivors := make([]fwksched.Endpoint, 0, len(pods))
	droppedRevisions := make(map[string]struct{})
	for _, endpoint := range pods {
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		revision := endpoint.GetMetadata().Labels[f.revisionLabelKey]
		if revision == "" {
			continue
		}
		gated := false
		for _, requiredRole := range requiredRoles {
			if !f.controller.podCache.HasRoleForRevision(revision, requiredRole) {
				gated = true
				break
			}
		}
		if gated {
			if _, reported := droppedRevisions[revision]; !reported {
				recordGatingDropped(revision)
				droppedRevisions[revision] = struct{}{}
			}
			continue
		}
		survivors = append(survivors, endpoint)
	}
	return survivors
}
