package disaggregation

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"

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
// the parsed config plus a cache-backed reader used by the gating filter
// and the boot-time role coverage check.
type Controller struct {
	config           Config
	reader           client.Reader
	namespace        string
	scope            labels.Selector
	revisionLabelKey string
	roleLabelKey     string
	typedName        fwkplugin.TypedName
}

var (
	_ fwkplugin.Plugin              = (*Controller)(nil)
	_ fwkrc.ResponseHeaderProcessor = (*Controller)(nil)
)

// NewController builds a controller from an already-validated config, a
// cache-backed reader (typically mgr.GetCache()), and the scope the reader
// should be filtered against. Prefer Register() from an EPP boot path;
// NewController is the low-level constructor for tests.
func NewController(config Config, reader client.Reader, namespace string, scope labels.Selector) *Controller {
	revisionLabelKey := ""
	if len(config.Selectors) > 0 {
		revisionLabelKey = config.Selectors[0].LabelKey
	}
	roleLabelKey := ""
	if config.Gating.Active() {
		roleLabelKey = config.Gating.RequireRoles.LabelKey
	}
	return &Controller{
		config:           config,
		reader:           reader,
		namespace:        namespace,
		scope:            scope,
		revisionLabelKey: revisionLabelKey,
		roleLabelKey:     roleLabelKey,
		typedName:        fwkplugin.TypedName{Type: ControllerType, Name: ControllerType},
	}
}

// TypedName implements plugin.Plugin.
func (c *Controller) TypedName() fwkplugin.TypedName { return c.typedName }

// Filter is a test-only entry point that applies every configured selector
// in declaration order. Production wiring uses the mode-specific wrappers
// installed by WireInto — never the Controller itself as a Filter.
//
// The returned slice is always a fresh allocation — matches the convention
// followed by every other scheduler filter and lets callers treat their
// input as read-only regardless of what happens downstream.
func (c *Controller) Filter(ctx context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	return c.filterSelectors(ctx, request, pods, func(Selector) bool { return true })
}

func (c *Controller) filterSelectors(_ context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint, keepSelector func(Selector) bool) []fwksched.Endpoint {
	current := append(make([]fwksched.Endpoint, 0, len(pods)), pods...)
	if request == nil || len(current) == 0 {
		return current
	}
	for _, selector := range c.config.Selectors {
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
func (c *Controller) ResponseHeader(_ context.Context, _ *fwksched.InferenceRequest, response *fwkrc.Response, endpoint *fwkdl.EndpointMetadata) {
	if endpoint == nil || response == nil || response.Headers == nil {
		return
	}
	for _, selector := range c.config.Selectors {
		if value := endpoint.Labels[selector.LabelKey]; value != "" {
			response.Headers[selector.HeaderName] = value
			recordHeaderStamped(selector.Name)
		}
	}
}

// scanCoverage lists Ready pods matching the controller's scope and groups
// them by (revision → role → count). One cache-backed List per call. Used
// by the gating filter (per request, for both coverage AND revision-weight
// computation) and by boot validation.
func (c *Controller) scanCoverage(ctx context.Context) (revisions map[string]struct{}, roleCounts map[string]map[string]int, err error) {
	var podList corev1.PodList
	if err := c.reader.List(ctx, &podList,
		client.InNamespace(c.namespace),
		client.MatchingLabelsSelector{Selector: c.scope},
	); err != nil {
		return nil, nil, fmt.Errorf("list pods in scope: %w", err)
	}
	revisions = make(map[string]struct{})
	roleCounts = make(map[string]map[string]int)
	for i := range podList.Items {
		pod := &podList.Items[i]
		if !isPodReady(pod) {
			continue
		}
		revision := pod.Labels[c.revisionLabelKey]
		if revision == "" {
			continue
		}
		revisions[revision] = struct{}{}
		if c.roleLabelKey == "" {
			continue
		}
		role := pod.Labels[c.roleLabelKey]
		if role == "" {
			// Missing role label — skip so it can't accidentally satisfy
			// a coverage check for that revision.
			continue
		}
		if roleCounts[revision] == nil {
			roleCounts[revision] = make(map[string]int)
		}
		roleCounts[revision][role]++
	}
	return revisions, roleCounts, nil
}

// isPodReady reports whether a pod is Running, not being terminated, and
// has a Ready=True condition. Matches what the endpoint controller uses
// when deciding whether a pod belongs in an EndpointSlice.
func isPodReady(pod *corev1.Pod) bool {
	if pod == nil || pod.Status.Phase != corev1.PodRunning || pod.DeletionTimestamp != nil {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}
