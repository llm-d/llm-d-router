# Extending llm-d-inference-scheduler with a custom filter

## Goal

This tutorial outlines the steps needed for creating and hooking a new filter
 for the llm-d-inference-scheduler.
 
The tutorial demonstrates the coding of a new filter, which selects inference
 serving Pods based on their labels. All relevant code is contained in the
 [`bylabel`](https://github.com/llm-d/llm-d-inference-scheduler/tree/main/pkg/epp/framework/plugins/scheduling/filter/bylabel) package
 (registered as the `label-selector-filter` plugin type).

## Introduction to filtering

Plugins are used to modify llm-d-inference-scheduler's default behavior. Filter plugins
 are provided with a list of candidate inference serving Pods and filter out the
 Pods which do not match the filtering criteria. Several filtering plugins can
 run in succession to produce the final candidate list which is then evaluated,
 through the process of _scoring_, to select the most appropriate target Pods.
 While llm-d-inference-scheduler comes with several existing filters and
 more are available in the upstream [Gateway API Inference Extension](https://sigs.k8s.io/gateway-api-inference-extension),
 in some cases it may be desirable to create and deploy custom filtering code to
 match your specific requirements.

Filters implement the `scheduling.Filter` interface and execute early in the scheduling pipeline:

```go
Filter(ctx context.Context, cycleState *scheduling.CycleState, request *scheduling.InferenceRequest, pods []scheduling.Endpoint) []scheduling.Endpoint
```

Key upstream types used in the signature:
- `scheduling.InferenceRequest` — parsed request with model, body, headers, and objectives
- `scheduling.Endpoint` — candidate endpoint interface with metadata and metrics

The `Filter` function accepts the request and a slice of candidate endpoints. Each endpoint exposes relevant inference attributes, such a model server metrics, which can be used to make scheduling decisions. The function returns a (possibly smaller) slice of endpoints which satisfy the filtering criteria.

## Code walkthrough

The top of the file has the expected Go package and import statements:

```go
package filter

import (
	"errors"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/scheduling/plugins"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/scheduling/types"
)
```

Specifically, we import the Kubernetes `meta/v1` and `labels` packages to allow
 defining and using `label.Selector` objects, and the Gateway API Infernce
 Extension's `plugin` (defininig the plugin interfaces) and `types` (defining
 scheduling related objects) packages.

Next we define the `ByLabels` struct type, along with the relevant fields,
 and a constructor function.

```go
// ByLabels filters out pods that do not match its label selector criteria
type ByLabels struct {
	name     string
	selector labels.Selector
}

var _ plugins.Filter = &ByLabels{} // validate interface conformance

// NewByLabel returns a new filter instance, configured with the provided
// name and label selector.
func NewByLabel(name string, selector *metav1.LabelSelector) (plugins.Filter, error) {
	if name == "" {
		return nil, errors.New("ByLabels: missing filter name")
	}
	labelSelector, err := metav1.LabelSelectorAsSelector(selector)
	if err != nil {
		return nil, err
	}

	return &ByLabels{
		name:     name,
		selector: labelSelector,
	}, nil
}
```

> Note that, since Go supports "duck typing", the`plugin` package is
 not strictly required. We use it to validate `ByLabels` interface conformance
 (a pattern known as "interface implementation assertion" or "compile-time
 interface" check). The statement asserts at compile time that `ByLabels`
 implements the `plugins.Filter` interface and is useful for catching errors
 early, especially when refactoring (e.g. interface methods or signatures change).

Next, we define the required `plugins.Filter` interface methods:

```go
// Name returns the name of the filter
func (blf *ByLabels) Name() string {
	return blf.name
}

// Filter filters out all pods that do not satisfy the label selector
func (blf *ByLabels) Filter(_ *types.SchedulingContext, pods []types.Pod) []types.Pod {
	filtered := []types.Pod{}

	for _, pod := range pods {
		labels := labels.Set(pod.GetPod().Labels)
		if blf.selector.Matches(labels) {
			filtered = append(filtered, pod)
		}
	}
	return filtered
}
```

Since the filter is only matching on candidate `types.Pod` labels,
 we leave the `types.SchedulingContext` parameter unnamed. Filters
 that need access to LLM request information (e.g., filtering based
 on prompt length) may use it.

## Hooking the filter into the scheduling flow

Once a filter is defined, it can be used to modify llm-d-inference-scheduler
 configuration. This would typically be done by modifying the
`pkg/config/config.go` file to
 
- Add the relevant import path (if defined outside this repository);
- Add any desired configuration knobs (e.g., environment variables); and
- Listing the new filter in the `LoadConfigPhaseTwo()` function's `cfg.loadPluginInfo`
 list of available plugins.

In the case of the llm-d-inference-scheduler, filters can be hooked into the
 `Prefill` and/or `Decode` scheduling cycles. For example, the following snippet
 adds the `ByLabels` filter to the list of plugins available to the `Decode`
 scheduler (assuming a `ByLabelFilterName` constant is defined along with other
 environment variables):

```go 
func (c *Config) LoadConfigPhaseTwo() {
	c.loadPluginInfo(c.DecodeSchedulerPlugins, false,
		KVCacheScorerName, ..., ByLabelFilterName, ... )
	c.loadPluginInfo(c.PrefillSchedulerPlugins, true, ... )
	// ...
}
```

> Note: a real filter would require unit tests, etc. These are left out to
 keep the tutorial short and focused.

## Next steps

If you have an idea for a new `Filter` (or other) plugin - we'd love to hear
 from you! Please open an [issue](https://github.com/llm-d/llm-d-inference-scheduler/issues/new/choose),
 describing your use case and requirements, and we'll reach out to refine
 and collaborate.
