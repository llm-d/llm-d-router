# Extending llm-d-router with a custom filter

## Goal

This tutorial outlines the steps needed for creating and hooking a new filter  
 for the llm-d-router.

The tutorial demonstrates the coding of a new filter, which selects inference
 serving endpoints based on their labels. All relevant code is contained in the
 [`bylabel`](https://github.com/llm-d/llm-d-router/tree/main/pkg/epp/framework/plugins/scheduling/filter/bylabel) package
 (registered as the `label-selector-filter` plugin type).

## Introduction to filtering

Plugins are used to modify llm-d-router's default behavior. Filter plugins
 are provided with a list of candidate inference serving endpoints and filter out the
 endpoints which do not match the filtering criteria. Several filtering plugins can
 run in succession to produce the final candidate list which is then evaluated,
 through the process of _scoring_, to select the most appropriate target endpoints.

The base [`plugin.Plugin`](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/interface/plugin/plugins.go) interface requires a single method:

```go
type Plugin interface {
    TypedName() TypedName
}
```

Filters implement the [`scheduling.Filter`](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/interface/scheduling/plugins.go) interface:

```go
type Filter interface {
    plugin.Plugin
    Filter(ctx context.Context, cycleState *CycleState, request *InferenceRequest, pods []Endpoint) []Endpoint
}
```

Key types used in the filter signature:
- `scheduling.CycleState` — per-scheduling-cycle state for plugin-to-plugin data sharing
- `scheduling.InferenceRequest` — parsed request with model, body, headers, and objectives
- `scheduling.Endpoint` — candidate endpoint interface exposing metadata (including labels) and metrics

The `Filter` function accepts the request and a slice of candidate endpoints. Each endpoint exposes relevant inference attributes, such as model server metrics, which can be used to make scheduling decisions. The function returns a (possibly smaller) slice of endpoints which satisfy the filtering criteria.

## Code walkthrough

The following walkthrough references [`selector.go`](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/scheduling/filter/bylabel/selector.go).

The top of the file has the expected Go package and import statements:

```go
package bylabel

import (
    "context"
    "encoding/json"
    "errors"
    "fmt"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/labels"

    "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
    "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)
```

Specifically, we import:
- Kubernetes `meta/v1` and `labels` — for label selector types
- framework's `plugin` — base plugin interfaces
- framework's `scheduling` — filter interface and scheduling-related types

Next we define the `Selector` struct type, a plugin type constant, and a compile-time interface check:

```go
const (
    LabelSelectorFilterType = "label-selector-filter"
)

var _ scheduling.Filter = &Selector{}

// Selector filters out endpoints that do not match its label selector criteria.
type Selector struct {
    typedName plugin.TypedName
    selector  labels.Selector
}
```

> Note the compile-time interface check `var _ scheduling.Filter = &Selector{}`.
 This asserts at compile time that `Selector` implements the `scheduling.Filter`
 interface and is useful for catching errors early, especially when refactoring
 (e.g., interface methods or signatures change).

### Factory function

Plugins are instantiated via factory functions. The factory receives the instance name, raw JSON parameters from the configuration, and a `plugin.Handle`:

```go
func SelectorFactory(name string, rawParameters json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
    parameters := metav1.LabelSelector{}
    if rawParameters != nil {
        if err := json.Unmarshal(rawParameters, &parameters); err != nil {
            return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", LabelSelectorFilterType, err)
        }
    }
    return NewSelector(name, &parameters)
}

func NewSelector(name string, selector *metav1.LabelSelector) (*Selector, error) {
    if name == "" {
        return nil, errors.New("Selector: missing filter name")
    }
    labelSelector, err := metav1.LabelSelectorAsSelector(selector)
    if err != nil {
        return nil, err
    }

    return &Selector{
        typedName: plugin.TypedName{Type: LabelSelectorFilterType, Name: name},
        selector:  labelSelector,
    }, nil
}
```

### Interface methods

Next, we define the required interface methods:
- `TypedName()` from `plugin.Plugin`
- `Filter()` from `scheduling.Filter`

```go
func (blf *Selector) TypedName() plugin.TypedName {
    return blf.typedName
}

func (blf *Selector) Filter(_ context.Context, _ *scheduling.CycleState, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
    filtered := []scheduling.Endpoint{}

    for _, endpoint := range endpoints {
        labels := labels.Set(endpoint.GetMetadata().Labels)
        if blf.selector.Matches(labels) {
            filtered = append(filtered, endpoint)
        }
    }
    return filtered
}
```

Since the filter is only matching on candidate endpoint labels, we leave the `context.Context`, `CycleState`, and `InferenceRequest` parameters unnamed. Filters that need access to LLM request information (e.g., filtering based on prompt length) may use them.

## Hooking the filter into the scheduling flow

Once a filter is defined, two steps are needed to make it available:

### 1. Register the factory

Add an import and a `plugin.Register` call in [`register.go`](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/register.go):

```go
import (
    // ...existing imports...
    "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
    // ...
)

func RegisterAllPlugins() {
    // ...existing registrations...
    plugin.Register(bylabel.LabelSelectorFilterType, bylabel.SelectorFactory)
}
```

### 2. Reference the plugin in the EndpointPickerConfig

The EPP is configured via an `EndpointPickerConfig`. First declare the plugin instance in the `plugins` section (with optional `parameters`), then reference it by name in a `schedulingProfiles` entry:

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: label-selector-filter
  name: my-label-filter
  parameters:
    matchLabels:
      role: decode
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: my-label-filter
```

> Note: a real filter would require unit tests, etc. These are left out to keep the tutorial short and focused.

## Next steps

If you have an idea for a new `Filter` (or other) plugin - we'd love to hear from you!
Please open an [issue](https://github.com/llm-d/llm-d-router/issues/new/choose), describing your use case and requirements, and we'll reach out to refine and collaborate.
