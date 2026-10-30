# Framework Developer Guide: Plugins vs Builtins

## Purpose

This document defines the distinction between **plugins** and **builtins** in the
llm-d Router framework. Following these guidelines ensures a
consistent architecture where users have clear control over what they can
configure, and core runtime machinery remains reliable and non-optional.

## Plugins

A **plugin** is a component that:

- **Can be enabled, disabled, or swapped** by a user through the YAML
  configuration (EndpointPickerConfig).
- **Lives under** `pkg/epp/framework/plugins/` (or in an external plugin
  repository; see [Registering out-of-tree plugins](#registering-out-of-tree-plugins)).
- **Implements `plugin.Plugin`** (specifically the `TypedName()` method),
  allowing the framework to identify and instantiate it from config.
- **Is registered** via `plugin.Register(type, stability, factory)` (with stability level `plugin.StabilityAlpha`, `plugin.StabilityBeta`, or `plugin.StabilityStable`) so the config parser can look it up by type name.

### Examples of plugins

- **Scheduling filters**: `label-selector-filter`, `encode-filter`, `prefill-filter`, `decode-filter`, etc.
- **Scheduling scorers**: `prefix`, `sessionaffinity`, `loadaware`, `loraaffinity`, etc.
- **Profile handlers**: `SingleProfileHandler`, `DisaggProfileHandler`, etc.
- **Eviction policies**: `EvictionOrderingPolicy` (e.g. `eviction-priority-then-time-ordering`),
  `EvictionFilterPolicy` (e.g. `eviction-sheddable-filter`).

### Key characteristics

- A user **chooses** which plugins to activate in their scheduling profiles.
- Multiple instances of the same plugin type may coexist with different names
  and parameters.
- Removing a plugin from config disables its behavior without breaking the EPP.

## Builtins

A **builtin** is a component that:

- **Is always enabled** and wired directly by the EPP runner or request-control
  layer.
- **Cannot be disabled or swapped** by users through config.
- **Does not implement `plugin.Plugin`** and does not have a `TypedName()`.
- **Is not registered** in the plugin registry.
- **Lives outside** `pkg/epp/framework/plugins/`, typically in the subsystem it
  belongs to (e.g. `pkg/epp/flowcontrol/`, `pkg/epp/requestcontrol/`).

### Examples of builtins

- **RequestEvictor** (`pkg/epp/flowcontrol/eviction/`): tracks in-flight
  requests and executes eviction decisions. It accepts pluggable **policies**
  (ordering, filtering) but the evictor runtime itself is always present.
- **Admission control** wiring in `pkg/epp/requestcontrol/`.
- **Flow control** infrastructure in `pkg/epp/flowcontrol/`.

### Key characteristics

- The EPP **constructs** the builtin during startup; no config entry needed.
- The builtin may **consume plugins** (e.g. `RequestEvictor` takes an
  `EvictionOrderingPolicy` plugin and an `EvictionFilterPolicy` plugin), but
  it is not itself a plugin.
- Removing or disabling a builtin would break core EPP behavior.

## Decision checklist

When adding a new component, ask:

1. **Should a user be able to turn this off?** If yes, it is a plugin.
2. **Does the EPP break without it?** If yes, it is a builtin.
3. **Is it a policy/strategy that can vary across deployments?** Plugin.
4. **Is it infrastructure that executes those policies?** Builtin.

If in doubt, start as a builtin. Promoting a builtin to a plugin later is
straightforward; demoting a plugin to a builtin requires removing it from the
config API and is a breaking change.

## Directory layout

```
pkg/epp/
├── framework/
│   ├── interface/          # Interfaces for plugins AND builtins
│   └── plugins/            # Plugin implementations (user-selectable)
│       ├── scheduling/     #   Filters, scorers
│       ├── flowcontrol/    #   Eviction policies (ordering, filtering)
│       └── requestcontrol/ #   Request data producers, etc.
├── flowcontrol/            # Builtin flow-control infrastructure
│   └── eviction/           #   RequestEvictor (builtin), queue, registry
├── requestcontrol/         # Builtin request-control infrastructure
└── scheduling/             # Builtin scheduler infrastructure
```

## Registering out-of-tree plugins

An out-of-tree plugin lives in its own Go module and is compiled into a custom
EPP binary. Fork the current `main` (`cmd/epp/main.go`), import the plugin
framework (`github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin`,
imported as `fwkplugin`), and register the custom plugin before running the
runner.

### Registration functions

`pkg/epp/framework/interface/plugin/registry.go` provides four functions. Each
takes a plugin type name, a stability level, and a `FactoryFunc`.

| Function | Use it when |
|---|---|
| `Register` | The plugin needs no extra wiring. |
| `RegisterAsDefaultProducer` | The plugin produces a `DataKey`. The EPP creates an instance automatically when a configured plugin consumes that key and no producer for it is configured. |
| `RegisterWithPluginDependencies` | The plugin's parameters name other plugin instances. The `ConfigParserFunc` returns the parsed parameters, and the loader instantiates the referenced plugins first. |
| `RegisterDeprecated` | The plugin type is scheduled for removal. The loader logs a deprecation warning for each configured instance. |

Use a type name that no in-tree plugin uses, because a later registration of
the same type name replaces the earlier factory.

### Example

The binary below registers a scorer that gives every endpoint the same score:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/cmd/epp/runner"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const ConstantScorerType = "constant-scorer"

type constantScorerParams struct {
	Score float64 `json:"score"`
}

type ConstantScorer struct {
	typedName fwkplugin.TypedName
	score     float64
}

func (s *ConstantScorer) TypedName() fwkplugin.TypedName { return s.typedName }

func (s *ConstantScorer) Category() fwksched.ScorerCategory { return fwksched.Balance }

func (s *ConstantScorer) Score(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		scores[ep] = s.score
	}
	return scores
}

func ConstantScorerFactory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := constantScorerParams{Score: 0.5}
	if parameters != nil {
		if err := parameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse parameters of %s plugin: %w", ConstantScorerType, err)
		}
	}
	return &ConstantScorer{
		typedName: fwkplugin.TypedName{Type: ConstantScorerType, Name: name},
		score:     params.Score,
	}, nil
}

func main() {
	// Register custom plugins before starting the runner. If registered as
	// fwkplugin.StabilityAlpha, pass --allow-experimental-plugins at startup.
	fwkplugin.Register(ConstantScorerType, fwkplugin.StabilityBeta, ConstantScorerFactory)

	// In-tree plugins are registered as part of the runner.
	if err := runner.NewRunner().WithExecutableName("my-epp").Run(ctrl.SetupSignalHandler()); err != nil {
		os.Exit(1)
	}
}
```

Reference the plugin type in the EPP config:

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- name: constant
  type: constant-scorer
  parameters:
    score: 0.8
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: constant
    weight: 10
```
