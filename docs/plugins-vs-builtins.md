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
  repository).
- **Implements `plugin.Plugin`** (specifically the `TypedName()` method),
  allowing the framework to identify and instantiate it from config.
- **Is registered** via `plugin.Register(type, factory)` so the config parser
  can look it up by type name.

### Examples of plugins

- **Scheduling filters**: `bylabel`, `bylabelselector`, `encoderole`, etc.
- **Scheduling scorers**: `prefix`, `sessionaffinity`, `loadaware`, `loraaffinity`, etc.
- **Profile handlers**: `DataParallelProfileHandler`, `DisaggProfileHandler`, etc.
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
