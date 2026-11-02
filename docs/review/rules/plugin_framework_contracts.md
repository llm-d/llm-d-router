# Plugin framework contracts

## Keep core runtime packages free of concrete plugin logic
Do not import concrete plugin packages or switch on concrete plugin types inside `cmd/epp/runner`, `pkg/epp/datastore`, or `pkg/epp/framework/interface`. Expose shared capabilities through `fwkplugin.Handle` and small optional interfaces verified with compile-time guards (`var _ Interface = (*Type)(nil)`), resolved once at startup.

## Separate data producers from scheduling filters and scorers
Keep stateful indexing and raw endpoint extraction in data producers and place policy evaluation in scheduling plugins. Restrict `Filter` plugins to binary eligibility with fail-open bypass when optional telemetry is absent, express soft preferences as weighted `Scorer` plugins, reserve `Screener` plugins for mandatory profile-wide constraints, and leave load shedding to saturation detectors.
- Accept: Strict single-endpoint request pinning as a pre-scheduling `Screener` once weighed against `Filter`, or a shared stateless helper package for header parsing without I/O or caching.

## Exchange cross-plugin state through typed data-layer slots and keys
Pass cross-plugin request and endpoint metadata through typed `datalayer.Slot[T]` handles or `Clone()`-safe values in `datalayer.AttributeMap` instead of global singletons or direct peer calls. Declare dependencies via `Produces()` and `Consumes()`, key instance-scoped attributes by non-empty plugin names, and validate producer-consumer DAG wiring at startup after default producers resolve.

## Validate plugin parameters once in the factory constructor
Validate parameter bounds, required handles, and forbidden system headers inside the factory constructor without exposing mutable post-construction setters or repeating checks per request. Register built-in plugins in the central catalog with an explicit stability tier and enforce experimental feature gates across explicit and default instances.
- Accept: Documenting redundant plugin combinations in plugin READMEs instead of rejecting sibling plugins inside factory constructors.

## Name plugin packages with concise domain nouns and qualify config units
Name plugin packages with short lowercase domain nouns (`bylabel`, `queuedepth`, `tokenload`, `sessionaffinity`), suffix newly registered plugin type strings with their role (`-filter`, `-scorer`, `-picker`, `-producer`, `-screener`, `-decider`), and omit package-name stutter on exported identifiers (shipped unsuffixed type strings keep their names). Suffix quantitative config fields with explicit units or portable token counts rather than backend block counts, and use `resource.Quantity` for byte sizes.
- Accept: Standard data-structure names on internal types and preserving shipped public constant names across internal variable renames.

## Share common helpers and delete dead code in the stranding change
Decouple pure payload parsing from HTTP response writing and reuse shared utilities in `pkg/common` rather than duplicating logic across packages. Delete dead branches, write-only fields, unreferenced methods, and single-caller pass-through wrappers in the change that strands them.
