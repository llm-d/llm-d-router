# Max Saturation Detector Plugin

**Type:** `max-saturation-detector`

Composite saturation detection that combines the signals of other saturation detector plugins.

## What it does

The plugin implements the `SaturationDetector` interface by delegating to a configured list of child detectors and reporting the maximum of their values. Flow control then gates dispatch on whichever load signal is the most constrained, for example in-flight concurrency (`concurrency-detector`) and scraped queue depth (`utilization-detector`) evaluated together.

Each child's signal is also exported through the `flow_control_detector_saturation` gauge, labeled by the detector reference name and the pipeline stage (`prefill` or `decode`), so operators can tell which signal is driving `flow_control_pool_saturation` in each stage.

The plugin implements only the `SaturationDetector` interface:

- It does not implement the scheduling `Filter` extension point. The config loader auto-injects a gating detector into scheduling profiles only when it implements `Filter`, so per-endpoint filtering stays with the child detectors, which are listed in profiles explicitly.
- It does not declare data dependencies of its own. The children are configured plugins, so the framework validates their data dependencies directly.

## Configuration

The plugin accepts JSON parameters decoding to the following fields:

- `detectors` (`[]string`): Names of the child saturation detector plugins to combine. Each entry must reference a plugin that implements `SaturationDetector`. The config loader instantiates the referenced plugins before the composite, so their position in the plugins list does not matter. Missing, duplicate, or wrong-type references are rejected at startup. At least one entry is required.
- `stages` (`map[string][]string`, optional): Restricts children to pipeline stages (`prefill`, `decode`), keyed by the child's entry in `detectors`. A child without an entry is evaluated for every stage. When flow control evaluates a stage, children scoped to other stages are skipped, and a stage with no child in scope reports 0, so it does not gate dispatch. Every child is evaluated when the endpoints are not partitioned by stage. Entries that are not listed in `detectors`, name no stage, or name an unsupported stage are rejected at startup. Stage scoping applies to pool-level `Saturation` only. Per-endpoint `Filter` behavior is unchanged wherever the child is listed in scheduling profiles

## Example

```yaml
plugins:
- name: pool-concurrency
  type: concurrency-detector
  parameters:
    maxConcurrency: 32
- name: pool-queue
  type: utilization-detector
  parameters:
    queueDepthThreshold: 5
    kvCacheUtilThreshold: 1.0
- name: pool-saturation
  type: max-saturation-detector
  parameters:
    detectors: [pool-concurrency, pool-queue]
flowControl:
  saturationDetector:
    pluginRef: pool-saturation
```

A complete P/D configuration using this plugin is in [`deploy/config/pd-hybrid-saturation-epp-config.yaml`](../../../../../../../deploy/config/pd-hybrid-saturation-epp-config.yaml).
