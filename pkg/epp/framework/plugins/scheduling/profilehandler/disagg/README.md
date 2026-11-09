# Disaggregated Profile Handler, PreRequest, and Decider Plugins

Plugins for disaggregated inference scheduling: a profile handler that selects the active stages: EPD (no disaggregation), P/D (Prefill/Decode), E/P/D (Encode/Prefill/Decode), or E/PD (Encode/Prefill-Decode), legacy headers handlers (deprecated) kept for backward compatibility, and decider plugins that control whether each disaggregation stage runs per request.

## Contents

- [Profile Handlers](#profile-handlers)
  - [DisaggProfileHandler](#disaggprofilehandler)
- [Decider Plugins](#decider-plugins)
  - [PrefixBasedPDDecider](#prefixbasedpddecider)
  - [AlwaysDisaggPDDecider](#alwaysdisaggpddecider)
  - [AlwaysDisaggMultimodalDecider](#alwaysdisaggmultimodaldecider)

---

## Profile Handlers

### DisaggProfileHandler

**Type:** `disagg-profile-handler`
**Interfaces**: `scheduling.ProfileHandler`

Orchestrates up to three scheduling stages per request — decode (always), and optionally encode and prefill — based on which decider plugins are configured.

#### What it does

Runs each scheduling stage in sequence and assembles the final result from all stages that ran.

1. Run the decode profile (always).
2. If an encode decider is configured and approves the request, run the encode profile.
3. If a prefill decider is configured and approves the request, run the prefill profile.
4. Return the assembled scheduling result with decode as the primary profile.

#### How It Works

The handler is invoked repeatedly by the framework until all stages are complete. Each optional stage is gated by a decider: if the decider returns false for a request, the stage is marked as skipped so the handler doesn't revisit it on the next invocation. The optional [experimental aggregated P/D fallback](#experimental-aggregated-pd-fallback) handles unavailable required Prefill or Decode stages.

#### Inputs consumed

- `PrefixCacheMatchInfo` — endpoint attribute read by the configured prefill decider (e.g. `prefix-based-pd-decider`) when deciding whether to run the prefill stage. The decider selects which prefix producer it reads, and the handler declares that same key as its data-layer dependency.

#### Configuration

##### Parameters
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `profiles.decode` | `string` | No | `"decode"` | Name of the decode scheduling profile. |
| `profiles.prefill` | `string` | No | `"prefill"` | Name of the prefill scheduling profile. |
| `profiles.encode` | `string` | No | `"encode"` | Name of the encode scheduling profile. |
| `profiles.fallback` | `string` | No | — | Name of the aggregated fallback scheduling profile. |
| `deciders.prefill` | `string` | No | — | Name of the prefill decider plugin. When set, enables P/D disaggregation. |
| `deciders.encode` | `string` | No | — | Name of the encode decider plugin. When set, enables E disaggregation. |

##### Example

Decode-only (no disaggregation):
```yaml
plugins:
  - type: disagg-profile-handler
```

P/D disaggregation:
```yaml
plugins:
  - type: disagg-profile-handler
    parameters:
      deciders:
        prefill: prefix-based-pd-decider
```

E/P/D disaggregation:
```yaml
plugins:
  - type: disagg-profile-handler
    parameters:
      deciders:
        prefill: prefix-based-pd-decider
        encode: always-disagg-multimodal-decider
```

#### Limitations

- In `decode-first` mode, Prefill and Encode each require a decider. In `prefill-first` mode, Prefill runs when its profile is configured.
- `deciders.prefill` and `deciders.encode` must reference plugins declared in the same configuration.

#### Experimental Aggregated P/D Fallback

##### What it does

Before forwarding a request, the router tries the configured fallback profile once when the Decode pool has no Ready candidates, or when the request requires Prefill and the Prefill pool has no Ready candidates. A successful fallback selects one backup worker for the whole request.

The normal Decode and Prefill profiles use their label or role selectors to identify pool members among the Ready candidates that pass request screening. A normal profile without a pool selector counts all of those candidates as members. If a normal pool has members but health, capacity, or other profile filters reject them all, the request returns `503 rejected-no-endpoints` without trying fallback.

When fallback runs, the backup profile uses its own filters. If no pod matches, the request returns `503 rejected-no-endpoints`.

Skipping Prefill or finding no Encode target does not trigger fallback on its own. Admission and flow control keep their configured waiting and rejection behavior.

##### Configuration

1. Label the backup pods in the same endpoint pool. Use `llm-d.ai/role: prefill-decode` or a custom label such as `inference-pool: fallback`. Check that these pods can run Prefill and Decode locally, plus Encode for multimodal requests.
2. Use `label-selector-filter` to select only the intended workers in each profile. The backup profile must have a pool selector that restricts worker labels. A missing or empty selector is rejected at startup. Use dedicated selectors for the normal profiles because the built-in `decode-filter` and `prefill-filter` also select aggregated workers.
3. Set `profiles.fallback` on `disagg-profile-handler` to the backup profile name. Omit the field to disable fallback. An empty name is rejected. Fallback requires `stageOrder: decode-first`, the default.

The backup profile can use the same filters and scorers as the normal profiles. This example requires Prefill for every request and uses the default picker.

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - name: dedicated-decode
    type: label-selector-filter
    parameters: {matchLabels: {llm-d.ai/role: decode}}
  - name: dedicated-prefill
    type: label-selector-filter
    parameters: {matchLabels: {llm-d.ai/role: prefill}}
  - name: aggregated-workers
    type: label-selector-filter
    parameters: {matchLabels: {llm-d.ai/role: prefill-decode}}
  - type: always-disagg-pd-decider
  - type: disagg-profile-handler
    parameters:
      profiles: {fallback: aggregated-fallback}
      deciders: {prefill: always-disagg-pd-decider}
schedulingProfiles:
  - name: decode
    plugins: [{pluginRef: dedicated-decode}]
  - name: prefill
    plugins: [{pluginRef: dedicated-prefill}]
  - name: aggregated-fallback
    plugins: [{pluginRef: aggregated-workers}]
```

---

## Decider Plugins

### PrefixBasedPDDecider

**Type:** `prefix-based-pd-decider`

Decides per-request whether P/D disaggregation should run, based on prompt length and how much of the prompt is already cached on the selected decode pod.

#### What it does

Compares the uncached portion of the request prompt against a configurable threshold, triggering P/D disaggregation only when the uncached suffix is long enough to justify the overhead. A separate minimum prompt-length gate can skip short prompts before the prefix cache is even consulted.

1. If `nonCachedTokens` is `0`, return false (the decider is disabled).
2. Read the prompt token count as `request.Body.TokenizedPrompt.TokenCount()`.
3. If `promptTokens` is set and the prompt is shorter than it, return false.
4. Read `PrefixCacheMatchInfo` from the decode endpoint attributes.
5. Compute uncached suffix length.
6. Return true (disaggregate) if uncached tokens ≥ `nonCachedTokens`.

#### How It Works

The prompt token count is `request.Body.TokenizedPrompt.TokenCount()`, populated by a `token-producer` — auto-created with the tokenizer-free `estimate` backend when none is configured. `promptTokens` gates on this count directly: prompts shorter than it never disaggregate, regardless of cache state. Prefix cache state is read from the `PrefixCacheMatchInfo` attribute on the decode endpoint, from the prefix producer selected by `prefixMatchInfoProducerName` (the approximate-prefix producer by default). If the attribute is absent or malformed, disaggregation is skipped. Setting `nonCachedTokens: 0` disables the decider entirely (always returns false).

#### Inputs consumed

- `PrefixCacheMatchInfo` — endpoint attribute from the prefix producer named by `prefixMatchInfoProducerName` (approximate by default), read from the decode endpoint.
- `request.Body.TokenizedPrompt` — token data from a `token-producer` plugin; `TokenCount()` is the prompt token count.

#### Configuration

##### Parameters
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `nonCachedTokens` | `int` | No | `0` | Uncached token threshold above which P/D disaggregation is triggered. `0` disables the decider. |
| `promptTokens` | `int` | No | `0` | Minimum prompt token count required before disaggregation is considered. Prompts shorter than this never disaggregate. `0` disables this gate. |
| `prefixMatchInfoProducerName` | `string` | No | `approx-prefix-cache-producer` | Prefix-cache producer instance whose `PrefixCacheMatchInfo` the decider reads. A named producer other than the default must be present in the configuration. |

##### Example
```yaml
plugins:
  - type: precise-prefix-cache-producer
  - type: prefix-based-pd-decider
    parameters:
      nonCachedTokens: 512
      promptTokens: 1024
      prefixMatchInfoProducerName: precise-prefix-cache-producer
  - type: disagg-profile-handler
    parameters:
      deciders:
        prefill: prefix-based-pd-decider
```

#### Limitations

- `nonCachedTokens: 0` disables disaggregation entirely (the decider always returns false).
- `promptTokens` gates on total prompt length, not on the uncached suffix; it only ever rules out short prompts, never overrides the `nonCachedTokens` check.
- A `token-producer` populates `TokenizedPrompt`; when none is configured the framework auto-creates one with the `estimate` backend, so disaggregation works without extra setup.
- Requires `PrefixCacheMatchInfo` on the decode endpoint; if absent, disaggregation is skipped with an error log.

---

### AlwaysDisaggPDDecider

**Type:** `always-disagg-pd-decider`

Unconditionally approves P/D disaggregation for every request, regardless of cache state or prompt length.

#### What it does

Returns true for every request. Useful for testing or environments where P/D disaggregation should always run.

#### Inputs consumed

None — ignores request content and endpoint state.

#### Configuration

##### Parameters

None.

---

### AlwaysDisaggMultimodalDecider

**Type:** `always-disagg-multimodal-decider`

Approves encode disaggregation for requests that contain multimodal content (images, audio, video); passes text-only requests through without disaggregation.

#### What it does

Inspects the chat completions message content blocks for `image_url`, `video_url`, or `input_audio` types and returns true when any such block is found.

#### Inputs consumed

- Request body (`ChatCompletions.Messages`) — inspected for multimodal content blocks.

#### Configuration

##### Parameters

None.

---

## Related Documentation

- [Disaggregation Architecture](../../../../../../../docs/disaggregation.md)
