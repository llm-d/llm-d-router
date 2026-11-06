# Disaggregated Profile Handler, PreRequest, and Decider Plugins

Plugins for disaggregated inference scheduling: a profile handler that selects the active stages: EPD (no disaggregation), P/D (Prefill/Decode), E/P/D (Encode/Prefill/Decode), or E/PD (Encode/Prefill-Decode), legacy headers handlers (deprecated) kept for backward compatibility, and decider plugins that control whether each disaggregation stage runs per request.

## Contents

- [Profile Handlers](#profile-handlers)
  - [DisaggProfileHandler](#disaggprofilehandler)
- [Decider Plugins](#decider-plugins)
  - [PrefixBasedPDDecider](#prefixbasedpddecider)
  - [AlwaysDisaggPDDecider](#alwaysdisaggpddecider)
  - [AlwaysDisaggMultimodalDecider](#alwaysdisaggmultimodaldecider)
  - [KVCompatPDDecider](#kvcompatpddecider)
- [Filter Plugins](#filter-plugins)
  - [KVCompatFilter](#kvcompatfilter)

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

The handler is invoked repeatedly by the framework until all stages are complete. Each optional stage is gated by a decider: if the decider returns false for a request, the stage is marked as skipped so the handler doesn't revisit it on the next invocation. If the decode stage finds no suitable endpoint, all remaining stages are skipped and the request fails.

#### Inputs consumed

- `PrefixCacheMatchInfo` — endpoint attribute read by the configured prefill decider (e.g. `prefix-based-pd-decider`) when deciding whether to run the prefill stage. The decider selects which prefix producer it reads, and the handler declares that same key as its data-layer dependency.

#### Configuration

##### Parameters
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `profiles.decode` | `string` | No | `"decode"` | Name of the decode scheduling profile. |
| `profiles.prefill` | `string` | No | `"prefill"` | Name of the prefill scheduling profile. |
| `profiles.encode` | `string` | No | `"encode"` | Name of the encode scheduling profile. |
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

- Without a configured decider, the corresponding stage is disabled for all requests — this is a static decision at startup, not per-request.
- The names in `deciders.prefill` and `deciders.encode` must match plugin names declared earlier in the same configuration.
- When using P/D disaggregation, a `PrefixCachePlugin` must be configured in the prefill and decode scheduling profiles.

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

### KVCompatPDDecider

**Type:** `kv-compat-pd-decider` (Alpha)

Declines P/D disaggregation when no prefill endpoint can exchange KV with the selected decode endpoint, and otherwise returns the answer of another decider.

#### What it does

Two vLLM instances exchange KV over NIXL only when their handshake compatibility hash matches. The hash covers the vLLM version, model, dtypes, KV cache layout and attention backend, so two GPU generations can differ even with the same model. When the hashes differ, the decode instance rejects every transfer from that prefill instance and the request fails.

Each endpoint carries a compatibility key in a label (`llm-d.ai/kv-compat-key` by default), set by whatever deploys or labels the model servers. Endpoints with equal keys can exchange KV. A decode endpoint also accepts a prefill key that is listed in `approvedPairs`, or for which it carries the label `<acceptLabelPrefix><prefillKey>=true`.

1. If no endpoint in the pool carries a key, ask the wrapped decider.
2. If the decode endpoint has no key, return false.
3. If a prefill-capable endpoint exists whose key the decode endpoint accepts, ask the wrapped decider.
4. Otherwise return false, so the request is served decode-only.

#### How It Works

The decider keeps a count of prefill-capable endpoints per key, updated from endpoint lifecycle events, so a decision reads a few map entries and does not scan the pool. Use it together with `kv-compat-filter` in the prefill profile, so the prefill profile picks one of the compatible endpoints.

##### Failed requests

A key can be wrong, or two endpoints with equal keys can still fail to exchange KV. The decider records the outcome of every disaggregated request against its pair: the prefill key and the decode endpoint, or the prefill endpoint itself when it has no key. A request counts as failed when its response status is in `failureCodes`, when its stream ends with an error, or when it ends normally after reading the prompt without generating a token. After `failureThreshold` consecutive failed requests the pair is excluded for `exclusionDuration`: the decider and the filter treat it as incompatible, so requests for that decode endpoint use another prefill endpoint or run decode-only. A successful request resets the count. Responses that arrive during an exclusion do not extend it.

When a cross-replica syncer is configured, each replica publishes its exclusions per decode endpoint and reads the union of all replicas' exclusions, so an exclusion observed by one replica applies on every replica after the next sync. Without a syncer, each replica excludes pairs from its own requests only.

#### Inputs consumed

- Endpoint labels: `llm-d.ai/role`, the key label and the accept labels, from endpoint lifecycle events and from the decode endpoint.
- Whatever the wrapped decider consumes.

#### Configuration

##### Parameters
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `decider` | `string` | Yes | | Name of the prefill decider asked when a compatible prefill endpoint exists. |
| `keyLabel` | `string` | No | `llm-d.ai/kv-compat-key` | Endpoint label holding the compatibility key. |
| `acceptLabelPrefix` | `string` | No | `accept.kv-compat.llm-d.ai/` | Prefix of the decode endpoint labels that accept another prefill key. `-` disables these labels. |
| `approvedPairs` | `[]string` | No | | `prefillKey->decodeKey` pairs that may exchange KV although their keys differ, in that direction only. |
| `failureThreshold` | `int` | No | `3` | Consecutive failed disaggregated requests that exclude a pair. `0` disables exclusion. |
| `exclusionDuration` | `duration` | No | `5m` | How long an excluded pair stays excluded. |
| `failureCodes` | `[]int` | No | `[500, 502, 503]` | Response statuses counted as a failed request. Each must be a 4xx or 5xx status. |
| `prefillProfile` | `string` | No | `prefill` | Scheduling profile that picks the prefill endpoint. |

##### Example
```yaml
plugins:
  - type: prefix-based-pd-decider
    parameters:
      nonCachedTokens: 512
  - type: kv-compat-pd-decider
    parameters:
      decider: prefix-based-pd-decider
  - type: kv-compat-filter
    parameters:
      decider: kv-compat-pd-decider
  - type: disagg-profile-handler
    parameters:
      deciders:
        prefill: kv-compat-pd-decider
schedulingProfiles:
  - name: prefill
    plugins:
      - pluginRef: prefill-filter
      - pluginRef: kv-compat-filter
      - pluginRef: max-score-picker
  - name: decode
    plugins:
      - pluginRef: decode-filter
      - pluginRef: max-score-picker
```

#### Limitations

- The key is only as accurate as the settings it is computed from. An endpoint labelled with the wrong key can still be paired with an incompatible one.
- Once any endpoint carries a key, an endpoint without one is never paired.
- Declined requests are counted in `llm_d_epp_kv_compat_prefill_declined_total`, and exclusions in `llm_d_epp_kv_compat_pair_exclusions_total`.
- A failure is counted against the pair even when its cause lies elsewhere, such as a request that fails on any path. A one-off failure does not exclude a pair, since exclusion needs `failureThreshold` consecutive failures.

---

## Filter Plugins

### KVCompatFilter

**Type:** `kv-compat-filter` (Alpha)

Keeps the prefill candidates that can exchange KV with the decode endpoint picked earlier in the cycle, using the rule of a `kv-compat-pd-decider`.

#### What it does

Reads the decode endpoint that `disagg-profile-handler` publishes before it runs the prefill profile, and keeps the candidates whose key the decode endpoint accepts and whose pair with it is not excluded. When no decode endpoint was published, or no endpoint carries a key and no pair is excluded, it keeps every candidate. It never keeps an incompatible candidate, so an empty result means no candidate can take the KV.

#### Inputs consumed

- The decode endpoint published by `disagg-profile-handler` (optional).
- Labels of the candidate and decode endpoints.

#### Configuration

##### Parameters
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `decider` | `string` | Yes | | Name of the `kv-compat-pd-decider` whose rule the filter applies. |

---

## Related Documentation

- [Disaggregation Architecture](../../../../../../../docs/disaggregation.md)
