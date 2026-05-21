# Disaggregated Profile Handler, PreRequest, and Decider Plugins

Plugins for disaggregated inference scheduling: a profile handler that selects the active stages: EPD (no disaggregation), P/D (Prefill/Decode), E/P/D (Encode/Prefill/Decode), or E/PD (Encode/Prefill-Decode), legacy headers handlers (deprecated) kept for backward compatibility, and decider plugins that control whether each disaggregation stage runs per request.

## Contents

- [Profile Handlers](#profile-handlers)
  - [DisaggProfileHandler](#disaggprofilehandler)
  - [PdProfileHandler (Deprecated)](#pdprofilehandler-deprecated)
- [PreRequest Plugins](#prerequest-plugins)
  - [DisaggHeadersHandler (Deprecated)](#disaggheadershandler-deprecated)
  - [PrefillHeaderHandler (Deprecated)](#prefillheaderhandler-deprecated)
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

The handler is invoked repeatedly by the framework until all stages are complete. Each optional stage is gated by a decider: if the decider returns false for a request, the stage is marked as skipped so the handler doesn't revisit it on the next invocation. If the decode stage finds no suitable endpoint, all remaining stages are skipped and the request fails.

#### Inputs consumed

- `PrefixCacheMatchInfo` — endpoint attribute from `approx-prefix-cache-producer`, read by the configured prefill decider (e.g. `prefix-based-pd-decider`) when deciding whether to run the prefill stage.

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

### PdProfileHandler (Deprecated)

**Type:** `pd-profile-handler`
**Interfaces**: `scheduling.ProfileHandler`

> **Deprecated:** Use `disagg-profile-handler` instead.

---

## PreRequest Plugins

### DisaggHeadersHandler (Deprecated)

**Type:** `disagg-headers-handler`
**Interfaces**: `fwkrhapi.PreRequest`

> **Deprecated:** Use `disagg-profile-handler` instead.
>
> `disagg-profile-handler` now implements `fwkrhapi.PreRequest` natively.
>
> Planned removal: `v0.11`.

Sets HTTP routing headers on the outgoing request so the inference proxy can forward prefill and encode work to the selected disaggregated pods.

#### What it does

Reads the scheduling result and writes pod addresses as request headers for each disaggregated stage that ran.

1. If a prefill endpoint was selected, write its `ip:port` to `x-prefiller-host-port`.
2. If one or more encode endpoints were selected, write their comma-separated `ip:port` list to `x-encoder-hosts-ports`.
3. If a stage did not run or found no endpoints, that header is omitted.

#### Inputs consumed

- `SchedulingResult.ProfileResults` — per-profile endpoint selections produced by `disagg-profile-handler`.

#### Output produced

- `x-prefiller-host-port` request header — `<ip:port>` of the selected prefill pod; absent when P/D disaggregation was skipped.
- `x-encoder-hosts-ports` request header — comma-separated `<ip:port>` list of selected encode pods; absent when encode disaggregation was skipped.

#### Configuration

##### Parameters
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `prefillProfile` | `string` | No | `"prefill"` | Name of the profile used for prefill scheduling. Only needed if the prefill profile is not named `prefill`. |
| `encodeProfile` | `string` | No | `"encode"` | Name of the profile used for encode scheduling. Only needed if the encode profile is not named `encode`. |

##### Example
```yaml
plugins:
  - type: disagg-headers-handler
```

Custom profile names:
```yaml
plugins:
  - type: disagg-headers-handler
    parameters:
      prefillProfile: "my-prefill"
      encodeProfile: "my-encode"
```

### PrefillHeaderHandler (Deprecated)

**Type:** `prefill-header-handler`
**Interfaces**: `fwkrhapi.PreRequest`

> **Deprecated:** Use `disagg-profile-handler` instead.
>
> Planned removal: `v0.11`.

---

## Decider Plugins

### PrefixBasedPDDecider

**Type:** `prefix-based-pd-decider`

Decides per-request whether P/D disaggregation should run, based on how much of the prompt is already cached on the selected decode pod.

#### What it does

Compares the uncached portion of the request prompt against a configurable threshold, triggering P/D disaggregation only when the uncached suffix is long enough to justify the overhead.

1. Estimate the prompt token count from the request body 
2. Read `PrefixCacheMatchInfo` from the decode endpoint attributes.
3. Compute uncached suffix length.
4. Return true (disaggregate) if uncached tokens ≥ `nonCachedTokens`.

#### How It Works

Token count is estimated by dividing raw character length by 4 (a fixed approximation). Prefix cache state is read from the `PrefixCacheMatchInfo` attribute on the decode endpoint, populated by `approx-prefix-cache-producer`. If the attribute is absent or malformed, disaggregation is skipped. Setting `nonCachedTokens: 0` disables the decider entirely (always returns false).

#### Inputs consumed

- `PrefixCacheMatchInfo` — endpoint attribute from `approx-prefix-cache-producer`, read from the decode endpoint.
- Request body (prompt text or chat messages, used to estimate token count).

#### Configuration

##### Parameters
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `nonCachedTokens` | `int` | No | `0` | Uncached token threshold above which P/D disaggregation is triggered. `0` disables the decider. |

##### Example
```yaml
plugins:
  - type: prefix-based-pd-decider
    parameters:
      nonCachedTokens: 512
  - type: disagg-profile-handler
    parameters:
      deciders:
        prefill: prefix-based-pd-decider
```

#### Limitations

- `nonCachedTokens: 0` disables disaggregation entirely (the decider always returns false).
- Token count is estimated (characters ÷ 4), not exact; behavior may differ for non-ASCII content.
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

- [Disaggregation Architecture](/docs/disaggregation.md)
