# HeaderPhaseProfileHandler

**Type:** `header-phase-profile-handler`

Runs exactly one scheduling profile per request: the one named by the value of a request
header. This lets a single EPP instance serve several phases of a disaggregated pipeline
(e.g. `encode`, `prefill`, `decode`) whose caller already knows, out of band, which phase
each request is for, instead of needing one EPP instance per phase.

## What it does

Reads the configured header from the incoming request and looks up the
`schedulingProfiles` entry with that exact name:

- If a matching profile hasn't run yet, it runs that profile alone.
- If the header is missing or names a profile that isn't configured, no profile runs. The
  scheduler reports that no profile could be run at all, without a reason, which the EPP
  maps to a 429 response to the client -- misleading, since the scheduler doesn't
  distinguish a malformed request from exhausted capacity. The EPP logs the specific
  reason (missing header vs. unconfigured value) for operators.

## How this differs from disagg-profile-handler

Both are `scheduling.ProfileHandler` implementations, but they answer a different
question. [`disagg-profile-handler`](../disagg/README.md) answers "which stages does
*this* request need?" for a caller that makes one scheduling call per request and needs
every needed pod picked up front. `header-phase-profile-handler` answers "which single
stage is *this specific call* for?" for a caller (the coordinator) that already knows the
answer and makes one separate scheduling call per phase.

| | `header-phase-profile-handler` | `disagg-profile-handler` |
|---|---|---|
| Selection signal | A request header naming the profile | Decider plugins, evaluated per optional stage |
| Profiles per request | Exactly one, ever | Decode always, plus encode/prefill when their decider approves -- up to three |
| Scheduling calls per request | One per phase (caller drives the cascade) | One cycle picks every stage the request needs |
| Primary profile | Whichever profile the header named | Always decode |
| `requestcontrol.PreRequest` | Not implemented -- nothing downstream reads pod addresses from headers | Implemented: stamps `x-prefiller-host-port` / `x-encoder-hosts-ports` for the decode sidecar |
| Fits | The coordinator model, which tracks cross-phase state itself | The sidecar model (llm-d-router), where the decode sidecar orchestrates the remaining hops |

## Configuration

### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `headerName` | string | `EPP-Phase` | Request header whose value names the scheduling profile to run. Matched case-insensitively: the EPP lowercases every incoming header name, so this is normalized to lowercase regardless of how it's written here. |

### Example

```yaml
plugins:
- type: encode-filter
- type: prefill-filter
- type: decode-filter
- type: header-phase-profile-handler
schedulingProfiles:
- name: encode
  plugins:
  - pluginRef: encode-filter
- name: prefill
  plugins:
  - pluginRef: prefill-filter
- name: decode
  plugins:
  - pluginRef: decode-filter
```

A request with `EPP-Phase: prefill` runs only the `prefill` profile.
