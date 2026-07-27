# HeaderProfileHandler

**Type:** `header-profile-handler`

Runs exactly one scheduling profile per request: the one named by the value of a request
header. This lets a single EPP instance serve several profiles of a disaggregated
pipeline (e.g. `encode`, `prefill`, `decode`) whose caller already knows, out of band,
which profile each request is for, instead of needing one EPP instance per profile.

## What it does

Reads the configured header from the incoming request and looks up the
`schedulingProfiles` entry with that exact name:

- With exactly one profile configured, that profile always runs, regardless of the
  header (or its absence). There is nothing else to choose, so a deployment
  scaled down to a single stage -- or one that never chooses at all -- works
  without swapping to a different profile handler.
- With more than one profile configured and a matching profile named by the header, it
  runs that profile alone.
- With more than one profile configured and the header missing or blank, `defaultProfile`
  runs instead of failing. This covers calls that never carry the header at all, such as
  pass-through requests (e.g. `/models`) that don't go through profile-tagged scheduling.
- With more than one profile configured and the header naming a profile that isn't
  configured, no profile runs -- an unrecognized value is a real error, not treated the
  same as an absent header. The scheduler reports that no profile could be run at all,
  without a reason, which the EPP maps to a 429 response to the client -- misleading,
  since the scheduler doesn't distinguish a malformed request from exhausted capacity.
  The EPP logs the specific reason (missing header vs. unconfigured value) for operators.
  A 400 would be the accurate status here, but `Pick` has no way to carry a typed error
  through the scheduler to the client; that needs the same kind of `ProfileHandler`
  extension-point signature change tracked for the conditional-decode gate in
  [llm-d/llm-d-router#1686](https://github.com/llm-d/llm-d-router/issues/1686).

The header value is matched verbatim against `schedulingProfiles[].name`; keeping the
header values a caller sends in sync with the profiles actually configured is a
deployment-time convention today, not something this plugin validates. Automatic
validation (e.g. at plugin construction) is future work.

## How this differs from disagg-profile-handler

Both are `scheduling.ProfileHandler` implementations, but they answer a different
question. [`disagg-profile-handler`](../disagg/README.md) answers "which stages does
*this* request need?" for a caller that makes one scheduling call per request and needs
every needed pod picked up front. `header-profile-handler` answers "which single stage is
*this specific call* for?" for a caller (the coordinator) that already knows the answer
and makes one separate scheduling call per profile.

| | `header-profile-handler` | `disagg-profile-handler` |
|---|---|---|
| Selection signal | A request header naming the profile | Decider plugins, evaluated per optional stage |
| Profiles per request | Exactly one, ever | Decode always, plus encode/prefill when their decider approves -- up to three |
| Scheduling calls per request | One per profile (caller drives the cascade) | One cycle picks every stage the request needs |
| Primary profile | Whichever profile the header named | Always decode |
| `requestcontrol.PreRequest` | Not implemented -- nothing downstream reads pod addresses from headers | Implemented: stamps `x-prefiller-host-port` / `x-encoder-hosts-ports` for the decode sidecar |
| Fits | The coordinator model, which tracks cross-profile state itself | The sidecar model (llm-d-router), where the decode sidecar orchestrates the remaining hops |

## Configuration

### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `headerName` | string | `EPP-Profile` | Request header whose value names the scheduling profile to run. Matched case-insensitively: the EPP lowercases every incoming header name, so this is normalized to lowercase regardless of how it's written here. |
| `defaultProfile` | string | `decode` | Scheduling profile to run when the header is missing or blank and more than one profile is configured. Matched case-sensitively against `schedulingProfiles` names, like the header value itself. Ignored when only one profile is configured, since that profile always runs. |

### Example

```yaml
plugins:
- type: encode-filter
- type: prefill-filter
- type: decode-filter
- type: header-profile-handler
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

A request with `EPP-Profile: prefill` runs only the `prefill` profile. A request with no
`EPP-Profile` header at all -- e.g. `GET /models` -- runs `decode`, the default.

To use a different fallback than `decode`:

```yaml
- type: header-profile-handler
  parameters:
    defaultProfile: prefill
```
