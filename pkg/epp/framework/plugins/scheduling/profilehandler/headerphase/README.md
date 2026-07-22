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
- If the header is missing or names a profile that isn't configured, no profile runs; the
  request fails with the scheduler's generic "failed to run any scheduler profile" error,
  and the EPP logs the specific reason (missing header vs. unconfigured value).

This differs from the [disagg profile handler](../disagg/README.md), which decides which
profiles to run via decider plugins rather than a header.

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
