# Proposal: Return per-endpoint scores in EPP dynamic metadata

| Field | Value |
| --- | --- |
| Status | Draft / Request for comments |
| Tracking issue | [#1843](https://github.com/llm-d/llm-d-router/issues/1843) |
| Related PR | [#1237](https://github.com/llm-d/llm-d-router/pull/1237) (DynamicMetadata encode-path regression test) |
| Affected interface | EPP -> Envoy ext-proc dynamic metadata on the request path |

## Summary

The Endpoint Picker (EPP) already communicates the chosen destination endpoint(s)
to Envoy two ways: as a request header and as an unstructured ext-proc dynamic
metadata key under the `envoy.lb` namespace. Today that metadata carries only the
endpoint address(es); the score the scheduler computed for each endpoint is
discarded inside the picker and never leaves the router.

This proposal adds an opt-in extension to the dynamic metadata contract so EPP can
emit the picker's score alongside each returned endpoint, without changing the
existing endpoint-address fields that downstream consumers already depend on.

## Motivation

The scheduler computes a weighted score per candidate endpoint, the picker ranks
on it, and then collapses the result to an ordered list of endpoints. The score is
lost at `runPickerPlugin` / `Pick`, so anything downstream of EPP only sees *which*
endpoints were chosen, not *how strongly* they were preferred.

Exposing the score enables:

- **Observability.** Operators and request tracing can record why an endpoint was
  selected and how close the runner-up was, without re-deriving scorer state.
- **Downstream re-ranking and fallback.** Gateway providers that integrate via the
  metadata path (the integration point that already motivates the dual
  header/metadata contract) can make load-aware fallback or hedging decisions over
  a ranked, scored set rather than a flat ordered list.
- **Debuggability of scoring changes.** Score visibility at the gateway boundary
  makes it possible to validate scorer/picker behavior in a live fleet from outside
  the router process.

This is an interface question for the community precisely because dynamic metadata
is a contract with Envoy and with gateway providers, not an internal detail.

## Current state

### Data flow

1. `SchedulerProfile.runScorerPlugins` produces
   `weightedScorePerEndpoint map[Endpoint]float64`
   (`pkg/epp/scheduling/scheduler_profile.go`).
2. `runPickerPlugin` wraps each entry in a `ScoredEndpoint{Endpoint, Score}` and
   calls the picker.
3. The picker (e.g. `MaxScorePicker.Pick`,
   `pkg/epp/framework/plugins/scheduling/picker/maxscore/picker.go`) ranks the
   `ScoredEndpoint`s but returns
   `ProfileRunResult{TargetEndpoints []Endpoint}` -- **the score is dropped here.**
4. `Director.prepareRequest` (`pkg/epp/requestcontrol/director.go`) reads
   `result.ProfileResults[primary].TargetEndpoints`, joins the addresses into a
   comma-separated string, and stores it on `reqCtx.TargetEndpoint`.
5. `StreamingServer.generateRequestHeaderResponse` /  `generateMetadata`
   (`pkg/epp/handlers/request.go`) emit the string as both a header and dynamic
   metadata.

### Wire format today

Request-header response dynamic metadata:

```
envoy.lb:                                  # metadata.DestinationEndpointNamespace
  x-gateway-destination-endpoint: "10.0.0.1:8200,10.0.0.2:8200"   # metadata.DestinationEndpointKey (string, comma-joined)
```

Relevant constants live in `pkg/epp/metadata/consts.go`.

The first structural gap: the score never survives step 3, so **no plumbing change
at the picker boundary, no metadata change downstream**. Both halves are needed.

## Goals

- Let EPP optionally emit a numeric score per returned endpoint in dynamic metadata.
- Preserve the existing `x-gateway-destination-endpoint` field and header exactly as
  they are today (backward compatible; no consumer is forced to change).
- Keep the scoring path pluggable: a custom picker decides whether/what scores to
  surface.

## Non-goals

- Changing how scores are *computed* (scorers, weights, `enforceScoreRange`).
- Changing the request header (`x-gateway-destination-endpoint`) format. Headers
  are a flat string surface; scores belong in structured metadata only.
- Defining gateway-side behavior. How a consumer uses the scores is out of scope;
  this proposal only defines what EPP emits.
- The response-path metadata (`reqCtx.Response.DynamicMetadata`, e.g.
  `x-gateway-inference-request-cost`) covered by #1237 is unchanged.

## Design

Two layers change: the in-process result carrying scores out of the picker, and the
wire format emitted to Envoy.

### 1. Carry scores out of the picker

`ProfileRunResult` currently exposes only `TargetEndpoints []Endpoint`. Add scores
without breaking existing readers. Preferred option:

```go
// pkg/epp/framework/interface/scheduling/types.go
type ProfileRunResult struct {
    TargetEndpoints []Endpoint
    // ScoredEndpoints, when set, is parallel to TargetEndpoints and carries the
    // picker's score for each. Pickers that do not score MAY leave it nil.
    ScoredEndpoints []ScoredEndpoint
}
```

`ScoredEndpoint{Endpoint, Score}` already exists, so pickers like `MaxScorePicker`
have the scored, sorted slice in hand and only need to stop discarding it. Existing
consumers that read `TargetEndpoints` keep working unchanged.

Alternatives considered for this layer:

- **Replace `TargetEndpoints` with `[]ScoredEndpoint`.** Cleaner long term but
  touches every picker, profile handler, and director read site at once. Heavier
  than the interface question warrants for a first step.
- **Side channel on the endpoint metadata.** Stuffing the score into
  `EndpointMetadata` conflates "what the endpoint is" with "what this request's
  picker thought of it." Rejected.

### 2. Wire format

Emit scores under the existing `envoy.lb` namespace as a new, additive key so the
endpoint address field is untouched. Two shapes are on the table:

**Option A -- parallel scores map (recommended).** Add a sibling key whose value is
a struct mapping endpoint address to score:

```
envoy.lb:
  x-gateway-destination-endpoint: "10.0.0.1:8200,10.0.0.2:8200"   # unchanged
  x-gateway-destination-endpoint-scores:                          # new, struct
    "10.0.0.1:8200": 0.91
    "10.0.0.2:8200": 0.74
```

- Pro: existing field byte-identical; consumers opt in by reading the new key.
- Pro: lookup by address is direct.
- Con: ordering/ranking is implied by the address list, not the scores struct
  (structpb maps are unordered).

**Option B -- structured ranked list.** Replace the implied parallelism with an
explicit ordered list of `{endpoint, score}` entries under a new key, leaving the
old string field in place for compatibility:

```
envoy.lb:
  x-gateway-destination-endpoint: "10.0.0.1:8200,10.0.0.2:8200"   # unchanged
  x-gateway-destination-endpoints:                                # new, list of structs
    - endpoint: "10.0.0.1:8200"
      score: 0.91
    - endpoint: "10.0.0.2:8200"
      score: 0.74
```

- Pro: rank order is explicit and self-contained.
- Pro: extensible -- future per-endpoint fields (profile name, reason) slot in.
- Con: duplicates the address list across two keys.

Recommendation: **Option A** as the minimal first step (smallest diff, strongest
compatibility), with Option B noted as the path if richer per-endpoint payloads are
wanted later. Both keep `x-gateway-destination-endpoint` as the source of truth for
the address list.

New constant in `pkg/epp/metadata/consts.go`, e.g.
`DestinationEndpointScoresKey = "x-gateway-destination-endpoint-scores"`.

### 3. Plumbing

- `Director.prepareRequest`: when `ScoredEndpoints` is populated, carry the
  endpoint->score pairs onto a new `reqCtx` field (parallel to `TargetEndpoint`),
  e.g. `reqCtx.TargetEndpointScores`.
- `generateMetadata`: when scores are present and emission is enabled, add the new
  key to the `envoy.lb` struct. When absent, emit exactly today's metadata.

### Enablement

Score emission should be opt-in to avoid changing the wire format for existing
deployments by default. Open question (below) on the exact knob.

## Backward compatibility

- `x-gateway-destination-endpoint` (header and metadata) is unchanged.
- `ProfileRunResult.TargetEndpoints` is unchanged; `ScoredEndpoints` is additive and
  nil-safe.
- With emission disabled, the dynamic metadata is byte-for-byte what it is today.

## Testing

- Unit, picker layer: `MaxScorePicker` (and at least one other) populates
  `ScoredEndpoints` consistently with `TargetEndpoints`.
- Unit, metadata layer: extend the `generateMetadata` / encode-path coverage (the
  surface #1237 added tests for) to assert the new key is present with correct
  values when scores exist, and absent when they do not.
- Director: scores propagate from `ProfileRunResult` to `reqCtx` and into the
  request-header response.

## Open questions

1. **Emission control.** Per-pool config flag, per-request header opt-in, or always
   on? Leaning config flag, default off.
2. **Which score?** The post-weight aggregate (`weightedScorePerEndpoint`,
   already range-enforced to [0,1]) vs. raw per-scorer scores. Aggregate is the
   natural first cut; per-scorer breakdown is a larger contract.
3. **Wire shape.** Option A vs. Option B above.
4. **Number format.** structpb only has `double`. Any precision/normalization
   guarantees consumers should be able to rely on?
5. **Profiles.** Should scores from non-primary profiles ever be surfaced, or only
   the primary profile that sets the destination?

## References

- `pkg/epp/handlers/request.go` -- `generateMetadata`, `generateRequestHeaderResponse`
- `pkg/epp/metadata/consts.go` -- metadata namespace/key constants
- `pkg/epp/scheduling/scheduler_profile.go` -- scoring and picker invocation
- `pkg/epp/framework/interface/scheduling/types.go` -- `ScoredEndpoint`, `ProfileRunResult`
- `pkg/epp/framework/plugins/scheduling/picker/maxscore/picker.go` -- reference picker
- `pkg/epp/requestcontrol/director.go` -- `prepareRequest` endpoint plumbing
- [docs/architecture.md](../architecture.md) -- filters, scorers, pickers, profiles
