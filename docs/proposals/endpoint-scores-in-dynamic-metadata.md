# Proposal: Return per-endpoint scores in EPP dynamic metadata

| Field | Value |
| --- | --- |
| Status | Draft / Request for comments |
| Tracking issue | [#1843](https://github.com/llm-d/llm-d-router/issues/1843) |
| Upstream contract issue | [GAIE #2985](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2985) |
| Related PR | [#1237](https://github.com/llm-d/llm-d-router/pull/1237) (DynamicMetadata encode-path regression test) |
| Affected interface | EPP -> Envoy ext-proc dynamic metadata on the request path |

## Summary

The Endpoint Picker (EPP) already communicates the chosen destination endpoint(s)
to Envoy two ways: as a request header and as an unstructured ext-proc dynamic
metadata key under the `envoy.lb` namespace. That metadata carries only the
endpoint address(es); the scores the scorer plugins computed for the candidate
endpoints are discarded inside the scheduler profile and never leave the router.

This proposal adds an opt-in extension to the dynamic metadata contract so EPP can
emit the score computed for each scored candidate endpoint, without changing the
existing endpoint-address fields that downstream consumers already depend on.

## Motivation

Scorer plugins compute a weighted score per candidate endpoint, the picker ranks
on those scores and selects a subset, and the result collapses to an ordered list
of endpoints. The scores are lost at the `ProfileRunResult` boundary, so anything
downstream of EPP only sees *which* endpoints were chosen, not *how strongly* they
were preferred nor how the alternatives compared.

Exposing the scores enables:

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
is a contract with Envoy and with gateway providers, not an internal detail. The
contract itself is tracked upstream in GAIE #2985, since the metadata namespace is
shared across EPP implementations. This document covers the llm-d-router side: how
scores reach the emission point, and which knob controls emission. The metadata key
name and value shape follow whatever GAIE #2985 settles on.

## Current state

### Data flow

1. `SchedulerProfile.runScorerPlugins` produces
   `weightedScorePerEndpoint map[Endpoint]float64`
   (`pkg/epp/scheduling/scheduler_profile.go`), covering every candidate endpoint
   the profile scored.
2. `runPickerPlugin` wraps each entry in a `ScoredEndpoint{Endpoint, Score}` and
   calls the picker.
3. The picker (e.g. `MaxScorePicker.Pick`,
   `pkg/epp/framework/plugins/scheduling/picker/maxscore/picker.go`) ranks the
   `ScoredEndpoint`s, truncates to `maxNumOfEndpoints`, and returns
   `ProfileRunResult{TargetEndpoints []Endpoint}` -- **the scores are dropped here,
   along with every candidate the picker did not select.**
4. `Director.prepareRequest` (`pkg/epp/requestcontrol/director.go`) reads
   `result.ProfileResults[primary].TargetEndpoints`, joins the addresses into a
   comma-separated string, and stores it on `reqCtx.TargetEndpoint`.
5. `StreamingServer.generateRequestHeaderResponse` /  `generateMetadata`
   (`pkg/epp/handlers/request.go`) emit the string as both a header and dynamic
   metadata.

The complete set of scores exists only at step 1. By step 3 both the scores and the
unselected candidates are gone, and `TargetEndpoints` is a subset of what was
scored -- for the default single-endpoint configuration, a subset of size one.

### Wire format today

Request-header response dynamic metadata:

```
envoy.lb:                                  # metadata.DestinationEndpointNamespace
  x-gateway-destination-endpoint: "10.0.0.1:8200,10.0.0.2:8200"   # metadata.DestinationEndpointKey (string, comma-joined)
```

Relevant constants live in `pkg/epp/metadata/consts.go`.

Two layers must change: the in-process result carrying scores out of the scheduler
profile, and the metadata emitted to Envoy.

## Goals

- Let EPP optionally emit a numeric score per scored candidate endpoint in dynamic
  metadata.
- Preserve the existing `x-gateway-destination-endpoint` field and header exactly as
  they are today (backward compatible; no consumer is forced to change).
- Source the scores where they are complete, independent of which picker is
  configured.

## Non-goals

- Changing how scores are *computed* (scorers, weights, `enforceScoreRange`).
- Changing the request header (`x-gateway-destination-endpoint`) format.
- Defining gateway-side behavior. How a consumer uses the scores is out of scope;
  this proposal only defines what EPP emits.
- Surfacing scores from non-primary profiles. A consumer acts on the primary
  endpoint decision; per-profile scores would bind gateways to EPP's internal
  profile configuration. Cases that genuinely need them, such as selecting
  prefill/decode pairs for fallback, warrant their own proposal and wire format.
- The response-path metadata (`reqCtx.Response.DynamicMetadata`, e.g.
  `x-gateway-inference-request-cost`) covered by #1237 is unchanged.

## Design

### 1. Capture scores in the scheduler profile

Scores are recorded where `weightedScorePerEndpoint` is complete, before the picker
runs, so their availability does not depend on the configured picker:

```go
// pkg/epp/framework/interface/scheduling/types.go
type ProfileRunResult struct {
    TargetEndpoints []Endpoint
    // ScoredCandidates carries the weighted score for every endpoint the profile
    // scored, including candidates the picker did not select. Ordering is
    // unspecified; consumers key by endpoint. Nil when the profile ran no scorers.
    ScoredCandidates []ScoredEndpoint
}
```

`SchedulerProfile.Run` populates it from `weightedScorePerEndpoint`. Pickers are
unchanged, and existing consumers that read `TargetEndpoints` keep working.

Alternatives considered for this layer:

- **Attach scores to the picker's selection** (a `ScoredEndpoints` slice parallel to
  `TargetEndpoints`). Pickers truncate to `maxNumOfEndpoints`, so this carries only
  the selected endpoints -- one score under the default configuration. It cannot
  express how the alternatives compared, which is the primary use case. Rejected.
- **Let each picker decide whether to surface scores.** Makes the metadata contract
  depend on picker implementation, so the same deployment emits different data for
  different picker configurations, and truncated candidates remain unrecoverable.
  Rejected.
- **Replace `TargetEndpoints` with `[]ScoredEndpoint`.** Cleaner long term but
  touches every picker, profile handler, and director read site at once. Heavier
  than the interface question warrants for a first step.
- **Side channel on the endpoint metadata.** Stuffing the score into
  `EndpointMetadata` conflates "what the endpoint is" with "what this request's
  scorers thought of it." Rejected.

### 2. Wire format

#### Metadata, not header

Scores are emitted as dynamic metadata only:

- Dynamic metadata stays inside Envoy. Request headers are forwarded to the
  model-server pod, which would leak EPP scoring internals to backends and add
  bytes to every upstream request.
- Metadata is structured (`structpb`). A header would require flattening scores
  into a string, introducing a parsing contract on a surface that has none today.
- Gateway providers already consume the endpoint decision through the metadata
  path, so scores land next to the field they qualify.

#### Shape

Emit under the existing `envoy.lb` namespace as a new, additive key so the endpoint
address field is untouched. Two shapes are on the table:

**Option A -- scores map (recommended).** Add a sibling key whose value is a struct
mapping endpoint address to score:

```
envoy.lb:
  x-gateway-destination-endpoint: "10.0.0.1:8200,10.0.0.2:8200"   # unchanged
  x-gateway-destination-endpoint-scores:                          # new, struct
    "10.0.0.1:8200": 0.91
    "10.0.0.2:8200": 0.74
```

- Pro: existing field byte-identical; consumers opt in by reading the new key.
- Pro: lookup by address is direct.
- Ranking is not encoded in the struct (structpb maps are unordered). This follows
  from the compatibility decision rather than working against it:
  `x-gateway-destination-endpoint` remains the source of truth for the selection and
  its order, and the scores serve reporting and fallback decisions.

**Option B -- structured ranked list.** An explicit ordered list of
`{endpoint, score}` entries under a new key, leaving the old string field in place
for compatibility:

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
- Con: more verbose per entry.

Both shapes repeat the endpoint addresses alongside the existing string field, so
address duplication does not distinguish them. Recommendation: **Option A** as the
minimal first step, with Option B noted as the path if richer per-endpoint payloads
are wanted later.

New constant in `pkg/epp/metadata/consts.go`, e.g.
`DestinationEndpointScoresKey = "x-gateway-destination-endpoint-scores"`. The key
name and shape are the shared half of this proposal and are settled in GAIE #2985;
llm-d-router follows that outcome rather than establishing the format on its own.

#### Number format

`structpb` numbers are doubles and the weighted aggregate is a `float64` already
range-enforced to [0,1], so EPP introduces no precision loss. Observability and
debugging consumers are unaffected. Fallback and hedging consumers comparing two
scores should use an explicit epsilon rather than exact equality, since equal
scores are tie-broken by shuffling in the picker and carry no ordering meaning.

### 3. Plumbing

- `Director.prepareRequest`: when `ScoredCandidates` is populated, carry the
  endpoint->score pairs onto a new `reqCtx` field (parallel to `TargetEndpoint`),
  e.g. `reqCtx.TargetEndpointScores`.
- `generateMetadata`: when scores are present and emission is enabled, add the new
  key to the `envoy.lb` struct. When absent, emit exactly today's metadata.

### Enablement

Score emission is opt-in through EPP configuration (flag or config file), default
off, so the wire format is unchanged for existing deployments. It is not an
InferencePool API field: the scores contract is scoped to llm-d rather than GAIE,
and a shared CRD should not carry a field the upstream contract does not define.

Emitting every scored candidate makes the metadata size scale with pool size. The
opt-in default bounds that exposure; a cap on the number of endpoints emitted is a
possible refinement (see open questions).

## Backward compatibility

- `x-gateway-destination-endpoint` (header and metadata) is unchanged.
- `ProfileRunResult.TargetEndpoints` is unchanged; `ScoredCandidates` is additive
  and nil-safe.
- With emission disabled, the dynamic metadata is byte-for-byte what it is today.

## Testing

- Unit, profile layer: `SchedulerProfile.Run` populates `ScoredCandidates` from
  `weightedScorePerEndpoint` for every scored candidate, including endpoints the
  picker does not select, and independent of the configured picker and its
  `maxNumOfEndpoints`.
- Unit, metadata layer: extend the `generateMetadata` / encode-path coverage (the
  surface #1237 added tests for) to assert the new key is present with correct
  values when scores exist and emission is enabled, and absent otherwise.
- Director: scores propagate from `ProfileRunResult` to `reqCtx` and into the
  request-header response.

## Open questions

1. **Wire shape.** Option A vs. Option B above. Decided upstream in GAIE #2985;
   listed here because the recommendation in this document is an input to that
   discussion, not a local decision.
2. **Cardinality.** Should emission cap the number of scored endpoints for large
   pools, and if so by count or by score threshold?

## References

- `pkg/epp/handlers/request.go` -- `generateMetadata`, `generateRequestHeaderResponse`
- `pkg/epp/metadata/consts.go` -- metadata namespace/key constants
- `pkg/epp/scheduling/scheduler_profile.go` -- scoring and picker invocation
- `pkg/epp/framework/interface/scheduling/types.go` -- `ScoredEndpoint`, `ProfileRunResult`
- `pkg/epp/framework/plugins/scheduling/picker/maxscore/picker.go` -- reference picker
- `pkg/epp/requestcontrol/director.go` -- `prepareRequest` endpoint plumbing
- [docs/architecture.md](../architecture.md) -- filters, scorers, pickers, profiles
- [GAIE #2985](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2985) -- upstream dynamic metadata contract discussion
