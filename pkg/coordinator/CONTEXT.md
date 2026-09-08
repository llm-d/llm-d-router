# Coordinator

The sidecar-less deployment mode that orchestrates disaggregated inference. It
drives an ordered pipeline of steps and calls the gateway once per phase,
seeing the client's end-to-end request that no other component (EPP included)
observes.

## Language

**Step**:
A stage of the coordinator's internal pipeline (render, replace-media-urls,
encode, prefill, conditional-decode, decode); its timing covers the whole
stage wall time, local work plus orchestration plus any backend calls it makes.
_Avoid_: phase (when you mean a pipeline stage)

**Phase**:
A single outbound gateway call carrying the epp-profile header, one of encode /
prefill / decode; counted once per call.
_Avoid_: step (when you mean one backend call)

**Fan-out**:
The encode step issuing one concurrent phase call per image, so a single step
produces N phase calls.

**Disagg path**:
The set of phases a request actually exercised: decode-only, prefill-decode, or
encode-prefill-decode. Not a routing choice the coordinator makes; it falls out
of the configured pipeline, the request content (has images?), and cache
outcome.
_Avoid_: decision (implies an agency the coordinator does not have; the pipeline
shape is fixed at startup)

**Cache hit / miss**:
The conditional-decode probe's outcome: a 412 from the decode server is a miss
(the pipeline continues to the decode step), any success is a hit (the response
is already streamed and the pipeline stops early).

## Relationships

- A **Step** issues zero or more **Phase** calls (render and replace-media-urls
  issue none; encode fans out one per image; prefill and decode issue one each).
- **encode** always implies **prefill**: encode produces the EC transfer params
  prefill consumes, so encode-without-prefill is not a reachable **Disagg path**.
- A **Cache hit** at conditional-decode ends the pipeline before the decode
  **Step** runs.

## Example dialogue

> **Dev:** "The decode step and the decode phase, aren't those the same latency?"
> **Domain expert:** "Nearly, for decode and prefill, one step is one gateway
> call. But encode is different: the encode step fans out one phase call per
> image and waits for all of them, so the step time is the fan-out envelope, not
> any single call. That gap is the only reason to keep a separate encode-phase
> latency metric."

## Flagged ambiguities

- **encode / prefill / decode** name both a **Step** and a **Phase** — resolved:
  a Step is a pipeline stage (whole wall time, may make 0..N backend calls); a
  Phase is one outbound gateway call. render and conditional-decode have no phase
  counterpart.
- **"decision"** (as in disagg_decision_total) implied the coordinator chooses
  which phases run — resolved: it records the **Disagg path** actually exercised;
  the pipeline shape is config-static, so nothing is decided per request.
