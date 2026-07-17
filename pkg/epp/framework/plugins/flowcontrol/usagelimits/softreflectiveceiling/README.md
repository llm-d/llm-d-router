# Soft Reflective Ceiling Usage Limit Policy

**Type:** `soft-reflective-ceiling-policy`

A usage limit policy that applies a graduated, priority-aware dispatch ceiling. Where the [static usage limit policy](../README.md) applies one ceiling uniformly across all priorities, this policy tightens the ceiling for lower-priority bands as pool saturation rises, reserving headroom for higher-priority traffic.

The ceiling controls **dispatch**, not admission. Requests continue to be enqueued as they arrive; a gated band simply is not drawn from for dispatch on that call, so its items remain in their queues until the ceiling opens.

## Why choose this policy?

- **Continuous Rate Control**: Once a band's ceiling is reached, the policy alternates open and closed across calls so that the effective dispatch rate degrades continuously with saturation. This is unlike step-function gating (`static-usage-limit-policy`, `priority-holdback-policy`), which is either fully open or fully closed at a fixed threshold.
- **Saturation-Adaptive Ceilings**: The per-band ceilings shift with the observed saturation itself. Under light load, lower bands see permissive ceilings; under heavy load, ceilings compress toward zero. Fixed-threshold policies (`priority-holdback-policy`, `static-usage-limit-policy`) hold their thresholds regardless of current load.
- **No Policy Parameters**: Behavior is derived entirely from saturation and the active priority count. Suits deployments that prefer a single well-defined schedule over per-environment tuning.

Deployments that need to bound the lowest band's gating aggressiveness explicitly should use `priority-holdback-policy` instead.

## What it does

For each call, the policy receives the current pool `saturation` (from the configured `SaturationDetector`) and an ordered list of `priorities` where `priorities[0]` is the highest. It returns one ceiling per band. Each ceiling is compared against saturation by the Flow Controller: when saturation exceeds the ceiling, the band is gated for that call and its items are held in queue.

**Reflective ceiling per band:**

    ceiling[i] = 1 - i * saturation / (N - 1)

where `N = len(priorities)`. Band 0 receives `ceiling[0] = 1.0` and dispatches while saturation is below 1.0. At `saturation >= 1.0` the Flow Controller halts every band, including band 0, because it gates on `saturation >= ceiling`.

**Per-band decision:**

- If `saturation < ceiling[i]`: the band is fully open (`1.0`); items dispatch normally.
- If `saturation >= 1.0`: non-critical bands are fully gated (`0.0`); their items remain queued until saturation drops.
- Otherwise the band is at or past its reflective ceiling. It alternates open (`1.0`) and closed (`0.0`) across calls with

        period = round(saturation / (1 - saturation))

    so that, on average, each gated band's ceiling is open on 1 out of every `period` calls. Higher saturation lengthens the gated intervals; the effective dispatch rate approximates `(1 - saturation) / saturation`.

All gated bands share a single policy-wide tick, so on any given call either every gated band's ceiling is open or every gated band's ceiling is closed. That monotonicity across ranks is required because the Flow Controller's dispatch loop aborts at the first band whose ceiling gates. The shared tick is bounded state used only to spread dispatch evenly across calls; signal conditioning (smoothing, hysteresis, trend detection) is delegated to the Saturation Detector layer per the `UsageLimitPolicy` contract.

## Inputs consumed

This policy consumes runtime signals passed by the Flow Controller:

- **Pool Saturation**: The current saturation value from the configured `SaturationDetector` plugin.
- **Priority Bands**: The ordered list of active priorities. Only the number of bands and their rank order matter, not the numeric priority values.

## Configuration

This policy takes no parameters. Any non-empty parameters block is rejected at load time.

```yaml
plugins:
  - type: soft-reflective-ceiling-policy
flowControl:
  usageLimitPolicyPluginRef: soft-reflective-ceiling-policy
```

Unlike the static usage limit policy, this policy is **not** framework-injected. Declare it explicitly to activate it.

## Trade-offs

- **Not Stateless**: The alternation pattern is only observable across calls, so the policy keeps one policy-wide atomic tick counter -- a deliberate deviation from the `UsageLimitPolicy` contract's guidance that implementations be stateless. The state is bounded and used only for dispatch spreading; two calls with the same `(saturation, priorities)` will not always return the same result.
- **Coarse Rate Control**: The effective dispatch rate is `1 / period` for integer `period`, so the gated dispatch rate transitions in discrete steps rather than smoothly with saturation.
- **Rank-Only, Not Magnitude**: The ceiling depends only on the rank of each priority, not the numeric spacing between them. Two adjacent bands with a wide gap in priority values are treated the same as two bands with a small gap.
- **Requires Multiple Bands**: With a single band the policy degenerates to always-open. Differentiation begins at two or more bands.
- **Queue Growth Under Sustained Saturation**: Because gated items remain queued rather than being rejected, lower-priority queues can grow while saturation is high. Bounding queue size is the responsibility of the eviction plugins, not this policy.

## Related Documentation
- [Static Usage Limit Policy](../README.md)
- [Flow Control User Guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.5.0/site-src/guides/flow-control.md)
