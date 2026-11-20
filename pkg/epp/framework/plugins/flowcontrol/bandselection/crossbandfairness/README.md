# Cross-Band Fairness Policy Plugin

**Type:** `cross-band-fairness-policy`

A band selection policy that shares dispatch opportunities across priority bands in proportion to configured weights once the pool is saturated.

This can be used in place of the default strict band selection policy (which serves bands from highest to lowest priority) to bound the starvation of lower-priority traffic under sustained load.

## What It Does

Below `activationThreshold`, bands are offered a dispatch opportunity from highest to lowest priority, matching strict band selection: while capacity is available there is nothing to arbitrate.

At or above `activationThreshold`, weighted fair queuing engages. Each band carries a virtual time measuring the service it has already claimed, and bands are offered opportunities in ascending virtual time, so the band furthest behind its configured share goes first. A dispatch advances the serving band's virtual time by `1 / weight`, so a band with a small weight falls behind faster and waits longer for its next turn. Over a sustained round the share each band receives converges on its weight relative to the total.

Ranking produces an order, not a single winner. The Flow Controller walks that order and dispatches from the first band that yields an item, so a band that is gated, empty, or fails to dispatch costs only its position.

### Virtual time across transitions

- **Saturation crossing the threshold.** Virtual times are retained while the policy is inactive and are not advanced. Saturation oscillating around the threshold resumes the round in progress rather than restarting it.
- **Idle bands.** A band ranked ahead of the band that dispatched was offered the opportunity and did not take it, so it is levelled with the band that was served. An idle band therefore banks no claim, and when its traffic returns it takes its turn without a burst of catch-up dispatches.
- **Gated bands.** A band held back by the usage limit policy is ranked last because it cannot dispatch, and its virtual time tracks the system virtual time: the smallest virtual time among the bands in service, which is the position the next band to be served occupies. It is first in line the moment it is released, without claiming the service it was held back from.
- **Band churn.** A band provisioned mid-round enters at the system virtual time on the same terms. A band that is garbage-collected is dropped, leaving the shares of the remaining bands unchanged.

## Configuration

### `activationThreshold`

The saturation level at or above which weighted fair queuing engages. Required, with no default: it expresses the operating point at which an operator considers contention worth arbitrating, and there is no universally correct value.

### `schedule`

The mechanism that spreads dispatch opportunities over time. Currently only `"interleaved"` is supported: virtual-time scheduling, spreading opportunities evenly across bands.

### `distribution`

How a weight is derived for each priority band.

#### `linear` (default)

Weights decrease linearly by ordinal position, ignoring numerical priority values.

    w_i = N - i

Where `i` is the index in descending priority order (0 = highest) and `N` is the count of active priorities. With three active bands the shares are 3:2:1.

Use when priorities represent ordinal categories (e.g., "critical", "normal", "batch") where the numerical values are arbitrary labels.

#### `explicit`

Maps priorities to dispatch weights using operator-supplied priority anchors in the `ratios` map.

Unlisted priorities are resolved as follows:
- Priorities between two configured anchors are linearly interpolated over numerical priority value (not ordinal rank). Value-based interpolation makes the mapping a total function from priority to weight, so shares remain stable regardless of dynamic priority band churn.
- Priorities outside the configured anchor range are clamped to the nearest boundary anchor.
- When only a single anchor is configured, all priorities receive that anchor's weight, which is an equal share each.

Only the proportions between weights matter, so `ratios` need not sum to any particular total.

Use when hand-tuned, per-band shares are required and the linear distribution does not produce the desired split.

**Constraints:**
- `ratios` (map of integer priority to float64) is required and every weight must be strictly positive.
- `ratios` must not be set when `distribution` is not `explicit`; it is rejected during configuration validation.

**Parameters:**

- `activationThreshold` (float64, required, no default): Saturation at which weighted fair queuing engages. Must be in `[0.0, 1.0)`.
- `schedule` (string, optional, default: `"interleaved"`): Dispatch spreading mechanism. Currently only `"interleaved"` is supported.
- `distribution` (string, optional, default: `"linear"`): Weight derivation: `"linear"` or `"explicit"`.
- `ratios` (map[int]float64, required when distribution is `"explicit"`): Maps priority levels (anchors) to their dispatch weights.

**Configuration Example (linear):**
```yaml
plugins:
  - type: cross-band-fairness-policy
    name: my-cross-band-fairness
    parameters:
      activationThreshold: 0.8
      distribution: linear
flowControl:
  bandSelectionPolicyPluginRef: my-cross-band-fairness
```

**Configuration Example (explicit):**
```yaml
plugins:
  - type: cross-band-fairness-policy
    name: my-explicit-fairness
    parameters:
      activationThreshold: 0.8
      distribution: explicit
      ratios:
        100: 6
        50: 3
        10: 1
flowControl:
  bandSelectionPolicyPluginRef: my-explicit-fairness
```

---

## Related Documentation
- [Priority Holdback Policy](../../usagelimits/priorityholdback/README.md)
- [Flow Control Overview](../../fairness/README.md)
