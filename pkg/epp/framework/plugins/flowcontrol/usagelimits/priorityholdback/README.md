# Priority Holdback Policy Plugin

**Type:** `priority-holdback-policy`

A usage limit policy that computes differentiated admission ceilings per priority level. As pool saturation rises, lower-priority traffic is gated first, reserving capacity for higher-priority work.

This can be used in place the default static usage limit policy (which applies a single ceiling to all priorities) with priority-aware stepped gating.

## What It Does

Each active priority level receives an admission ceiling in [0.0, 1.0]. During each dispatch cycle, the Flow Controller compares current pool saturation against each priority's ceiling. When saturation exceeds a priority's ceiling, that priority's traffic is held back.

Higher-priority traffic continues to flow until saturation reaches its own (higher) ceiling, providing quality-of-service differentiation under load.

## Configuration

Behavior is configured via two independent parameters: `shape` and `domain`. The resulting ceilings always range from `maxCeiling` for the highest priority to `minCeiling` for the lowest.

### `shape`

The interpolation curve used to distribute ceilings across the range. Currently only `"linear"` is supported.

### `domain`

How priority levels are mapped to positions in the ceiling range.

#### `rank` (default)

Distributes ceilings in equal steps by ordinal rank, ignoring numerical priority values. Ceilings are evenly spaced across `[minCeiling, maxCeiling]`.

    c_i = maxCeiling - i * (maxCeiling - minCeiling) / (N - 1)

Where `i` is the index in descending priority order (0 = highest) and `N` is the count of active priorities.

Use when priorities represent ordinal categories (e.g., "critical", "normal", "batch") where the numerical values are arbitrary labels.

#### `value`

Scales ceilings proportionally to the numerical priority value within the observed active range.

    r_i = (p_i - pMin) / (pMax - pMin)
    c_i = minCeiling + r_i * (maxCeiling - minCeiling)

Use when the numerical spacing between priority values carries meaning and priorities that are numerically close should behave similarly under pressure.

**Parameters:**

- `shape` (string, optional, default: `"linear"`): Interpolation curve. Currently only `"linear"` is supported.
- `domain` (string, optional, default: `"rank"`): Priority mapping: `"rank"` or `"value"`.
- `minCeiling` (float64, required, no default): Ceiling for the lowest priority. Must be in `[0.0, 1.0)`.
- `maxCeiling` (float64, optional, default: `1.0`): Ceiling for the highest priority. Must be in `(0.0, 1.0]`.

`minCeiling` is required because it determines how aggressively low-priority traffic is gated and there is no universally correct default.

**Configuration Example:**
```yaml
plugins:
  - type: priority-holdback-policy
    name: my-holdback-policy
    parameters:
      shape: linear
      domain: rank
      minCeiling: 0.4
      maxCeiling: 0.9
flowControl:
  usageLimitPolicyPluginRef: my-holdback-policy
```

---

## Related Documentation
- [Static Usage Limit Policy](../README.md)
- [Flow Control Overview](../../fairness/README.md)
