# Diffusion Load Producer

**Type:** `diffusion-load-producer`

Tracks the outstanding declared cost of in-flight diffusion requests per endpoint and publishes it as the `DiffusionLoad` endpoint attribute, consumed by the `diffusion-cost-scorer`.

Unlike LLM requests, whose output length is unknown at admission, a diffusion request declares its compute cost in the body. Image generation is the only diffusion request type currently recognized; its declared cost is:

```
cost = num_inference_steps x width x height x n
```

Fields the client omitted fall back to the configured defaults. The cost is committed to the chosen endpoint when the request is scheduled and released when its response stream ends. Requests without a recognized diffusion body contribute no cost.

The producer is registered as the default for the `DiffusionLoad` data key, so configuring a consumer (e.g. `diffusion-cost-scorer`) auto-creates it with defaults.

**Parameters:**
- `defaultNumInferenceSteps` (int, optional, default: `50`): Step count assumed when a request omits `num_inference_steps`. Match it to the served model's default (e.g. a low value for turbo/distilled models).
- `defaultSize` (string, optional, default: `"1024x1024"`): Output resolution (`WIDTHxHEIGHT`) assumed when a request omits `size`.

**Configuration Example:**
```yaml
plugins:
  - type: diffusion-load-producer
    parameters:
      defaultNumInferenceSteps: 9
      defaultSize: "1024x1024"
```
