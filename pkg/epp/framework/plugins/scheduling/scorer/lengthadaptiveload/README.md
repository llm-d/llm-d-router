# Length-Adaptive Load Scorer Plugin

**Type:** `length-adaptive-load-scorer`

This plugin scores candidate endpoints by dynamically blending request-concurrency headroom and in-flight token-load headroom based on the incoming request's estimated token length. It is adapted from the Decode Load Balancer design in DeepSeek's [`DeepInfer`](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md).

## What it does

For each scheduling cycle, the plugin estimates the incoming request's prompt token count $L_r$ from `request.Body.TokenizedRequest.TokenCount()` (with a fallback to `request.RequestSizeBytes / 4`) and computes a per-request weight $\alpha(L_r) \in [0.0, 1.0]$:

$$
\alpha(L_r) = \text{clip}\!\left(\frac{L_r - \text{shortThresholdTokens}}{\text{longThresholdTokens} - \text{shortThresholdTokens}},\, 0.0,\, 1.0\right)
$$

For each candidate endpoint $w$ with in-flight requests $N_w$ and in-flight tokens $K_w$ read from `InFlightLoad`:

$$
S_{\text{queue}}(w) = 1.0 - \min\!\left(1.0,\, \frac{N_w}{\max(\text{maxConcurrency},\, \max_u N_u)}\right)
$$

$$
S_{\text{token}}(w) = 1.0 - \min\!\left(1.0,\, \frac{K_w}{\max(\text{maxTokenCapacity},\, \max_u K_u)}\right)
$$

$$
S_{\text{raw}}(w, r) = \bigl(1.0 - \alpha(L_r)\bigr) \cdot S_{\text{queue}}(w) + \alpha(L_r) \cdot S_{\text{token}}(w)
$$

- **Short requests ($L_r \le \text{shortThresholdTokens} \implies \alpha = 0.0$):** Scored by $S_{\text{queue}}(w)$, routing to endpoints with the fewest active requests to fill idle decode batch slots.
- **Long requests ($L_r \ge \text{longThresholdTokens} \implies \alpha = 1.0$):** Scored by $S_{\text{token}}(w)$, routing to endpoints with the fewest active KV cache tokens.
- **Medium requests ($\text{shortThresholdTokens} < L_r < \text{longThresholdTokens}$):** Linearly interpolated between $S_{\text{queue}}(w)$ and $S_{\text{token}}(w)$.
- **Capacity Ceiling Guardrail:** Endpoints where $(N_w + 1) > \text{maxConcurrency}$ or $(K_w + L_r) > \text{maxTokenCapacity}$ receive a score of `0.0`. If all candidate endpoints exceed the guardrail, the scorer falls back to returning $S_{\text{raw}}(w, r)$ so the least-loaded endpoint is selected.

## Scheduling intent

The scorer returns category `Distribution`, balancing both batch concurrency and KV cache token capacity across decode endpoints.

## Inputs consumed

The plugin consumes:
- `tokenproducer.TokenizedPromptDataKey` (`fwksched.TokenizedRequest`)
- `attrconcurrency.InFlightLoadDataKey` (`*attrconcurrency.InFlightLoad`)

## Configuration

The scorer supports the following runtime parameters:

- `shortThresholdTokens` (integer, default: `512`): Prompt token count at or below which the scorer routes purely by active request count ($\alpha = 0.0$).
- `longThresholdTokens` (integer, default: `16384`): Prompt token count at or above which the scorer routes purely by in-flight token load ($\alpha = 1.0$).
- `maxConcurrency` (integer, default: `256`): Default maximum concurrent requests per endpoint (e.g., vLLM `--max-num-seqs`).
- `maxTokenCapacity` (integer, default: `150000`): Default maximum KV cache token capacity per endpoint.
- `maxConcurrencyLabel` (string, default: `"llm-d.ai/max-num-seqs"`): Optional pod label key used to override `maxConcurrency` per endpoint in heterogeneous accelerator pools.
- `maxTokenCapacityLabel` (string, default: `"llm-d.ai/max-kv-tokens"`): Optional pod label key used to override `maxTokenCapacity` per endpoint in heterogeneous accelerator pools.
- `inFlightLoadProducerName` (string, optional): Name of a specific `inflight-load-producer` instance to read from.

**Configuration Example (P/D Disaggregation):**
```yaml
plugins:
  - type: token-producer
  - type: inflight-load-producer
  - type: length-adaptive-load-scorer
    parameters:
      shortThresholdTokens: 512
      longThresholdTokens: 16384
      maxConcurrency: 256
      maxTokenCapacity: 150000
schedulingProfiles:
  - name: decode
    plugins:
      - pluginRef: decode-filter
      - pluginRef: length-adaptive-load-scorer
      - pluginRef: max-score-picker
```
