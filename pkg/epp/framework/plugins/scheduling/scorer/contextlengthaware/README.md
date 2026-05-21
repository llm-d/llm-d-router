# Context Length Aware Scorer

**Type:** `context-length-aware`

Routes inference requests based on context length (token count), with optional filtering. Scoring is always applied; filtering is off by default. Enables optimized resource allocation by directing requests to pods configured for specific context length ranges.

**Use Cases:**
- Route short prompts to pods with smaller GPU memory.
- Direct long-context requests to specialized high-memory pods.
- Optimize performance by matching workload characteristics to hardware capabilities.
- Support heterogeneous deployments with different GPU configurations.

Each pod declares its supported range via the `label` parameter (default: `"llm-d.ai/context-length-range"`), formatted as `"min-max"` (e.g. `"0-2048"`).

Scoring rules:
- **In-range match (0.3–1.0]:** Higher scores for tighter/more specific ranges; lower scores for very wide generalist ranges. Always strictly above `0.3`.
- **Out-of-range fallback [0.0–0.3):** Pods are ranked by proximity to the request (e.g. a 9000-token request prefers a pod with `max=8192` over `max=2048`).
- **Neutral score (0.5):** Pods without the context length label.

When `enableFiltering` is `true`, pods whose range does not contain the request's context length are also filtered out.

#### Pod Label Format

```yaml
metadata:
  labels:
    llm-d.ai/context-length-range: "0-2048"   # min-max token count supported by this pod
```

**Parameters:**
- `label` (string, optional, default: `"llm-d.ai/context-length-range"`): Pod label key carrying the `"min-max"` range.
- `enableFiltering` (bool, optional, default: `false`): Also act as a filter, removing out-of-range pods before scoring.

**Configuration Example:**
```yaml
plugins:
  - type: context-length-aware
    name: context-router
    parameters:
      label: "llm-d.ai/context-length-range"
      enableFiltering: false
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: context-router
        weight: 8
```

#### Token Counting

Reads tokenized prompt data from `request.Body.TokenizedPrompt` as written by the `token-producer` DataProducer plugin. When a token-producer is configured, the exact token count is used. Otherwise falls back to character-based estimation (characters × 0.25).

**Example — Scorer with token-producer:**
```yaml
plugins:
  - type: token-producer
    parameters:
      modelName: meta-llama/Llama-3.1-8B-Instruct
      vllm:
        http: http://localhost:8000
  - type: context-length-aware
    parameters:
      label: llm-d.ai/context-length-range
  - type: load-aware-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: context-length-aware
        weight: 3
      - pluginRef: load-aware-scorer
        weight: 1
      - pluginRef: max-score-picker
```

**Example — Scorer with filtering enabled:**
```yaml
plugins:
  - type: context-length-aware
    parameters:
      enableFiltering: true
      label: llm-d.ai/context-length-range
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: context-length-aware
      - pluginRef: max-score-picker
```

**Example Pod Labels:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-short-context
  labels:
    llm-d.ai/context-length-range: "0-2048"
---
apiVersion: v1
kind: Pod
metadata:
  name: vllm-long-context
  labels:
    llm-d.ai/context-length-range: "2048-8192"
```
