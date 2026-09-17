# Context Length Aware Scorer

**Type:** `context-length-aware`

Routes inference requests based on a token count, with optional filtering.
Scoring is always applied; filtering is off by default. The routing length is
the total prompt length unless `reusableTokensProducerName` enables P2P cache
subtraction.

**Use Cases:**
- Route short prompts to pods with smaller GPU memory.
- Direct long-context requests to specialized high-memory pods.
- Optimize performance by matching workload characteristics to hardware capabilities.
- Support heterogeneous deployments with different GPU configurations.

Each pod declares its range via the `label` parameter (default:
`"llm-d.ai/context-length-range"`), formatted as `"min-max"` (e.g.
`"0-2048"`).

Scoring rules:
- **In-range match (0.3–1.0]:** Higher scores for tighter/more specific ranges; lower scores for very wide generalist ranges. Always strictly above `0.3`.
- **Out-of-range fallback [0.0–0.3):** Pods are ranked by proximity to the request (e.g. a 9000-token request prefers a pod with `max=8192` over `max=2048`).
- **Neutral score (0.5):** Pods without the configured range label.

When `enableFiltering` is `true`, pods whose range does not contain the
request's routing length are also filtered out.

#### Pod Label Format

```yaml
metadata:
  labels:
    llm-d.ai/context-length-range: "0-2048"   # min-max token count supported by this pod
```

**Parameters:**
- `label` (string, optional, default: `"llm-d.ai/context-length-range"`): Pod label key carrying the `"min-max"` range.
- `enableFiltering` (bool, optional, default: `false`): Also act as a filter, removing out-of-range pods before scoring.
- `reusableTokensProducerName` (string, optional): Name of the
  `p2p-source-producer` instance whose `ReusablePrefixTokens` request attribute
  is subtracted from the total prompt length. Requires a `label` other than
  `llm-d.ai/context-length-range`. Empty disables cache subtraction.

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

Reads total prompt tokens from `request.Body.TokenizedRequest.TokenCount()`.
A `token-producer` populates this data and is auto-created with the
tokenizer-free `estimate` backend when none is configured.

When `reusableTokensProducerName` is configured, the plugin requires the
name-bound attribute from that `p2p-source-producer`. Missing request data at
runtime leaves the total prompt length unchanged. This includes requests for
which the producer finds no pullable source. The default configuration does
not consume this attribute and always uses total prompt length.

#### P2P Cache-Aware Prefill Work

The named [P2P source producer](../../../requestcontrol/dataproducer/p2psource/README.md)
publishes a request-wide reusable-prefix floor `G` from confirmed CPU-tier
cache data. The context-length-aware plugin uses:

```text
routingLength = 0                              when totalPromptTokens is 0
routingLength = max(1, totalPromptTokens - G) otherwise
```

Missing floor data leaves a known total unchanged.

Let `S` be the confirmed prefix on the sampled source, `D` be
`minCachedTokenDelta`, and `newTokens` be the prompt tokens after that prefix.
For a positive floor and a follow-up with positive `newTokens`:

```text
totalPromptTokens = S + newTokens
G = S - D + 1
routingLength = newTokens + D - 1
```

The delta is part of the routed work. If P-short is intended to accept at most
`N` new tokens, its work-range ceiling must be at least `N + D - 1`; a ceiling
of `N` is sufficient only when `D` is `1`. Start the next work range above the
adjusted ceiling.

The routing length is a conservative upper bound under the producer's cache
snapshot. Every destination, including P-short, must still support the
request's full logical sequence because cache state can change and a transfer
can fail.

Use cache subtraction only in the prefill scheduling profile. Use a separate
work-range label such as `llm-d.ai/prefill-work-range`; the default
`llm-d.ai/context-length-range` describes the logical context length supported
by a pod. Label every candidate in the prefill profile because unlabeled pods
are retained by filtering as general-purpose candidates.

Configure `p2p-cache-source` as shown in the
[producer configuration](../../../requestcontrol/dataproducer/p2psource/README.md#configuration).
Its `prefillProfileName` must match both the profile selected by
`disagg-profile-handler` and the profile that references the work classifier.
The profile's candidate filters and picker are omitted here.

```yaml
plugins:
  - type: context-length-aware
    name: prefill-work-router
    parameters:
      label: llm-d.ai/prefill-work-range
      enableFiltering: true
      reusableTokensProducerName: p2p-cache-source
schedulingProfiles:
  - name: prefill
    plugins:
      - pluginRef: prefill-work-router
```

The producer and serving pods must satisfy the
[deployment requirements](../../../requestcontrol/dataproducer/p2psource/README.md#deployment-requirements).
With `D = 1`, a short-work class for up to 8192 new tokens can use:

```yaml
metadata:
  labels:
    llm-d.ai/prefill-work-range: "0-8192"
    llm-d.ai/context-length-range: "0-131072"
```

P-long candidates can use `llm-d.ai/prefill-work-range: "8193-131072"` while
retaining the same logical context range. For larger `D`, raise the short
ceiling and the long lower bound by `D - 1`.

**Example — Scorer with token-producer:**
```yaml
plugins:
  - type: token-producer
    parameters:
      modelName: meta-llama/Llama-3.1-8B-Instruct
      vllm:
        url: http://localhost:8000
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
