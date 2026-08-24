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
  is subtracted from the total prompt length. Empty disables cache subtraction.

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
publishes a request-wide reusable-prefix floor `G`. The context-length-aware
plugin uses:

```text
routingLength = max(0, totalPromptTokens - G)
```

The routing length is a conservative upper bound on the prefill work under the
router's cache snapshot: a destination either has at least `G` tokens locally
or qualifies for the P2P pull. Cache state can change and transfers can fail,
so every destination, including P-short, must support the request's full
logical sequence.

Use cache subtraction only in the prefill scheduling profile. Use a separate
work-range label such as `llm-d.ai/prefill-work-range`; the default
`llm-d.ai/context-length-range` describes the logical context length supported
by a pod. Label every candidate in the prefill profile because unlabeled pods
are retained by filtering as general-purpose candidates.

The following excerpt configures a precise cache producer, P2P source
selection, and a prefill-only work classifier in a disaggregated
configuration. The `p2p-cache-source` name binds the producer and consumer.
The `prefill` name must also be the prefill profile selected by
`disagg-profile-handler`; that handler and the profile's existing candidate
filters and picker are omitted here.

```yaml
plugins:
  - type: precise-prefix-cache-producer
    name: precise-cache
    parameters:
      tokenProcessorConfig:
        blockSizeTokens: 64
      speculativeIndexing: false
  - type: p2p-source-producer
    name: p2p-cache-source
    parameters:
      prefixMatchInfoProducerName: precise-cache
      minCachedTokenDelta: 1
      prefillProfileName: prefill
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

The cache producer and serving pods must satisfy the
[`p2p-source-producer` deployment requirements](../../../requestcontrol/dataproducer/p2psource/README.md#deployment-requirements).
A short-work pod can use labels such as:

```yaml
metadata:
  labels:
    llm-d.ai/prefill-work-range: "0-8192"
    llm-d.ai/context-length-range: "0-131072"
```

The P-long candidates can use `llm-d.ai/prefill-work-range: "8193-131072"`
while retaining the same logical context range.

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
