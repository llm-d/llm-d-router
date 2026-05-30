# llm-d Inference Scheduler Architecture

---

## Overview

**llm-d** is an extensible architecture designed to schedule inference requests efficiently across model-serving pods.
 A central component of this architecture is the **Inference Gateway**, which builds on the Kubernetes-native
 **Gateway API Inference Extension** (GIE) to enable scalable, flexible, and pluggable request scheduling.

The design enables:

- Support for **multiple base models** within a shared cluster (see [serving multiple inference pools](https://gateway-api-inference-extension.sigs.k8s.io/guides/serving-multiple-inference-pools-latest/))
- Efficient routing based on **KV cache locality**, **session affinity**, **load**, and
**model metadata**
- Disaggregated **Prefill/Decode (P/D)** execution
  - We have introduced experimental **Encode/Prefill/Decode (E/P/D and all its permutations)** execution. For a detailed explanation, see [Disaggregated Inference Serving](./disaggregation.md)
- Pluggable **filters**, **scorers**, and **scrapers** for extensible scheduling

---

## Core Goals

- Schedule inference requests to optimal pods based on:
  - Base model compatibility
  - KV cache reuse
  - Load balancing
- Support multi-model deployments on heterogeneous hardware
- Enable runtime extensibility with pluggable logic (filters, scorers, scrapers)
- Community-aligned implementation using GIE and Envoy + External Processing (EPP)

---

## Architecture Design

![Inference Gateway Architecture](./images/architecture.png)

The inference scheduler is built on top of:

- The [Envoy] gateway, as a programmable data plane.
- An [EPP] (Endpoint Picker), making scheduling decisions, as the control plane.
  The llm-d inference scheduler extends the EPP in [GIE] with state of the art
  scheduling algorithms.
- An optional [BBR] (Body Based Routing) component, to associate requests with
  their corresponding model before the EPP is consulted.

[Envoy]:https://www.envoyproxy.io/
[EPP]:https://gateway-api-inference-extension.sigs.k8s.io/#endpoint-picker
[BBR]:https://gateway-api-inference-extension.sigs.k8s.io/#concepts-and-definitions
[GIE]:https://github.com/kubernetes-sigs/gateway-api-inference-extension

---

### Pluggability

![Pluggability Architecture](./images/plugability.png)

Routing decisions are governed by dynamic components:

- **Filters**: Exclude pods based on static or dynamic criteria
- **Scorers**: Assign scores to candidate pods
- **Scrapers**: Collect pod metadata and metrics for scorers

These components are maintained in the `llm-d-inference-scheduler` repository and can evolve independently.
A [sample filter plugin guide](./create_new_filter.md) is provided to illustrate how one could extend the
 Inference Gateway functionality to address unique requirements.

---

## Filters, Scorers, and Scrapers

### Core Design Principles

- **Pluggability**: No core changes are needed to add new scorers or filters
- **Isolation**: Each component operates independently

---

### Routing Flow

1. **Filtering**
   - Pods in an `InferencePool` go through a sequential chain of filters
   - Pods may be excluded based on criteria like model compatibility, resource usage, or custom logic

2. **Scoring**
   - Filtered pods are scored using a weighted set of scorers
   - Scorers currently run sequentially (future: parallel execution)
   - Scorers access a shared datastore populated by scrapers

3. **Pod Selection**
   - The highest-scored pod is selected
   - If multiple pods share the same score, one is selected at random

---

### Lifecycle Hooks

- `Pre-call`
- `Scoring`
- `Post-choice`
- `After-response`

---

## Configuration

The set of lifecycle hooks (plugins) that are used by the inference scheduler is determined by how
 it is configured. The configuration is in the form of YAML text, which can either be in a file or
 specified in-line as a parameter. The configuration defines the set of plugins to be instantiated
 along with their parameters. Each plugin is also given a name, enabling the same plugin type to be
 instantiated multiple times, if needed. Also defined is a set of SchedulingProfiles, which determine
 the set of plugins to be used when scheduling a request. The set of plugins instantiated must also
 include a Profile Handler, which determines which SchedulingProfiles will be used for a particular
 request and how their results will be processed.

The configuration text has the following form:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- ....
- ....
schedulingProfiles:
- ....
- ....
```

The first two lines of the configuration are constant and must appear as is.

The plugins section defines the set of plugins that will be instantiated and their parameters.
 Each entry in this section has the following form:

```yaml
- name: aName
  type: a-type
  parameters:
    param1: val1
    param2: val2
```

The fields in a plugin entry are:
- **name** (optional): provides a name by which the plugin instance can be referenced. If this
field is omitted, the plugin's type will be used as its name.
- **type**: specifies the type of the plugin to be instantiated.
- **parameters** (optional): defines the set of parameters used to configure the plugin in question.
The actual set of parameters varies from plugin to plugin.

The schedulingProfiles section defines the set of scheduling profiles that can be used in scheduling
requests to pods. The number of scheduling profiles one defines, depends on the use case. For simple
serving of requests, one is enough. For disaggregated prefill, two profiles are required. Each entry
in this section has the following form:

```yaml
- name: aName
  plugins:
  - pluginRef: plugin1
  - pluginRef: plugin2
    weight: 50
```
The fields in a schedulingProfile entry are:
- **name**: specifies the scheduling profile's name.
- **plugins**: specifies the set of plugins to be used when this scheduling profile is chosen for a request.
  - **pluginRef**: reference to the name of the plugin instance to be used
  - **weight**: weight to be used if the referenced plugin is a scorer.

A complete configuration might look like this:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: precise-prefix-cache-scorer
  parameters:
    indexerConfig:
      tokenProcessorConfig:
        blockSize: 5
      kvBlockIndexConfig:
        maxPrefixBlocksToMatch: 256
- type: decode-filter
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: precise-prefix-cache-scorer
    weight: 50
```

If the configuration is in a file, the EPP command line argument `--configFile` should be used
 to specify the full path of the file in question. If the configuration is passed as in-line
 text the EPP command line argument `--configText` should be used.

---

### Plugin Configuration

This section describes how to setup the various plugins available with the llm-d-inference-scheduler

#### DisaggHeadersHandler

Sets headers for use in disaggregated prefill/decode and encode/prefill/decode

- **Type**: `disagg-headers-handler`
- **Parameters**:
  - `prefillProfile`: specifies the name of the profile used for the prefill scheduling. Only needed if the prefill profile is not named `prefill`.
  - `encodeProfile`: specifies the name of the profile used for the encode scheduling. Only needed if the encode profile is not named `encode`.

---

#### DisaggProfileHandler


Selects the profiles to use when running with disaggregation

- **Type**: `disagg-profile-handler`
- **Parameters**:
  - `profiles` (optional): names of scheduling profiles to use. Defaults match the profile names.
    - `decode`: name of the decode scheduling profile. Defaults to `decode`.
    - `prefill`: name of the prefill scheduling profile. Defaults to `prefill`.
    - `encode`: name of the encode scheduling profile. Defaults to `encode`.
  - `deciders` (optional): decider plugins that control whether each disaggregation stage runs.
    - `prefill`: name of the prefill decider plugin. When set, enables P/D disaggregation.
    - `encode`: name of the encode decider plugin. When set, enables E disaggregation.


> [!NOTE]
> When using this plugin with P/D disaggregation, you must also have a PrefixCachePlugin configured in the prefill and decode scheduling profiles.

**Examples**

Decode-only (no disaggregation):
```yaml
- type: disagg-profile-handler
```

P/D disaggregation:
```yaml
- type: disagg-profile-handler
  parameters:
    deciders:
      prefill: prefix-based-pd-decider
```

E/P/D disaggregation:
```yaml
- type: disagg-profile-handler
  parameters:
    deciders:
      prefill: prefix-based-pd-decider
      encode: always-disagg-multimodal-decider
```

---

#### Prefix Based Decider Plugin

Type: `prefix-based-pd-decider`

**Parameters**
- `nonCachedTokens`: length, in token, of the uncached part of the user input above which disaggregated PD is triggered.

> [!NOTE]
> `prepareDataPlugins` feature gate should be enabled

**Example**
```yaml
kind: EndpointPickerConfig
featureGates:
- prepareDataPlugins
plugins:
- type: prefix-based-pd-decider
  parameters:
    nonCachedTokens: 4
- type: disagg-profile-handler
  parameters:
    deciders:
      prefill: prefix-based-pd-decider
```

---

#### ByLabelSelector

Filters out pods using a standard Kubernetes label selector.

> [!NOTE]
>  Only the matching labels feature of Kubernetes label selectors is supported.

- **Type**: `by-label-selector`
- **Parameters**: A standard Kubernetes label selector.
  - `matchLabels`: map of `{key,value}` pairs. If more than one pair are in the map, all of the keys are checked and the results are combined with AND logic.

Example configuration with the above parameters set:

```yaml
plugins:
  - type: by-label-selector
    parameters:
      matchLabels:
        inference-role: decode
        hardware-type: H100
```

In this example:
- Only pods that have **both** labels  
  `inference-role=decode` **and** `hardware-type=H100`  
  will be selected.
- Pods missing either label, or having a different value (e.g., `inference-role=prefill`), are **filtered out**.
- The matching logic follows standard Kubernetes label selector semantics: all key-value pairs in `matchLabels` must match (**AND** logic).

---

#### ByLabel

Filters out pods that do not have a specific label with one of the allowed values. Pods missing the label are either filtered out or retained based on the `allowsNoLabel` setting.

- **Type**: `by-label`  
- **Parameters**:
  - `label` (string, required): The name of the Kubernetes label to inspect on each pod.
  - `validValues` (list of strings, required unless `allowsNoLabel=true`): A list of acceptable label values. A pod is kept if its label value matches any entry in this list.
  - `allowsNoLabel` (boolean, optional, default: `false`): If `true`, pods that **do not have the specified label at all** will be **included** in the candidate set. If `false` (default), such pods are filtered out.

Example configuration with the above parameters set:

```yaml
plugins:
  - type: by-label
    parameters:
      label: "inference-role"
      validValues: ["decode", "prefill-decode"]
      allowsNoLabel: false
```

In this example:
- Only pods labeled for decoding (`inference-role=decode`) or supporting both stages (`inference-role=prefill-decode`) are selected.
- Pods missing the `inference-role` label are not considered for decode scheduling.

---

#### DecodeFilter

Filters out pods that are not marked either as decode or both prefill and decode. The filter looks for
 the label `llm-d.ai/role`, with a value of either `decode`, `prefill-decode` or `encode-prefill-decode`. In addition pods that are missing the label will not be filtered out.

- **Type**: `decode-filter`
- **Parameters**: None

---

#### PrefillFilter

Filters out pods that are not marked as prefill. The filter looks for the label `llm-d.ai/role`, with a value of `prefill`, `encode-prefill`, `prefill-decode` or `encode-prefill-decode`. In addition pods that are missing the label will not be filtered out.

- **Type**: `prefill-filter`
- **Parameters**: None

---

#### EncodeFilter

Filters out pods that are not marked as encode. The filter looks for the label `llm-d.ai/role`, with a value of `encode`, `encode-prefill` or `encode-prefill-decode`. In addition pods that are missing the label will not be filtered out.

- **Type**: `encode-filter`
- **Parameters**: None

---

#### PrecisePrefixCacheScorer

The `precise-prefix-cache-scorer` scores a request based on KV-cache localities.
Similarly to the IGW `prefix-cache-scorer`, it provides a score based on the number of
 matching KV-cache blocks between the request's prompt and the KV-cache contents of each pod.
 However, unlike the IGW `prefix-cache-scorer`, which relies on estimations based on scheduling history,
 the `precise-prefix-cache-scorer` tracks the real-time KV-cache states across the vLLM instances to
 provide more accurate scoring.

When enabled, the scorer will use the `llm-d-kv-cache` to track the KV-cache states
 across the vLLM instances. It will use the `kvcache.Indexer` to score the pods based on the
 number of matching blocks in the KV-cache. It will also use the `kvevents.Pool` to subscribe
 to the KV-Events emitted by the vLLM instances and update the KV-cache states in near-real-time.

Configuration:

- **Type**: `precise-prefix-cache-scorer`
- **Parameters**:
  - `tokenProcessorConfig`: Configuration for the `kvblock.TokenProcessor`.
  - `indexerConfig`: Configuration for the `kvcache.Indexer`.
  - `kvEventsConfig`: Configuration for the `kvevents.Pool`.

See list of parameters at [llm-d-kv-cache/docs/configuration.md](https://github.com/llm-d/llm-d-kv-cache/blob/fa85b60207ba0a09daf23071e10ccb62d7977b40/docs/configuration.md).

> [!NOTE] 
> In most cases you will only need to set:
> - Model name in the `tokenizersPoolConfig` to match the model used in the vLLM deployment.
> - HuggingFace token for the `tokenizersPoolConfig` or the `tokenizersCacheDir` to a mounted directory containing the tokenizers.
>  - For the HuggingFace token, the inference-scheduler also accepts the environment variable `HF_TOKEN` - this is the practical option for security. 
> - **IMPORTANT**: Token processor's block-size and hash-seed to match those used in the vLLM deployment.
> - `KVBlockIndex` metrics to true if you wish to enable metrics for the KV-Block Index (admissions, evictions, lookups and hits).

Example configuration with the above parameters set:

```yaml
plugins:
  - type: precise-prefix-cache-scorer
    parameters:
      tokenProcessorConfig:
        blockSize: 64                    # must match vLLM block size
        hashSeed: "12345"                # must match vLLM PYTHONHASHSEED env var
      indexerConfig:
        kvBlockIndexConfig:
          enableMetrics: true    
        tokenizersPoolConfig:
          modelName: hf-repo/model-name
          hf:
            huggingFaceToken: your_hf_token_here    # automatically set by `HF_TOKEN` environment variable
```

Example configuration for automatic pod discovery in active-active multi-replica scheduler deployments:
```yaml
  - type: precise-prefix-cache-scorer
    parameters:
      tokenProcessorConfig:
        blockSize: 64
        hashSeed: "42"
      indexerConfig:
        tokenizersPoolConfig:
          modelName: "Qwen/Qwen3-32B"
          hf:
            tokenizersCacheDir: "/tmp/tokenizers"
      kvEventsConfig:
        topicFilter: "kv@"
        concurrency: 4 
        discoverPods: true      # enables automatic pod discovery for active-active HA
        podDiscoveryConfig:
          socketPort: 5556
```

Where the vLLM engines are configured to emit KV-Events on port `5556` as follows:
```yaml
  --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:5556\",\"topic\":\"kv@${POD_IP}@Qwen/Qwen3-32B\"}"
```

Example configuration with all parameters set:

```yaml
plugins:
  - type: precise-prefix-cache-scorer
    parameters:
        tokenProcessorConfig:
          blockSize: 16
          hashSeed: "12345"
        kvEventsConfig:
          topicFilter: "kv@"
          concurrency: 4
          discoverPods: true    # enables automatic pod discovery for active-active HA
          podDiscoveryConfig:
            socketPort: 5556
        indexerConfig:
          prefixStoreConfig:
            cacheSize: 500000
            blockSize: 256
          kvBlockIndexConfig:
            inMemoryConfig:
              size: 100000000
              podCacheSize: 10
            enableMetrics: true
          tokenizersPoolConfig:
            modelName: hf-repo/model-name
            workersCount: 8
            hf:
              huggingFaceToken: your_hf_token_here    # automatically set by `HF_TOKEN` environment variable
              tokenizersCacheDir: /tmp/tokenizers
```

---

#### LoadAwareScorer

Scores pods based on their load, based on the number of requests concurrently being processed.
A threshold is provided which is used to determine what is considered an overloaded pod.

Scores are given to the pods in the range of 0-1. Currently the metrics contain the number of
requests waiting in the queue, there is no information about number of requests that can be
processed in the given pod immediately.

Pods with an empty waiting requests queue are scored with 0.5.

Pods with requests in the queue will get score between 0.5 and 0.

- **Type**: `load-aware-scorer`
- **Parameters**:
  - `threshold`: specifies the threshold at which a pod is considered overloaded.

---

#### ActiveRequestScorer

Scores pods based on the number of active requests being served per pod. Each request is tracked 
individually with its own TTL to ensure accurate timeout handling. Pods with fewer active 
requests receive higher scores.

Scores are normalized to a range of 0-1, where pods with fewer active requests get higher scores.

- **Type**: `active-request-scorer`
- **Parameters**:
  - `requestTimeout`: specifies the timeout for requests in seconds. Once a request is "in-flight" 
    for this duration, it is considered timed out and automatically removed.

---

#### SessionAffinity

Scores the candidate pods by giving a higher score to the pods that were previously used for the same session.

- **Type**: `session-affinity-scorer`
- **Parameters**:
  - `maxAge`: the cookie expiration time, in seconds. The default value is 0, which denotes a session cookie (expires when the browser is closed).
  - `sessionIDProducerName`: name of the `session-id-producer` instance to read session data from. Must match the `name` of a configured `session-id-producer` plugin. Defaults to the built-in default producer name when omitted.

Example configuration with the above parameter set:

```yaml
plugins:
  - type: session-affinity-scorer
    parameters:
      maxAge: 3600  # Cookie expires after 1 hour
```
---

#### NoHitLRUScorer

Scores pods based on least recently used (LRU) ordering for cold requests (requests with no KV cache hits).
This helps evenly distribute cache growth across pods, since cold requests result in new KV blocks being created.

The scorer integrates with a prefix cache plugin to determine if a request has cache hits:
- For cold requests (no cache hits): Ranks pods by LRU order, with never-used or least recently used pods
  receiving higher scores (up to 1.0) and most recently used pods receiving lower scores (approaching 0.0)
- For warm requests (cache hits): Returns neutral scores (0.5) for all pods to avoid interfering with
  cache locality optimization

The LRU tracking is specific to cold requests only - pods are added to the LRU cache when they serve
a cold request, not when they serve requests with cache hits.

- **Type**: `no-hit-lru-scorer`
- **Parameters**:
  - `prefixPluginName` (optional): The name of the prefix cache plugin to read state from. Defaults to `prefix-cache-scorer`.
  - `lruSize` (optional): The maximum number of pods to track in the LRU cache. Defaults to 1024.

Example configuration:

```yaml
plugins:
  - type: precise-prefix-cache-scorer
    parameters:
      indexerConfig:
        tokenProcessorConfig:
          blockSize: 5
        kvBlockIndexConfig:
          maxPrefixBlocksToMatch: 256
  - type: no-hit-lru-scorer
    parameters:
      lruSize: 2048
  - type: decode-filter
  - type: max-score-picker
  - type: single-profile-handler
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: decode-filter
      - pluginRef: max-score-picker
      - pluginRef: precise-prefix-cache-scorer
        weight: 2
      - pluginRef: no-hit-lru-scorer
        weight: 1
```

> [!NOTE]
>  This scorer is designed to work alongside a prefix cache scorer (such as `prefix-cache-scorer` or
> `precise-prefix-cache-scorer`). If no prefix cache state is available, all requests are treated as cold.
> When integrating with a prefix-cache scorer, the prefix-cache scorer should be defined first in the scheduling 
> profile.

---

#### ContextLengthAware

A **Scorer** plugin that routes inference requests based on context length (token count), with optional
**filtering** gated behind `enableFiltering`. Scoring is always applied; filtering is off by default.
This enables optimized resource allocation by directing requests to pods configured for specific
context length ranges.

**Use Cases:**
- Route short prompts to pods with smaller GPU memory
- Direct long-context requests to specialized high-memory pods
- Optimize performance by matching workload characteristics to hardware capabilities
- Support heterogeneous deployments with different GPU configurations

The plugin scores all pods based on how well their ranges match the request:
- **In-range match (0.3–1.0]:** Higher scores for tighter/more specific ranges (specialized pods), lower scores for very wide ranges (generalist pods). In-range scores are always strictly above 0.3, guaranteeing they beat any out-of-range fallback.
- **Out-of-range fallback [0.0–0.3):** When no range matches, pods are ranked by proximity to the request. For example, a 9000-token request prefers a pod with `max=8192` over one with `max=2048`.
- **Neutral score (0.5):** Pods without the context length label.

When `enableFiltering` is set to true, the plugin also filters out pods whose range does not contain the request's context length.

**Configuration:**

- **Type**: `context-length-aware`
- **Parameters**:
  - `label` (optional): Pod label name containing context length range(s).
    Default: `llm-d.ai/context-length-range`
  - `enableFiltering` (optional): If true, the plugin operates as a filter, excluding non-matching pods.
    Default: false

**Token Counting:**

This plugin reads tokenized prompt data from CycleState, as written by the `tokenizer` plugin.
When the tokenizer plugin is configured in the same scheduling profile, the context-length-aware plugin
uses the exact token count for routing decisions. This avoids double-tokenization with other plugins
that also consume tokens (e.g., `precise-prefix-cache-scorer`).

When the tokenizer scorer is not configured, the plugin falls back to character-based estimation
(characters × 0.25).

> [!NOTE]
> The `tokenizer` plugin must appear **before** `context-length-aware` in the scheduling profile
> so that CycleState is populated before scoring runs.

**Label Format:**

Pods should be labeled with context length ranges using the format `"min-max"`, where _min_ and _max_ are both positive integers:

```yaml
llm-d.ai/context-length-range: "0-2048"
```

**Example Configuration - Scorer:**

```yaml
plugins:
  - type: tokenizer
    parameters:
      modelName: meta-llama/Llama-3.1-8B-Instruct
      udsTokenizerConfig:
        socketFile: /tmp/tokenizer/tokenizer-uds.socket
  - type: context-length-aware
    parameters:
      label: llm-d.ai/context-length-range
  - type: load-aware-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: tokenizer
        weight: 1
      - pluginRef: context-length-aware
        weight: 3
      - pluginRef: load-aware-scorer
        weight: 1
      - pluginRef: max-score-picker
```

**Example Configuration - Scorer with Filtering Enabled:**

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

---

### Sample Disaggregated Prefill/Decode Configuration

The following is an example of what a configuration for disaggregated Prefill/Decode might look like:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: disagg-headers-handler
- type: prefix-cache-scorer
  parameters:
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 31250
- type: prefill-filter
- type: decode-filter
- type: max-score-picker
- type: prefix-based-pd-decider
    parameters:
      nonCachedTokens: 8
- type: disagg-profile-handler
  parameters:
    deciders:
      prefill: prefix-based-pd-decider
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 50
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 50
```

Several things should be noted:
1. The `PrefillHeader`, `DisaggProfileHandler`, `DecodeFilter`, `PrefillFilter` and the `PrefixCachePlugin`
 plugins must be in the list of plugins instantiated.
2. There must be two scheduler profiles defined.
3. The scheduler profile for prefill, must include the `PrefillFilter`
4. The scheduler profile for decode, must include the `DecodeFilter`

### Speculative Indexing

Speculative indexing closes the blind spot between a routing decision and KV event arrival by immediately writing a predicted cache entry to the prefix-cache index. This lets the next request with the same prefix hit the cache without waiting for engine confirmation. See [#538](https://github.com/llm-d/llm-d-router/issues/538) for background.

Enable via `precise-prefix-cache-scorer` parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speculativeIndexing` | bool | `false` | Enable speculative index entries on routing decisions. |
| `speculativeTTL` | duration string | `"2s"` | TTL for speculative entries. Accepts Go duration strings (e.g. `"2s"`, `"500ms"`). |

Requires the `prepareDataPlugins` feature gate and KV events from vLLM engines.

---

## Metric Scraping

- Scrapers collect metrics (e.g., memory usage, active adapters)
- Data is injected into the shared datastore for scorers
- Scoring can rely on numerical metrics or metadata (model ID, adapter tags)

---

## Disaggregated Encode/Prefill/Decode (E/P/D)

When enabled, the router:

- Selects one pod for **Prefill** (prompt processing)
- Selects another pod for **Decode** (token generation)

> [!NOTE] 
> Encode disaggregation is an experimental feature. When enabled, the router 
> identifies all pods capable of encoding, and the vLLM sidecar distributes multimedia 
> requests to randomly selected pods from that subset. More sophisticated selection 
> strategies are planned for future versions.

The **vLLM sidecar** handles orchestration between Encode, Prefill and Decode stages. It allows:

- Queuing
- Local memory management
- Experimental protocol compatibility

> [!NOTE]
> The detailed E/P/D design is available in this document:
> [Disaggregated Inference Serving in llm-d](./disaggregation.md)

---

## InferencePool & InferenceModel Design

### Current Assumptions

- Single `InferencePool` and single `EPP` due to Envoy limitations
- Model-based filtering can be handled within EPP
- Currently only one base model **per `InferencePool`** is supported.
  Multiple models are supported via multiple `InferencePools`.

> [!NOTE]
> The `InferenceModel` CRD is in the process of being significantly changed in IGW.
> Once finalized, these changes would be reflected in llm-d as well.

---

## References

- [GIE Spec](https://gateway-api-inference-extension.sigs.k8s.io/)
- [Envoy External Processing](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter)
