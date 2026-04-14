# llm-d Inference Scheduler Architecture

## Table of Contents

- [Overview](#overview)
- [Core Goals](#core-goals)
- [Filters, Scorers, and Scrapers](#filters-scorers-and-scrapers)
  - [Core Design Principles](#core-design-principles)
  - [Routing Flow](#routing-flow)
- [Configuration](#configuration)
  - [`Plugins` Configuration](#plugins-configuration)
  - [`SchedulingProfiles` Configuration](#schedulingprofiles-configuration)
  - [Available plugins](#available-plugins)
- [Metric Scraping](#metric-scraping)
- [Disaggregated Encode/Prefill/Decode (E/P/D)](#disaggregated-encodeprefilldecodesepd-epd)
- [InferencePool & InferenceModel Design](#inferencepool--inferencemodel-design)
  - [Current Assumptions](#current-assumptions)
- [References](#references)

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

## Configuration

The inference scheduler relies on a YAML-based configuration—provided either as a file or an in-line parameter—to determine which lifecycle hooks (plugins) are active.

Specifically, this configuration establishes the following components:

- `Plugins`: The specific plugins to instantiate, along with their parameters. Because each instantiated plugin is assigned a unique name, you can configure the same plugin type multiple times if necessary.

- `SchedulingProfiles`: A collection of profiles that dictate the exact set of plugins invoked when scheduling a given request.

The configuration text has the following form:

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- ....
- ....
schedulingProfiles:
- ....
- ....
```

The first two lines of the configuration are constant and must appear as is.

### `Plugins` Configuration

The `plugins` section in the configuration defines the set of plugins that will be instantiated and their parameters. Each entry in this section has the following form:

```yaml
- name: aName
  type: a-type
  parameters:
    param1: val1
    param2: val2
```

#### `Plugin` Fields:

The fields in a plugin entry are:

- **name** (optional): provides a name by which the plugin instance can be referenced. If this field is omitted, the plugin's type will be used as its name.
- **type**: specifies the type of the plugin to be instantiated.
- **parameters** (optional): defines the set of parameters used to configure the plugin in question. The actual set of parameters varies from plugin to plugin.

### `SchedulingProfiles` Configuration

The `schedulingProfiles` section defines the set of scheduling profiles that can be used in scheduling
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

#### `SchedulingProfile` Fields

The fields in a schedulingProfile entry are:

- **name**: specifies the scheduling profile's name.
- **plugins**: specifies the set of plugins to be used when this scheduling profile is chosen for a request.
- **pluginRef**: reference to the name of the plugin instance to be used
- **weight**: weight to be used if the referenced plugin is a scorer.

A complete configuration might look like this:

```yaml
apiVersion: llm-d.ai/v1alpha1
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


### Available plugins

To learn more about the available plugins, check the plugins [README.md](../pkg/epp/framework/plugins/README.md) file.

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

## Chunked Decode (Experimental)

Chunked decode is an experimental feature of the pd-sidecar that splits the decode stage into a
sequence of shorter decode calls, each capped at a configurable token budget.
It applies at the decode stage regardless of whether P/D disaggregation is in use.
After each chunk the generated text is appended to the conversation context so the next
chunk continues seamlessly from where the previous one left off.

### Why to use it

- Improve average Time-To-First-Token among all requests
- Prevent head-of-line blocking by long requests in run-to-completion
- Get more predictable execution time

### How it works

1. The sidecar receives a `/v1/chat/completions` or `/v1/completions` request at the decode stage.
2. Each chunk is dispatched as a separate request to the local decoder with `max_tokens` capped
   at `decode-chunk-size`.
3. From the second chunk onward, `continue_final_message=true` and `add_generation_prompt=false` are
   set so the model continues the existing assistant turn rather than starting a new one. The
   generated text from the previous chunk is also appended to the request context.
4. Generation stops when the model returns a terminal `finish_reason` (anything other than `length`),
   or when the original token budget is exhausted.
5. For **non-streaming** requests, all chunk outputs are concatenated and returned as a single
   response. The `usage` field reports the original `prompt_tokens` (from the first chunk) and the
   total `completion_tokens` across all chunks.
6. For **streaming** requests, each chunk's tokens are re-emitted as SSE delta events in real time,
   and a `[DONE]` sentinel closes the stream once all chunks are complete.

### Configuration

Enable chunked decode via the pd-sidecar flag:

| Flag | Default | Description |
|---|---|---|
| `--decode-chunk-size` | `0` (disabled) | Token budget per chunk. Set to a positive integer to enable chunked decode. For best performance use a multiple of the KV cache block size. |

> [!NOTE]
> If the request's `max_tokens` / `max_completion_tokens` is less than or equal to `--decode-chunk-size`,
> the sidecar falls back to a single regular decode call without chunking.

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

- [GIE Spec](../README.md#relation-to-gie-igw)
- [Envoy External Processing](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter)
