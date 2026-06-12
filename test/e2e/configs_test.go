package e2e

// Simple EPP configuration for running without P/D
const simpleConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
  parameters:
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 256
- type: prefix-cache-scorer
- type: decode-filter
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
`

// EPP configuration for running with P/D
// Uses deprecated pd-profile-handler
const deprecatedPdConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefill-header-handler
- type: approx-prefix-cache-producer
  parameters:
    blockSizeTokens: 16
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 256
- type: prefix-cache-scorer
- type: prefill-filter
- type: decode-filter
- type: max-score-picker
- type: prefix-based-pd-decider
  parameters:
    nonCachedTokens: 16
- type: pd-profile-handler
  parameters:
    deciderPluginName: prefix-based-pd-decider
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
`

// epdEncodeDecodeConfig configures E/PD (encode + P/D) using disagg-profile-handler.
// The encode stage is triggered only for multimodal requests (image_url / video_url / input_audio).
const epdEncodeDecodeConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: encode-filter
- type: decode-filter
- type: max-score-picker
- type: always-disagg-multimodal-decider
- type: disagg-profile-handler
  parameters:
    deciders:
      encode: always-disagg-multimodal-decider
schedulingProfiles:
- name: encode
  plugins:
  - pluginRef: encode-filter
- name: decode
  plugins:
  - pluginRef: decode-filter
`

// epdConfig configures E/P/D (encode + prefill + decode) using disagg-profile-handler.
// The encode stage is triggered only for multimodal requests (image_url / video_url / input_audio).
const epdConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: encode-filter
- type: prefill-filter
- type: decode-filter
- type: approx-prefix-cache-producer
  parameters:
    blockSizeTokens: 16
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 256
- type: prefix-cache-scorer
- type: max-score-picker
- type: always-disagg-multimodal-decider
- type: prefix-based-pd-decider
  parameters:
    nonCachedTokens: 16
- type: disagg-profile-handler
  parameters:
    deciders:
      encode: always-disagg-multimodal-decider
      prefill: prefix-based-pd-decider
schedulingProfiles:
- name: encode
  plugins:
  - pluginRef: encode-filter
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
`

// EPP configuration for running with P/D using the unified disagg-profile-handler
const pdConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
  parameters:
    blockSizeTokens: 16
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 256
- type: prefix-cache-scorer
- type: prefill-filter
- type: decode-filter
- type: max-score-picker
- type: prefix-based-pd-decider
  parameters:
    nonCachedTokens: 16
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
    weight: 2
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
`

// EPP configuration for running decode-only using disagg-profile-handler (no prefill, no encode)
const decodeOnlyConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
  parameters:
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 256
- type: prefix-cache-scorer
- type: encode-filter
- type: prefill-filter
- type: decode-filter
- type: max-score-picker
- type: disagg-profile-handler
schedulingProfiles:
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
`

// EPP config for running with precise prefix scoring (i.e. KV events).
const kvConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: token-producer
  parameters:
    modelName: Qwen/Qwen2.5-1.5B-Instruct
    vllm:
      url: http://localhost:8000
- type: precise-prefix-cache-scorer
  parameters:
    tokenProcessorConfig:
      blockSize: 16
      hashSeed: "42"
    kvEventsConfig:
      zmqEndpoint: tcp://0.0.0.0:5557
    indexerConfig:
      kvBlockIndexConfig:
        enableMetrics: false                  # enable kv-block index metrics (prometheus)
        metricsLoggingInterval: 6000000000    # log kv-block metrics as well (1m in nanoseconds)
- type: decode-filter
- type: max-score-picker
- type: disagg-profile-handler
schedulingProfiles:
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: precise-prefix-cache-scorer
    weight: 10
`

// Alias of kvConfig retained for tests that reference the external-tokenizer name.
const kvExternalTokenizerConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: token-producer
  parameters:
    modelName: Qwen/Qwen2.5-1.5B-Instruct
    vllm:
      url: http://localhost:8000
- type: precise-prefix-cache-scorer
  parameters:
    tokenProcessorConfig:
      blockSize: 16
      hashSeed: "42"
    kvEventsConfig:
      zmqEndpoint: tcp://0.0.0.0:5557
    indexerConfig:
      kvBlockIndexConfig:
        enableMetrics: false
- type: decode-filter
- type: max-score-picker
- type: disagg-profile-handler
schedulingProfiles:
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: precise-prefix-cache-scorer
    weight: 10
`

// EPP configuration for multimodal cache-affinity routing. Pairs the
// mm-embeddings-cache producer+scorer with token-producer at loopback; the sim
// sidecar (swapped in by usesSimRender) provides mm_features.
const mmCacheAffinityConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: token-producer
  parameters:
    modelName: food-review
    vllm:
      url: http://localhost:8000
- type: mm-embeddings-cache-producer
  parameters:
    cacheSizeInMBPerServer: 64
- type: mm-embeddings-cache-scorer
- type: decode-filter
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: mm-embeddings-cache-scorer
    weight: 10
  - pluginRef: max-score-picker
`

// Same as mmCacheAffinityConfig but the token-producer uses the estimate
// backend, which derives mm_features in-EPP and never calls the render endpoint.
const mmCacheAffinityEstimateConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: token-producer
  parameters:
    estimate: {}
- type: mm-embeddings-cache-producer
  parameters:
    cacheSizeInMBPerServer: 64
- type: mm-embeddings-cache-scorer
- type: decode-filter
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: mm-embeddings-cache-scorer
    weight: 10
  - pluginRef: max-score-picker
`

// EPP configuration for running scale model server test
const scaleConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
`

// EPP configuration for running with vLLM Data Parallel support
const dataParallelConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: decode-filter
- type: max-score-picker
- type: data-parallel-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
`
