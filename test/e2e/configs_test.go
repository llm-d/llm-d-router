package e2e

// Simple EPP configuration for running without P/D
const simpleConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefix-cache-scorer
  parameters:
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 256
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
- type: prefix-cache-scorer
  parameters:
    blockSizeTokens: 16
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 256
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
- type: prefix-cache-scorer
  parameters:
    blockSizeTokens: 16
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 256
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
- type: prefix-cache-scorer
  parameters:
    blockSizeTokens: 16
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 256
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
- type: prefix-cache-scorer
  parameters:
    hashBlockSize: 10
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 256
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
      http: http://localhost:8000
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
      http: http://localhost:8000
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

// EPP config for multimodal encoder-cache affinity plus precise KV-prefix affinity.
// The multimodal data producer runs before scheduling and the scorer consumes its endpoint match info.
const mmCacheAffinityConfig = `apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: multimodal-encoder-producer
  parameters:
    cacheSize: 10000
- type: tokenizer
  parameters:
    modelName: Qwen/Qwen2.5-1.5B-Instruct
- type: precise-prefix-cache-scorer
  parameters:
    tokenProcessorConfig:
      blockSize: 16
      hashSeed: "42"
    kvEventsConfig:
      zmqEndpoint: tcp://0.0.0.0:5557
    indexerConfig:
      prefixStoreConfig:
        blockSize: 16
      tokenizersPoolConfig:
        modelName: Qwen/Qwen2.5-1.5B-Instruct
        uds:
          socketFile: "/tmp/tokenizer/tokenizer-uds.socket"
      kvBlockIndexConfig:
        enableMetrics: false
- type: mm-cache-affinity-scorer
- type: decode-filter
- type: max-score-picker
- type: disagg-profile-handler
schedulingProfiles:
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: tokenizer
  - pluginRef: max-score-picker
  - pluginRef: precise-prefix-cache-scorer
    weight: 10
  - pluginRef: mm-cache-affinity-scorer
    weight: 4
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
