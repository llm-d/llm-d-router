/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package e2e

import (
	_ "embed"
	"fmt"
)

// Simple EPP configuration for running without P/D.
//
//go:embed testdata/configs/simple.yaml
var simpleConfig string

// modelRolloutConfig adds models-responder, which screens requests by the model
// lists the Pods report.
//
//go:embed testdata/configs/model-rollout.yaml
var modelRolloutConfig string

// epdEncodeDecodeConfig configures E/PD (encode + P/D) using disagg-profile-handler.
// The encode stage is triggered only for multimodal requests (image_url / video_url / input_audio).
//
//go:embed testdata/configs/epd-encode-decode.yaml
var epdEncodeDecodeConfig string

// epdConfig configures E/P/D (encode + prefill + decode) using disagg-profile-handler.
// The encode stage is triggered only for multimodal requests (image_url / video_url / input_audio).
//
//go:embed testdata/configs/epd.yaml
var epdConfig string

// generateEncodeConfig is the encode-only EPP config for /inference/v1/generate.
// Uses single-profile-handler so the EPP routes directly to encode pods without
// requiring a decode stage.
//
//go:embed testdata/configs/generate-encode.yaml
var generateEncodeConfig string

// generatePrefillConfig is the prefill-only EPP config for /inference/v1/generate.
// Uses single-profile-handler so the EPP routes directly to prefill pods without
// requiring a decode stage.
//
//go:embed testdata/configs/generate-prefill.yaml
var generatePrefillConfig string

// EPP configuration for running with P/D using the unified disagg-profile-handler.
// pdConfig uses vllmhttp-parser as the request handler so the EPP can parse
// both OpenAI-style and /inference/v1/generate (token-in) traffic. The parser
// delegates non-generate paths to the embedded OpenAI parser, so existing
// chat/completions tests are unaffected.
//
//go:embed testdata/configs/pd.yaml
var pdConfig string

// EPP configuration for running decode-only using disagg-profile-handler (no prefill, no encode).
//
//go:embed testdata/configs/decode-only.yaml
var decodeOnlyConfig string

// renderURL is the vllm-render Service URL for the token-producer plugin. The
// renderer is shared across parallel processes and lives in the base namespace,
// so the URL has to be fully qualified: a bare hostname resolves against the
// EPP's own per-process namespace, which has no renderer.
func renderURL() string {
	return fmt.Sprintf("http://vllm-render.%s.svc:%s", baseNsName, vllmRenderPort)
}

// kvConfig returns the EPP config for precise prefix scoring with KV events.
func kvConfig() string {
	return fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: token-producer
  parameters:
    modelName: Qwen/Qwen2.5-1.5B-Instruct
    vllm:
      url: %s
- type: precise-prefix-cache-producer
  parameters:
    tokenProcessorConfig:
      blockSizeTokens: 16
      hashSeed: "42"
    kvEventsConfig:
      zmqEndpoint: tcp://0.0.0.0:5557
      allowUntrustedPublishers: true  # simulator pods publish to the EPP's bound socket
    indexerConfig:
      kvBlockIndexConfig:
        enableMetrics: false                  # enable kv-block index metrics (prometheus)
        metricsLoggingInterval: 60000000000   # log kv-block metrics as well (1m in nanoseconds)
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
- type: decode-filter
- type: max-score-picker
- type: disagg-profile-handler
schedulingProfiles:
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 10
`, renderURL())
}

// kvExternalTokenizerConfig returns the EPP config for the KV-events test that
// covers both completions and chat completions.
func kvExternalTokenizerConfig() string {
	return fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: token-producer
  parameters:
    modelName: Qwen/Qwen2.5-1.5B-Instruct
    vllm:
      url: %s
- type: precise-prefix-cache-producer
  parameters:
    tokenProcessorConfig:
      blockSizeTokens: 16
      hashSeed: "42"
    kvEventsConfig:
      zmqEndpoint: tcp://0.0.0.0:5557
      allowUntrustedPublishers: true  # simulator pods publish to the EPP's bound socket
    indexerConfig:
      kvBlockIndexConfig:
        enableMetrics: false
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
- type: decode-filter
- type: max-score-picker
- type: disagg-profile-handler
schedulingProfiles:
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 10
`, renderURL())
}

// EPP configuration for running scale model server test.
//
//go:embed testdata/configs/scale.yaml
var scaleConfig string

// EPP configuration for running with vLLM Data Parallel support.
//
//go:embed testdata/configs/data-parallel.yaml
var dataParallelConfig string
