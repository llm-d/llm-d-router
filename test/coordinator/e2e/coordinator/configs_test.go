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

package coordinate2e

// coordinatorConfigNIXL is the coordinator pipeline config for the e-p-d-pools topology.
// ${NAMESPACE} and ${VLLM_RENDER_PORT} are substituted by createCoordinator before the ConfigMap is built.
const coordinatorConfigNIXL = `log_level: 5
server:
  listen_addr: ":8080"
  read_timeout: 30s
  write_timeout: 120s
  # Recreated per spec behind a suite-lived Envoy; drain fast so a deleted
  # coordinator stops serving immediately instead of lingering on a stale
  # endpoint the gateway may still route to. 0s is avoided: it makes the
  # server Shutdown context expire instantly and the process exit non-zero.
  shutdown_timeout: 1s

gateway:
  address: "http://envoy.${NAMESPACE}.svc:8081"
  max_idle_conns_per_host: 100
  idle_conn_timeout: 90s
  timeout: 60s

pipeline:
  kv_connector: kv-nixl
  ec_connector: ec-nixl
  use_openai_format: true
  steps:
    - type: replace-media-urls
      params:
        download_timeout: 30s
        max_concurrent_downloads: 10
    - type: render
      params:
        address: "http://vllm-render.${NAMESPACE}.svc:${VLLM_RENDER_PORT}"
        timeout: 60s
    - type: encode
      params:
        max_parallel: 8
    - type: prefill
    - type: decode
`

// eppConfig is the scheduling config for the single EPP that serves all three
// phases. Each request runs exactly one scheduling profile, named by its
// EPP-Phase header value (see header-phase-profile-handler); the role filters
// narrow the combined encode+prefill+decode pod pool down to the profile's own
// role, since the InferencePool now selects across all three roles at once.
//
// The decode profile uses label-selector-filter instead of decode-filter: the
// decode-filter shorthand hardcodes allowsNoLabel=true (bylabel.NewDecodeRole),
// which is safe only when decode has its own role-scoped InferencePool, so a
// pod without the role label was never a candidate in the first place. Here
// the InferencePool selects across all three roles, so an unlabeled or
// mislabeled encode/prefill pod would otherwise be accepted as a decode
// candidate; label-selector-filter's matchExpressions has no such exception.
const eppConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: openai-parser
- type: encode-filter
- type: prefill-filter
- type: label-selector-filter
  name: decode-filter
  parameters:
    matchExpressions:
    - key: llm-d.ai/role
      operator: In
      values: ["decode", "prefill-decode", "both", "encode-prefill-decode"]
- type: queue-scorer
- type: max-score-picker
- type: header-phase-profile-handler
requestHandler:
  parsers:
  - pluginRef: openai-parser
schedulingProfiles:
- name: encode
  plugins:
  - pluginRef: encode-filter
  - pluginRef: max-score-picker
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: queue-scorer
    weight: 1
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: queue-scorer
    weight: 1
  - pluginRef: max-score-picker
`
