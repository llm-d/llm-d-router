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

package runner

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

const e2eConfigsGlob = "../../../test/e2e/testdata/configs/*.yaml"

var e2eArtifactSmokeConfigs = []string{
	"../../../test/e2e/artifacts/config/pd-topology-epp-config.yaml",
	"../../../test/e2e/artifacts/config/sim-epp-gpu-config.yaml",
}

// inlineSmokeConfigs covers startup parse and dependency validation for
// configurations that do not have a standalone YAML file under test/e2e/.
var inlineSmokeConfigs = []struct {
	name string
	yaml string
}{
	{
		name: "coordinator_prefill_topology",
		yaml: `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: prefill-filter
- type: max-score-picker
- type: topology-extractor
- type: topology-stamp-handler
  parameters:
    profileName: default
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: max-score-picker
`,
	},
	{
		name: "coordinator_decode_topology",
		yaml: `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: decode-filter
- type: max-score-picker
- type: topology-extractor
- type: topology-affinity-filter
  parameters:
    peerTopologyHeader: x-llm-d-peer-topology
- type: topology-affinity-scorer
  parameters:
    peerTopologyHeader: x-llm-d-peer-topology
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: topology-affinity-filter
  - pluginRef: max-score-picker
  - pluginRef: topology-affinity-scorer
    weight: 1
`,
	},
	{
		// Regression test for #1471: a precise-prefix-cache-producer paired with
		// an explicit vllm-backed token-producer must load cleanly through the
		// runner validation path, where ValidateAndOrderDataDependencies runs
		// Level-2 attribute validation after auto-defaults.
		name: "precise_vllm_token_producer",
		yaml: `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: token-producer
  parameters:
    modelName: test-model
    vllm:
      url: http://localhost:9999
- type: precise-prefix-cache-producer
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
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
`,
	},
	{
		// Regression test for #3982: a token-only pool can pair
		// precise-prefix-cache-producer with a passthrough token-producer without
		// configuring an unused HTTP render endpoint.
		name: "precise_passthrough_token_producer",
		yaml: `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: token-producer
  parameters:
    backend: passthrough
- type: precise-prefix-cache-producer
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
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
`,
	},
	{
		name: "endpoint_attribute_builtin_metrics",
		yaml: `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: endpoint-attribute-filter
  name: running-requests-filter
  parameters:
    attribute: RunningRequestsSize
    onMissing: Pass
    algorithm:
      type: threshold
      threshold:
        operator: LessThan
        value: 64
- type: endpoint-attribute-scorer
  name: ea-queue-scorer
  parameters:
    attributeKey: WaitingQueueSize
    algorithm:
      type: linear_lower_is_better
      normalization:
        adaptiveRange: {}
- type: endpoint-attribute-scorer
  name: ea-kv-cache-scorer
  parameters:
    attributeKey: KVCacheUsagePercent
    algorithm:
      type: linear_lower_is_better
      normalization:
        fixedRange:
          min: 0.0
          max: 1.0
- type: queue-scorer
- type: running-requests-size-scorer
- type: kv-cache-utilization-scorer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: running-requests-filter
  - pluginRef: ea-queue-scorer
    weight: 2
  - pluginRef: ea-kv-cache-scorer
    weight: 2
  - pluginRef: queue-scorer
    weight: 1
  - pluginRef: running-requests-size-scorer
    weight: 1
  - pluginRef: kv-cache-utilization-scorer
    weight: 1
  - pluginRef: max-score-picker
`,
	},
}

func TestE2EConfigs_StartupParseSmoke(t *testing.T) {
	files, err := filepath.Glob(e2eConfigsGlob)
	require.NoError(t, err, "failed to glob e2e configs")
	require.NotEmpty(t, files, "no e2e configs found at %s", e2eConfigsGlob)

	files = append(files, e2eArtifactSmokeConfigs...)
	for _, file := range files {
		name := strings.ReplaceAll(strings.TrimSuffix(filepath.Base(file), ".yaml"), "-", "_")
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(file)
			require.NoError(t, err, "failed to read %s", file)
			validateConfigText(t, string(data))
		})
	}

	for _, tc := range inlineSmokeConfigs {
		t.Run(tc.name, func(t *testing.T) {
			validateConfigText(t, tc.yaml)
		})
	}
}
