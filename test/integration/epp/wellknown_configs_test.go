/*
Copyright 2026 The Kubernetes Authors.

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

package epp

import (
	"testing"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
)

// wellKnownConfigs are configs in the llm-d well-lit path guides.
// precise-prefix-cache-aware config is not included because it requires a tokenizer.
// Failed tests indicate EPP regression, or that the config needs to be updated.
var wellKnownConfigs = map[string]struct {
	yaml            string
	expectedPlugins []configapi.PluginSpec
}{
	"optimized-baseline": {
		yaml: `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: prefix-cache-scorer
- type: no-hit-lru-scorer
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
    weight: 2
  - pluginRef: kv-cache-utilization-scorer
    weight: 2
  - pluginRef: prefix-cache-scorer
    weight: 3
  - pluginRef: no-hit-lru-scorer
    weight: 2
`,
		expectedPlugins: []configapi.PluginSpec{
			{Name: "queue-scorer", Type: "queue-scorer"},
			{Name: "kv-cache-utilization-scorer", Type: "kv-cache-utilization-scorer"},
			{Name: "no-hit-lru-scorer", Type: "no-hit-lru-scorer"},
			{Name: "prefix-cache-scorer", Type: "prefix-cache-scorer"},
			// The producer is auto created because the prefix-cache-scorer consumes its data.
			{Name: "approx-prefix-cache-producer", Type: "approx-prefix-cache-producer"},
		},
	},
	// TODO: Enable this test case once the bug #1069 is fixed.
	// 	"tiered-prefix-cache-cpu": {
	// 		yaml: `
	// apiVersion: llm-d.ai/v1alpha1
	// kind: EndpointPickerConfig
	// plugins:
	// - type: approx-prefix-cache-producer
	//   name: cpu-prefix-cache-producer
	//   parameters:
	//     autoTune: false
	//     lruCapacityPerServer: 1000
	// - type: queue-scorer
	// - type: kv-cache-utilization-scorer
	// - type: prefix-cache-scorer
	//   name: gpu-prefix-cache-scorer
	// - type: prefix-cache-scorer
	//   name: cpu-prefix-cache-scorer
	//   parameters:
	//     producer: cpu-prefix-cache-producer
	// schedulingProfiles:
	// - name: default
	//   plugins:
	//   - pluginRef: queue-scorer
	//     weight: 2
	//   - pluginRef: kv-cache-utilization-scorer
	//     weight: 2.0
	//   - pluginRef: gpu-prefix-cache-scorer
	//     weight: 1.0
	//   - pluginRef: cpu-prefix-cache-scorer
	//     weight: 1.0
	// `,
	// 		expectedPlugins: []configapi.PluginSpec{
	// 			{Name: "approx-prefix-cache-producer", Type: "approx-prefix-cache-producer"}, // this one is auto configured.
	// 			{Name: "cpu-prefix-cache-producer", Type: "approx-prefix-cache-producer"},    // this one is configured manually.
	// 			{Name: "queue-scorer", Type: "queue-scorer"},
	// 			{Name: "kv-cache-utilization-scorer", Type: "kv-cache-utilization-scorer"},
	// 			{Name: "gpu-prefix-cache-scorer", Type: "prefix-cache-scorer"},
	// 			{Name: "cpu-prefix-cache-scorer", Type: "prefix-cache-scorer"},
	// 		},
	// 	},
	"pd-disaggregation": {
		yaml: `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: disagg-headers-handler
- type: always-disagg-pd-decider
- type: disagg-profile-handler
  parameters:
    deciderPluginName: always-disagg-pd-decider
- type: prefill-filter
- type: decode-filter
- type: prefix-cache-scorer
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: active-request-scorer
- type: max-score-picker
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: prefix-cache-scorer
    weight: 3
  - pluginRef: queue-scorer
    weight: 2
  - pluginRef: kv-cache-utilization-scorer
    weight: 2
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: active-request-scorer
    weight: 2
  - pluginRef: prefix-cache-scorer
    weight: 3
  - pluginRef: max-score-picker
`,
		expectedPlugins: []configapi.PluginSpec{
			{Name: "disagg-headers-handler", Type: "disagg-headers-handler"},
			{Name: "always-disagg-pd-decider", Type: "always-disagg-pd-decider"},
			{Name: "disagg-profile-handler", Type: "disagg-profile-handler"},
			{Name: "prefill-filter", Type: "by-label"},
			{Name: "decode-filter", Type: "by-label"},
			{Name: "prefix-cache-scorer", Type: "prefix-cache-scorer"},
			// The producer is auto created because the prefix-cache-scorer consumes its data.
			{Name: "approx-prefix-cache-producer", Type: "approx-prefix-cache-producer"},
			{Name: "queue-scorer", Type: "queue-scorer"},
			{Name: "kv-cache-utilization-scorer", Type: "kv-cache-utilization-scorer"},
			{Name: "active-request-scorer", Type: "active-request-scorer"},
			{Name: "max-score-picker", Type: "max-score-picker"},
		},
	},
	"wide-ep-lws": {
		yaml: `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: disagg-headers-handler
- type: always-disagg-pd-decider
- type: disagg-profile-handler
  parameters:
    deciderPluginName: always-disagg-pd-decider
- type: prefill-filter
- type: decode-filter
- type: prefix-cache-scorer
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: active-request-scorer
- type: max-score-picker
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: prefix-cache-scorer
    weight: 3
  - pluginRef: queue-scorer
    weight: 2
  - pluginRef: kv-cache-utilization-scorer
    weight: 2
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: active-request-scorer
    weight: 2
  - pluginRef: prefix-cache-scorer
  - pluginRef: max-score-picker
`,
		expectedPlugins: []configapi.PluginSpec{
			{Name: "disagg-headers-handler", Type: "disagg-headers-handler"},
			{Name: "always-disagg-pd-decider", Type: "always-disagg-pd-decider"},
			{Name: "disagg-profile-handler", Type: "disagg-profile-handler"},
			{Name: "prefill-filter", Type: "by-label"},
			{Name: "decode-filter", Type: "by-label"},
			{Name: "prefix-cache-scorer", Type: "prefix-cache-scorer"},
			// The producer is auto created because the prefix-cache-scorer consumes its data.
			{Name: "approx-prefix-cache-producer", Type: "approx-prefix-cache-producer"},
			{Name: "queue-scorer", Type: "queue-scorer"},
			{Name: "kv-cache-utilization-scorer", Type: "kv-cache-utilization-scorer"},
			{Name: "active-request-scorer", Type: "active-request-scorer"},
			{Name: "max-score-picker", Type: "max-score-picker"},
		},
	},
	"wide-ep-lws-experimental-dp-aware": {
		yaml: `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefix-based-pd-decider
- type: prefill-header-handler
- type: prefill-filter
- type: decode-filter
- type: prefix-cache-scorer
- type: active-request-scorer
- type: queue-scorer
- type: pd-profile-handler
  parameters:
    threshold: 0
    hashBlockSize: 5
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: prefix-cache-scorer
    weight: 3
  - pluginRef: active-request-scorer
    weight: 2
  - pluginRef: queue-scorer
    weight: 2
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: active-request-scorer
    weight: 1
  - pluginRef: queue-scorer
    weight: 1
`,
		expectedPlugins: []configapi.PluginSpec{
			{Name: "prefill-header-handler", Type: "disagg-headers-handler"},
			{Name: "prefill-filter", Type: "by-label"},
			{Name: "decode-filter", Type: "by-label"},
			{Name: "prefix-cache-scorer", Type: "prefix-cache-scorer"},
			// The producer is auto created because the prefix-cache-scorer consumes its data.
			{Name: "approx-prefix-cache-producer", Type: "approx-prefix-cache-producer"},
			{Name: "active-request-scorer", Type: "active-request-scorer"},
			{Name: "queue-scorer", Type: "queue-scorer"},
			{Name: "pd-profile-handler", Type: "pd-profile-handler"},
		},
	},
	"flow-control": {
		yaml: `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: prefix-cache-scorer
- type: round-robin-fairness-policy
- type: fcfs-ordering-policy
- type: concurrency-detector
  parameters:
    maxConcurrency: 132
    concurrencyMode: requests
    headroom: 0.0
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
    weight: 2
  - pluginRef: kv-cache-utilization-scorer
    weight: 2
  - pluginRef: prefix-cache-scorer
    weight: 3
saturationDetector:
  pluginRef: concurrency-detector
flowControl:
  maxBytes: "10Gi"
  maxRequests: "1k"
  defaultRequestTTL: "60s"
  priorityBands:
  - priority: 100
    maxRequests: "500"
    fairnessPolicyRef: round-robin-fairness-policy
    orderingPolicyRef: fcfs-ordering-policy
  - priority: 0
    maxRequests: "200"
    fairnessPolicyRef: round-robin-fairness-policy
    orderingPolicyRef: fcfs-ordering-policy
  - priority: -10
    maxRequests: "50"
    fairnessPolicyRef: round-robin-fairness-policy
    orderingPolicyRef: fcfs-ordering-policy
`,
		expectedPlugins: []configapi.PluginSpec{
			{Name: "queue-scorer", Type: "queue-scorer"},
			{Name: "kv-cache-utilization-scorer", Type: "kv-cache-utilization-scorer"},
			{Name: "prefix-cache-scorer", Type: "prefix-cache-scorer"},
			// The producer is auto created because the prefix-cache-scorer consumes its data.
			{Name: "approx-prefix-cache-producer", Type: "approx-prefix-cache-producer"},
			{Name: "round-robin-fairness-policy", Type: "round-robin-fairness-policy"},
			{Name: "fcfs-ordering-policy", Type: "fcfs-ordering-policy"},
			{Name: "concurrency-detector", Type: "concurrency-detector"},
		},
	},
	"predicted-latency-slo": {
		yaml: `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: prefix-cache-scorer
- type: metrics-data-source
  parameters:
    insecureSkipVerify: true
    path: /metrics
    scheme: http
- type: core-metrics-extractor
- type: predicted-latency-producer
  parameters:
    streamingMode: true
- type: prefix-cache-affinity-filter
  name: strict-affinity-filter
  parameters:
    affinityThreshold: 0.99
- type: prefix-cache-affinity-filter
  name: loose-affinity-filter
  parameters:
    affinityThreshold: 0.8
- type: latency-scorer
- type: weighted-random-picker
- type: slo-headroom-tier-filter
- type: latency-slo-admitter
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: predicted-latency-producer
  - pluginRef: strict-affinity-filter
  - pluginRef: slo-headroom-tier-filter
  - pluginRef: loose-affinity-filter
  - pluginRef: latency-scorer
  - pluginRef: weighted-random-picker
`,
		expectedPlugins: []configapi.PluginSpec{
			{Name: "queue-scorer", Type: "queue-scorer"},
			{Name: "kv-cache-utilization-scorer", Type: "kv-cache-utilization-scorer"},
			{Name: "prefix-cache-scorer", Type: "prefix-cache-scorer"},
			// The producer is auto created because the prefix-cache-scorer consumes its data.
			{Name: "approx-prefix-cache-producer", Type: "approx-prefix-cache-producer"},
			{Name: "metrics-data-source", Type: "metrics-data-source"},
			{Name: "core-metrics-extractor", Type: "core-metrics-extractor"},
			{Name: "predicted-latency-producer", Type: "predicted-latency-producer"},
			{Name: "strict-affinity-filter", Type: "prefix-cache-affinity-filter"},
			{Name: "loose-affinity-filter", Type: "prefix-cache-affinity-filter"},
			{Name: "latency-scorer", Type: "latency-scorer"},
			{Name: "weighted-random-picker", Type: "weighted-random-picker"},
			{Name: "slo-headroom-tier-filter", Type: "slo-headroom-tier-filter"},
			{Name: "latency-slo-admitter", Type: "latency-slo-admitter"},
		},
	},
}

func TestWellKnownConfigs(t *testing.T) {
	for name, tc := range wellKnownConfigs {
		t.Run(name, func(t *testing.T) {
			runner := NewTestHarness(t.Context(), t, WithConfigText(tc.yaml)).Runner

			t.Logf("All plugins: %v", runner.PluginHandle.GetAllPluginsWithNames())
			// Validate that the expected plugins exist in runner.PluginHandle.
			for _, expectedPlugin := range tc.expectedPlugins {
				p := runner.PluginHandle.Plugin(expectedPlugin.Name)
				if p == nil {
					t.Errorf("Expected plugin %q of type %q was not instantiated", expectedPlugin.Name, expectedPlugin.Type)
					continue
				}
				if p.TypedName().Name != expectedPlugin.Name {
					t.Errorf("Plugin %q has unexpected name: got %q, want %q", expectedPlugin.Name, p.TypedName().Name, expectedPlugin.Name)
					continue
				}
				if p.TypedName().Type != expectedPlugin.Type {
					t.Errorf("Plugin %q has unexpected type: got %q, want %q", expectedPlugin.Name, p.TypedName().Type, expectedPlugin.Type)
					continue
				}
			}
		})
	}
}
