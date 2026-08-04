/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package epp

import "testing"

// e2eConfigsForSmoke mirrors the YAML strings in test/e2e/configs_test.go so
// the hermetic harness validates that the e2e fixtures actually parse +
// instantiate plugins. Required because the real e2e suite only runs in CI
// with a kind cluster, and an EPP-startup parse error there manifests as an
// opaque pod CrashLoopBackOff + Ginkgo timeout (#1134 review feedback). This
// keeps the same orphan-field-mismatch loop short.
//
// When the fixtures in test/e2e/configs_test.go are updated, mirror them here.
var e2eConfigsForSmoke = map[string]string{
	"simpleConfig": `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
  parameters:
    maxPrefixTokensToMatch: 16384
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
`,
	"deprecatedPdConfig": `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefill-header-handler
- type: approx-prefix-cache-producer
  parameters:
    blockSizeTokens: 16
    maxPrefixTokensToMatch: 16384
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
`,
	"pdConfig": `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
  parameters:
    blockSizeTokens: 16
    maxPrefixTokensToMatch: 16384
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
`,
	"epdConfig": `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: encode-filter
- type: prefill-filter
- type: decode-filter
- type: approx-prefix-cache-producer
  parameters:
    blockSizeTokens: 16
    maxPrefixTokensToMatch: 16384
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
`,
	"decodeOnlyConfig": `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
  parameters:
    maxPrefixTokensToMatch: 16384
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
`,
	"gpuUtilConfig": `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: dcgm-data-source
  name: dcgm-source
  parameters:
    port: 9400
- type: dcgm-extractor
  name: dcgm-extractor
- type: endpoint-attribute-filter
  name: gpu-utilization-filter
  parameters:
    attribute: "GPUUtilization/dcgm-extractor"
    onMissing: "Pass"
    fallbackOnEmpty: true
    algorithm:
      type: "threshold"
      threshold:
        operator: "LessThanOrEqual"
        value: 0.90
- type: endpoint-attribute-scorer
  name: gpu-utilization-scorer
  parameters:
    attributeKey: "GPUUtilization/dcgm-extractor"
    algorithm:
      type: "linear_lower_is_better"
      normalization:
        fixedRange:
          min: 0.0
          max: 1.0
- type: kv-cache-utilization-scorer
- type: max-score-picker
- type: single-profile-handler
- type: decode-filter
dataLayer:
  sources:
  - pluginRef: dcgm-source
    extractors:
    - pluginRef: dcgm-extractor
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: gpu-utilization-filter
  - pluginRef: gpu-utilization-scorer
    weight: 2.0
  - pluginRef: kv-cache-utilization-scorer
    weight: 1.0
  - pluginRef: max-score-picker
`,
}

func TestE2EConfigs_StartupParseSmoke(t *testing.T) {
	for name, yaml := range e2eConfigsForSmoke {
		t.Run(name, func(t *testing.T) {
			// NewTestHarness fails the test if config-parse / plugin-init
			// errors — exactly what we want to catch before pushing.
			_ = NewTestHarness(t.Context(), t, WithConfigText(yaml))
		})
	}
}
