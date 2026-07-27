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
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"
	"k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
	integration "github.com/llm-d/llm-d-router/test/integration"
)

// Keep metrics neutral so attribute weight is the only scoring signal.
const gpuWeightScorerConfig = `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: label-producer
  name: gpu-product
  parameters:
    label: nvidia.com/gpu.product
    attributeKey: gpu.product
    valueType: string
- type: endpoint-attribute-weight-scorer
  parameters:
    attributeKey: gpu.product
    weights:
      NVIDIA-H100: 4
      NVIDIA-L40S: 1
- type: max-score-picker
- type: passthrough-parser
- type: mock-metrics-source
requestHandler:
  parsers:
  - pluginRef: passthrough-parser
dataLayer:
  sources:
  - pluginRef: mock-metrics-source
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: endpoint-attribute-weight-scorer
  - pluginRef: max-score-picker
`

type gpuPod struct {
	index int
	label string
}

func withGPUPods(h *TestHarness, pods []gpuPod) *TestHarness {
	h.t.Helper()

	metricsMap := make(map[types.NamespacedName]*fwkdl.Metrics, len(pods))
	for _, p := range pods {
		metricsMap[types.NamespacedName{Namespace: h.Namespace, Name: fmt.Sprintf("pod-%d-rank-0", p.index)}] = &fwkdl.Metrics{
			ActiveModels:  make(map[string]int),
			WaitingModels: make(map[string]int),
		}
	}
	h.metricsBackend.SetPodMetrics(metricsMap)

	for _, p := range pods {
		labels := map[string]string{"app": testPoolName}
		if p.label != "" {
			labels["nvidia.com/gpu.product"] = p.label
		}

		pod := testutil.MakePod(fmt.Sprintf("pod-%d", p.index)).
			Namespace(h.Namespace).
			ReadyCondition().
			Labels(labels).
			IP(fmt.Sprintf("192.168.1.%d", p.index+1)).
			Complete().
			ObjRef()

		intendedStatus := pod.Status
		require.NoError(h.t, k8sClient.Create(h.ctx, pod), "failed to create pod pod-%d", p.index)
		pod.Status = intendedStatus
		require.NoError(h.t, k8sClient.Status().Update(h.ctx, pod), "failed to update status for pod pod-%d", p.index)
	}
	return h
}

func TestAttributeWeightScorer_PrefersHigherWeightedGPU(t *testing.T) {
	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithConfigText(gpuWeightScorerConfig), WithStandardMode())
	h = h.WithBaseResources()

	pods := []gpuPod{
		{index: 0, label: "NVIDIA-L40S"},
		{index: 1, label: "NVIDIA-H100"},
	}
	withGPUPods(h, pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	requests := integration.ReqRaw(
		map[string]string{"hi": "mom", reqcommon.RequestIDHeaderKey: "test-request-id"},
		"passthrough-body",
	)
	want := ExpectPassthroughRouteTo("192.168.1.2:8000", []byte("passthrough-body"))
	responses, err := integration.StreamedRequest(t, h.Client, requests, len(want))
	require.NoError(t, err)

	if diff := cmp.Diff(want, responses, protocmp.Transform()); diff != "" {
		t.Errorf("expected routing to the higher-weighted (H100) pod at 192.168.1.2:8000 (-want +got): %s", diff)
	}
}

func TestAttributeWeightScorer_UnlabeledPodStaysEligible(t *testing.T) {
	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithConfigText(gpuWeightScorerConfig), WithStandardMode())
	h = h.WithBaseResources()

	pods := []gpuPod{
		{index: 0, label: ""},
		{index: 1, label: "NVIDIA-A10"},
	}
	withGPUPods(h, pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	requests := integration.ReqRaw(
		map[string]string{"hi": "mom", reqcommon.RequestIDHeaderKey: "test-request-id"},
		"passthrough-body",
	)
	// Both endpoints receive the fallback score, so either may be selected.
	responses, err := integration.StreamedRequest(t, h.Client, requests, 2)
	require.NoError(t, err)
	require.Len(t, responses, 2)
}
