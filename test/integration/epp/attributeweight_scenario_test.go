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

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
	integration "github.com/llm-d/llm-d-router/test/integration"
)

// Keep metrics neutral so attribute weight is the only scoring signal.
const gpuWeightScorerConfig = `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: label-producer
  name: endpoint-labels
  parameters:
    labels:
    - label: nvidia.com/gpu.product
      attributeKey: gpu.product
    - label: topology.kubernetes.io/region
      attributeKey: region
- type: endpoint-attribute-weight-scorer
  name: gpu-product
  parameters:
    attributeKey: gpu.product
    producer: endpoint-labels
    weights:
      NVIDIA-H100: 2
      NVIDIA-A100: 1
- type: endpoint-attribute-weight-scorer
  name: region
  parameters:
    attributeKey: region
    producer: endpoint-labels
    weights:
      region-1: 2
      region-2: 1
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
  - pluginRef: region
    weight: 3
  - pluginRef: gpu-product
    weight: 1
  - pluginRef: max-score-picker
`

type gpuPod struct {
	index  int
	label  string
	region string
}

func withGPUPods(h *TestHarness, pods []gpuPod) *TestHarness {
	h.t.Helper()

	metricsMap := make(map[types.NamespacedName]*fwkdl.Metrics, len(pods))
	for _, p := range pods {
		key := types.NamespacedName{Namespace: h.Namespace, Name: fmt.Sprintf("pod-%d-rank-0", p.index)}
		metricsMap[key] = fwkdl.NewMetrics()
	}
	h.metricsBackend.SetPodMetrics(metricsMap)

	for _, p := range pods {
		labels := map[string]string{"app": testPoolName}
		if p.label != "" {
			labels["nvidia.com/gpu.product"] = p.label
		}
		if p.region != "" {
			labels["topology.kubernetes.io/region"] = p.region
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

func TestAttributeWeightScorer(t *testing.T) {
	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithConfigText(gpuWeightScorerConfig), WithStandardMode(), WithEmitEndpointScores())
	h = h.WithBaseResources()

	pods := []gpuPod{
		{index: 0, label: "NVIDIA-A100", region: "region-1"},
		{index: 1, label: "NVIDIA-H100", region: "region-1"},
		{index: 2, label: "NVIDIA-H100", region: "region-2"},
		{index: 3, label: "NVIDIA-A100", region: "region-2"},
		{index: 4, label: "NVIDIA-H100"},
		{index: 5, region: "region-1"},
		{index: 6, label: "NVIDIA-H100", region: "unknown"},
		{index: 7, label: "NVIDIA-A10", region: "region-1"},
	}
	withGPUPods(h, pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	requests := integration.ReqRaw(
		map[string]string{"hi": "mom", reqcommon.RequestIDHeaderKey: "test-request-id"},
		"passthrough-body",
	)
	responses, err := integration.StreamedRequest(t, h.Client, requests, 2)
	require.NoError(t, err)
	require.Len(t, responses, 2)

	res := responses[0]
	require.NotNil(t, res.DynamicMetadata, "expected DynamicMetadata in the request headers response")
	envoyLB, ok := res.DynamicMetadata.Fields[metadata.DestinationEndpointNamespace]
	require.True(t, ok, "expected envoy.lb namespace in DynamicMetadata")
	endpoint, ok := envoyLB.GetStructValue().Fields[metadata.DestinationEndpointKey]
	require.True(t, ok, "expected destination endpoint in envoy.lb namespace")
	require.Equal(t, "192.168.1.2:8000", endpoint.GetStringValue(), "expected routing to the H100 pod in region-1")
	scoresValue, ok := envoyLB.GetStructValue().Fields[metadata.DestinationEndpointScoresKey]
	require.True(t, ok, "expected endpoint scores in envoy.lb namespace")
	scores := scoresValue.GetStructValue().Fields
	require.Len(t, scores, len(pods), "expected missing and unrecognized values to remain eligible")
	for endpoint, want := range map[string]float64{
		"192.168.1.1:8000": 3.5,
		"192.168.1.2:8000": 4,
		"192.168.1.3:8000": 2.5,
		"192.168.1.4:8000": 2,
		"192.168.1.5:8000": 2.5,
		"192.168.1.6:8000": 3.5,
		"192.168.1.7:8000": 2.5,
		"192.168.1.8:8000": 3.5,
	} {
		score, ok := scores[endpoint]
		require.True(t, ok, "expected a score for endpoint %s", endpoint)
		require.InDelta(t, want, score.GetNumberValue(), 1e-9, "unexpected score for endpoint %s", endpoint)
	}
}
