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

package epp

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
	fwkk8s "github.com/llm-d/llm-d-router/test/framework/k8s"
)

// TestDataParallelRank_VllmInference runs pod discovery, scheduling, and the header
// mutation together. The pool has one pod without ranks and one pod that serves two
// data-parallel ranks behind its port. The request is a vllm.Inference call, where
// the rank header reaches the model server as gRPC metadata.
func TestDataParallelRank_VllmInference(t *testing.T) {
	const (
		rankCountAnnotation = "llm-d.ai/data-parallel-size-local"
		clientRank          = "7"
	)
	endpoints := []string{"pod-0-rank-0", "pod-1-rank-0", "pod-1-rank-1"}

	tests := []struct {
		name string
		// queues holds the queue size of each entry of endpoints. The shortest queue wins.
		queues       [3]int
		wantEndpoint string
		wantRank     string
	}{
		{
			name:         "a rank endpoint gets its rank in place of the client value",
			queues:       [3]int{5, 9, 0},
			wantEndpoint: "192.168.1.2:8000",
			wantRank:     "1",
		},
		{
			name:         "a pod without ranks keeps the client value",
			queues:       [3]int{0, 5, 9},
			wantEndpoint: "192.168.1.1:8000",
			wantRank:     clientRank,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := t.Context()
			h := eppharness.NewTestHarness(ctx, t, eppharness.WithStandardMode(),
				eppharness.WithConfigText(testConfigWithVllmInferenceParser)).WithBaseResources()

			metrics := make(map[types.NamespacedName]*fwkdl.Metrics, len(endpoints))
			for i, name := range endpoints {
				metrics[types.NamespacedName{Namespace: h.Namespace, Name: name}] = &fwkdl.Metrics{
					WaitingQueueSize: tt.queues[i],
					ActiveModels:     map[string]int{},
					WaitingModels:    map[string]int{},
				}
			}
			h.SetPodMetrics(metrics)

			for i, annotations := range []map[string]string{nil, {rankCountAnnotation: "2"}} {
				pod := fwkk8s.MakePod(fmt.Sprintf("pod-%d", i)).
					Namespace(h.Namespace).
					ReadyCondition().
					Labels(map[string]string{"app": eppharness.TestPoolName}).
					IP(fmt.Sprintf("192.168.1.%d", i+1)).
					Complete().
					ObjRef()
				pod.Annotations = annotations

				// Create drops the status, so the test sets it again.
				status := pod.Status
				require.NoError(t, eppharness.K8sClient().Create(ctx, pod))
				pod.Status = status
				require.NoError(t, eppharness.K8sClient().Status().Update(ctx, pod))
			}
			h.WaitForSync(len(endpoints), modelMyModel).WaitForMetricsDelivery()

			payload, err := grpcframe.MarshalProto(&vllmpb.GenerateRequest{
				Model:  modelMyModel,
				Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{11, 22, 33}}},
			})
			require.NoError(t, err)
			// The EPP answers the request with one header response and one body response.
			responses := streamRequests(t, h, fwkepp.ReqRaw(map[string]string{
				reqcommon.RequestIDHeaderKey:   "rank-request",
				":path":                        vllmpb.Inference_GenerateStream_FullMethodName,
				metadata.ObjectiveKey:          inferenceObjectiveWithPriority4,
				routing.DataParallelRankHeader: clientRank,
			}, string(payload)), 2)

			setHeaders := responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
			require.Equal(t, tt.wantEndpoint, headerValue(setHeaders, metadata.DestinationEndpointKey))
			require.Equal(t, tt.wantRank, headerValue(setHeaders, routing.DataParallelRankHeader))
		})
	}
}
