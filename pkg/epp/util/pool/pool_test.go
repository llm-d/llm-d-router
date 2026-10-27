/*
Copyright 2025 The Kubernetes Authors.

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

package pool

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

func TestInferencePoolToEndpointPool(t *testing.T) {
	t.Run("nil input returns nil", func(t *testing.T) {
		assert.Nil(t, InferencePoolToEndpointPool(nil))
	})

	t.Run("fields are mapped", func(t *testing.T) {
		inferencePool := &v1.InferencePool{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "my-pool",
				Namespace: "my-ns",
			},
			Spec: v1.InferencePoolSpec{
				Selector: v1.LabelSelector{
					MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "vllm"},
				},
				TargetPorts: []v1.Port{{Number: 8000}, {Number: 8001}},
				AppProtocol: v1.AppProtocolHTTP,
			},
		}

		endpointPool := InferencePoolToEndpointPool(inferencePool)
		require.NotNil(t, endpointPool)
		assert.Equal(t, "my-pool", endpointPool.Name)
		assert.Equal(t, "my-ns", endpointPool.Namespace)
		assert.Equal(t, []int{8000, 8001}, endpointPool.TargetPorts)
		assert.Equal(t, v1.AppProtocolHTTP, endpointPool.AppProtocol)
		assert.True(t, endpointPool.Selector.Matches(labels.Set{"app": "vllm"}))
		assert.False(t, endpointPool.Selector.Matches(labels.Set{"app": "other"}))
	})
}
