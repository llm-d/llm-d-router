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

package bylabel

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

func createEndpoint(nsn k8stypes.NamespacedName, ipaddr string, labels map[string]string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      nsn,
			Address: ipaddr,
			Labels:  labels,
		},
		&fwkdl.Metrics{},
		nil,
	)
}

func TestRoleFilterDecodeRole(t *testing.T) {
	endpoints := []scheduling.Endpoint{
		createEndpoint(k8stypes.NamespacedName{Name: "decode-pod"}, "10.0.0.1",
			map[string]string{RoleLabel: RoleDecode}),
		createEndpoint(k8stypes.NamespacedName{Name: "prefill-pod"}, "10.0.0.2",
			map[string]string{RoleLabel: RolePrefill}),
		createEndpoint(k8stypes.NamespacedName{Name: "pd-pod"}, "10.0.0.3",
			map[string]string{RoleLabel: RolePrefillDecode}),
		createEndpoint(k8stypes.NamespacedName{Name: "epd-pod"}, "10.0.0.5",
			map[string]string{RoleLabel: RoleEncodePrefillDecode}),
		createEndpoint(k8stypes.NamespacedName{Name: "encode-pod"}, "10.0.0.6",
			map[string]string{RoleLabel: RoleEncode}),
		createEndpoint(k8stypes.NamespacedName{Name: "ep-pod"}, "10.0.0.7",
			map[string]string{RoleLabel: RoleEncodePrefill}),
		createEndpoint(k8stypes.NamespacedName{Name: "no-label-pod"}, "10.0.0.8",
			map[string]string{"app": "vllm"}),
		createEndpoint(k8stypes.NamespacedName{Name: "empty-labels-pod"}, "10.0.0.9",
			map[string]string{}),
	}

	ctx := utils.NewTestContext(t)
	rf := NewDecodeRole()

	filtered := rf.Filter(ctx, nil, endpoints)

	names := make([]string, len(filtered))
	for i, ep := range filtered {
		names[i] = ep.GetMetadata().ID.Name
	}

	assert.ElementsMatch(t, []string{
		"decode-pod", "pd-pod", "epd-pod",
		"no-label-pod", "empty-labels-pod",
	}, names)
}

func TestRoleFilterPrefillRole(t *testing.T) {
	endpoints := []scheduling.Endpoint{
		createEndpoint(k8stypes.NamespacedName{Name: "decode-pod"}, "10.0.0.1",
			map[string]string{RoleLabel: RoleDecode}),
		createEndpoint(k8stypes.NamespacedName{Name: "prefill-pod"}, "10.0.0.2",
			map[string]string{RoleLabel: RolePrefill}),
		createEndpoint(k8stypes.NamespacedName{Name: "pd-pod"}, "10.0.0.3",
			map[string]string{RoleLabel: RolePrefillDecode}),
		createEndpoint(k8stypes.NamespacedName{Name: "epd-pod"}, "10.0.0.5",
			map[string]string{RoleLabel: RoleEncodePrefillDecode}),
		createEndpoint(k8stypes.NamespacedName{Name: "encode-pod"}, "10.0.0.6",
			map[string]string{RoleLabel: RoleEncode}),
		createEndpoint(k8stypes.NamespacedName{Name: "ep-pod"}, "10.0.0.7",
			map[string]string{RoleLabel: RoleEncodePrefill}),
		createEndpoint(k8stypes.NamespacedName{Name: "no-label-pod"}, "10.0.0.8",
			map[string]string{"app": "vllm"}),
	}

	ctx := utils.NewTestContext(t)
	rf := NewPrefillRole()

	filtered := rf.Filter(ctx, nil, endpoints)

	names := make([]string, len(filtered))
	for i, ep := range filtered {
		names[i] = ep.GetMetadata().ID.Name
	}

	assert.ElementsMatch(t, []string{
		"prefill-pod", "pd-pod", "epd-pod", "ep-pod",
	}, names)
}

func TestRoleFiltersUseDynamicLayout(t *testing.T) {
	endpoints := []scheduling.Endpoint{
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "prefill-pod-1"}, "10.0.0.1",
			map[string]string{RoleLabel: RolePrefill, TensorParallelSizeLabel: "1", RoleTransferGroupLabel: "test-group", ModelIdentityLabel: "test-model-v1", AcceleratorClassLabel: "test-accelerator"}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "prefill-pod-2"}, "10.0.0.2",
			map[string]string{RoleLabel: RolePrefill, TensorParallelSizeLabel: "1", RoleTransferGroupLabel: "test-group", ModelIdentityLabel: "test-model-v1", AcceleratorClassLabel: "test-accelerator"}),
		createEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "decode-pod"}, "10.0.0.3",
			map[string]string{RoleLabel: RoleDecode, TensorParallelSizeLabel: "1", RoleTransferGroupLabel: "test-group", ModelIdentityLabel: "test-model-v1", AcceleratorClassLabel: "test-accelerator"}),
	}

	manager := NewLayoutManager()
	prefillFilter := NewPrefillRole().WithLayoutManager(manager)
	decodeFilter := NewDecodeRole().WithLayoutManager(manager)
	ctx := utils.NewTestContext(t)

	// Observing either filter seeds the manager from the discovered Pod roles.
	require.Len(t, prefillFilter.Filter(ctx, nil, endpoints), 2)
	result, err := manager.Apply("p_1tp1_d_2tp1_m_none")
	require.NoError(t, err)
	require.Equal(t, "ok", result.Status)

	prefill := prefillFilter.Filter(ctx, nil, endpoints)
	decode := decodeFilter.Filter(ctx, nil, endpoints)
	require.Len(t, prefill, 1)
	require.Len(t, decode, 2)
	assert.Equal(t, "prefill-pod-1", prefill[0].GetMetadata().ID.Name)
	assert.ElementsMatch(t, []string{"prefill-pod-2", "decode-pod"}, []string{
		decode[0].GetMetadata().ID.Name,
		decode[1].GetMetadata().ID.Name,
	})
}

func TestRoleFilterEncodeRole(t *testing.T) {
	endpoints := []scheduling.Endpoint{
		createEndpoint(k8stypes.NamespacedName{Name: "decode-pod"}, "10.0.0.1",
			map[string]string{RoleLabel: RoleDecode}),
		createEndpoint(k8stypes.NamespacedName{Name: "prefill-pod"}, "10.0.0.2",
			map[string]string{RoleLabel: RolePrefill}),
		createEndpoint(k8stypes.NamespacedName{Name: "encode-pod"}, "10.0.0.3",
			map[string]string{RoleLabel: RoleEncode}),
		createEndpoint(k8stypes.NamespacedName{Name: "ep-pod"}, "10.0.0.4",
			map[string]string{RoleLabel: RoleEncodePrefill}),
		createEndpoint(k8stypes.NamespacedName{Name: "epd-pod"}, "10.0.0.5",
			map[string]string{RoleLabel: RoleEncodePrefillDecode}),
		createEndpoint(k8stypes.NamespacedName{Name: "no-label-pod"}, "10.0.0.6",
			map[string]string{"app": "vllm"}),
	}

	ctx := utils.NewTestContext(t)
	rf := NewEncodeRole()

	filtered := rf.Filter(ctx, nil, endpoints)

	names := make([]string, len(filtered))
	for i, ep := range filtered {
		names[i] = ep.GetMetadata().ID.Name
	}

	assert.ElementsMatch(t, []string{
		"encode-pod", "ep-pod", "epd-pod",
	}, names)
}

func TestRoleFilterFactory(t *testing.T) {
	tests := []struct {
		name         string
		roleName     string
		expectedName string
	}{
		{
			name:         "DecodeRoleFactory returns RoleFilter",
			roleName:     DecodeRoleType,
			expectedName: "test-decode",
		},
		{
			name:         "PrefillRoleFactory returns RoleFilter",
			roleName:     PrefillRoleType,
			expectedName: "test-prefill",
		},
		{
			name:         "EncodeRoleFactory returns RoleFilter",
			roleName:     EncodeRoleType,
			expectedName: "test-encode",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rf *RoleFilter
			switch tt.roleName {
			case DecodeRoleType:
				p, err := DecodeRoleFactory("test-decode", nil, nil)
				require.NoError(t, err)
				var ok bool
				rf, ok = p.(*RoleFilter)
				require.True(t, ok, "factory should return *RoleFilter")
			case PrefillRoleType:
				p, err := PrefillRoleFactory("test-prefill", nil, nil)
				require.NoError(t, err)
				var ok bool
				rf, ok = p.(*RoleFilter)
				require.True(t, ok, "factory should return *RoleFilter")
			case EncodeRoleType:
				p, err := EncodeRoleFactory("test-encode", nil, nil)
				require.NoError(t, err)
				var ok bool
				rf, ok = p.(*RoleFilter)
				require.True(t, ok, "factory should return *RoleFilter")
			}

			assert.Equal(t, tt.roleName, rf.TypedName().Type)
			assert.Equal(t, tt.expectedName, rf.TypedName().Name)
		})
	}
}

func TestRoleFilterWithName(t *testing.T) {
	rf := NewDecodeRole()
	assert.Equal(t, DecodeRoleType, rf.TypedName().Name)
	assert.Equal(t, DecodeRoleType, rf.TypedName().Type)

	rf.WithName("my-custom-name")
	assert.Equal(t, "my-custom-name", rf.TypedName().Name)
	assert.Equal(t, DecodeRoleType, rf.TypedName().Type)
}

func TestRoleFilterEmptyEndpoints(t *testing.T) {
	ctx := utils.NewTestContext(t)
	rf := NewDecodeRole()

	result := rf.Filter(ctx, nil, []scheduling.Endpoint{})
	assert.Empty(t, result)

	result = rf.Filter(ctx, nil, nil)
	assert.Empty(t, result)
}
