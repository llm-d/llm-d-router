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
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func layoutTestEndpoint(name, role, tp string) scheduling.Endpoint {
	return layoutTestEndpointInGroup(name, role, tp, "test-group")
}

func layoutTestEndpointInGroup(name, role, tp, group string) scheduling.Endpoint {
	return scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: k8stypes.NamespacedName{Namespace: "default", Name: name},
		Labels: map[string]string{
			RoleLabel:               role,
			TensorParallelSizeLabel: tp,
			RoleTransferGroupLabel:  group,
			ModelIdentityLabel:      "test-model-v1",
			AcceleratorClassLabel:   "test-accelerator",
		},
	}, nil, nil)
}

func TestLayoutManagerListsTransferableLayouts(t *testing.T) {
	manager := NewLayoutManager()
	manager.Observe([]scheduling.Endpoint{
		layoutTestEndpoint("prefill-1", RolePrefill, "1"),
		layoutTestEndpoint("prefill-2", RolePrefill, "1"),
		layoutTestEndpoint("decode-1", RoleDecode, "1"),
	})

	state, err := manager.State()
	require.NoError(t, err)
	assert.Equal(t, "p_2tp1_d_1tp1_m_none", state.CurrentLayout)
	assert.ElementsMatch(t, []string{
		"p_none_d_none_m_3tp1",
		"p_1tp1_d_1tp1_m_1tp1",
		"p_1tp1_d_2tp1_m_none",
		"p_2tp1_d_1tp1_m_none",
	}, state.TransferableLayouts)
}

func TestLayoutManagerRejectsIncompatibleTPWithoutChangingState(t *testing.T) {
	manager := NewLayoutManager()
	manager.Observe([]scheduling.Endpoint{
		layoutTestEndpoint("prefill", RolePrefill, "1"),
		layoutTestEndpoint("decode", RoleDecode, "1"),
	})

	_, err := manager.Apply("p_1tp2_d_1tp2_m_none")
	require.ErrorContains(t, err, "does not match")
	state, stateErr := manager.State()
	require.NoError(t, stateErr)
	assert.Equal(t, "p_1tp1_d_1tp1_m_none", state.CurrentLayout)
}

func TestLayoutManagerAcceptsHeterogeneousTPBuckets(t *testing.T) {
	manager := NewLayoutManager()
	manager.Observe([]scheduling.Endpoint{
		layoutTestEndpoint("prefill", RolePrefill, "1"),
		layoutTestEndpoint("decode", RoleDecode, "1"),
		layoutTestEndpoint("mixed", RolePrefillDecode, "2"),
	})

	state, err := manager.State()
	require.NoError(t, err)
	assert.Equal(t, "p_1tp1_d_1tp1_m_1tp2", state.CurrentLayout)
}

func TestLayoutManagerAcceptsHeterogeneousMixedOnlyLayout(t *testing.T) {
	manager := NewLayoutManager()
	manager.Observe([]scheduling.Endpoint{
		layoutTestEndpoint("mixed-1", RolePrefillDecode, "1"),
		layoutTestEndpoint("mixed-2", RolePrefillDecode, "2"),
	})

	change, err := manager.Apply("p_none_d_none_m_1tp1-1tp2")
	require.NoError(t, err)
	assert.Equal(t, "skipped", change.Status)
}

func TestLayoutManagerTransfersRolesWithinMatchingTPBucket(t *testing.T) {
	
	manager := NewLayoutManager()
	manager.Observe([]scheduling.Endpoint{
		layoutTestEndpoint("prefill-tp4-a", RolePrefill, "4"),
		layoutTestEndpoint("prefill-tp4-b", RolePrefill, "4"),
		layoutTestEndpoint("decode-tp1-a", RoleDecode, "1"),
		layoutTestEndpoint("decode-tp1-b", RoleDecode, "1"),
		layoutTestEndpoint("decode-tp1-c", RoleDecode, "1"),
		layoutTestEndpoint("decode-tp1-d", RoleDecode, "1"),
	})

	before, err := manager.State()
	require.NoError(t, err)
	assert.Equal(t, "p_2tp4_d_4tp1_m_none", before.CurrentLayout)
	assert.Contains(t, before.TransferableLayouts, "p_1tp4_d_4tp1-1tp4_m_none")

	change, err := manager.Apply("p_1tp4_d_1tp4-4tp1_m_none")
	require.NoError(t, err)
	assert.Equal(t, "ok", change.Status)
	assert.Equal(t, "p_1tp4_d_4tp1-1tp4_m_none", change.Layout)

	state, err := manager.State()
	require.NoError(t, err)
	assert.Equal(t, "p_1tp4_d_4tp1-1tp4_m_none", state.CurrentLayout)
	require.Len(t, state.CurrentInstances["prefill"], 1)
	assert.Equal(t, 4, state.CurrentInstances["prefill"][0].TensorParallelSize)
	require.Len(t, state.CurrentInstances["decode"], 5)
	decodeTPCounts := map[int]int{}
	for _, endpoint := range state.CurrentInstances["decode"] {
		decodeTPCounts[endpoint.TensorParallelSize]++
	}
	assert.Equal(t, map[int]int{1: 4, 4: 1}, decodeTPCounts)

	repeated, err := manager.Apply("p_1tp4_d_1tp4-4tp1_m_none")
	require.NoError(t, err)
	assert.Equal(t, "skipped", repeated.Status)
}

func TestRoleFilterIgnoresEndpointsWithoutTPLabelForLayoutManagement(t *testing.T) {
	manager := NewLayoutManager()
	filter := NewPrefillRole().WithLayoutManager(manager)
	endpoint := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:     k8stypes.NamespacedName{Namespace: "default", Name: "prefill"},
		Labels: map[string]string{RoleLabel: RolePrefill},
	}, nil, nil)

	filtered := filter.Filter(context.Background(), nil, []scheduling.Endpoint{endpoint})
	require.Len(t, filtered, 1)
	_, err := manager.State()
	require.ErrorContains(t, err, TensorParallelSizeLabel)
}

func TestLayoutManagerRejectsMultipleTransferGroups(t *testing.T) {
	manager := NewLayoutManager()
	manager.Observe([]scheduling.Endpoint{
		layoutTestEndpointInGroup("prefill", RolePrefill, "1", "group-a"),
		layoutTestEndpointInGroup("decode", RoleDecode, "1", "group-b"),
	})

	_, err := manager.State()
	require.ErrorContains(t, err, "exactly one role transfer group")
	_, err = manager.Apply("p_1tp1_d_1tp1_m_none")
	require.ErrorContains(t, err, "exactly one role transfer group")
}

func TestLayoutManagerRejectsInconsistentHardShapeWithinGroup(t *testing.T) {
	prefill := layoutTestEndpoint("prefill", RolePrefill, "1")
	decode := layoutTestEndpoint("decode", RoleDecode, "1")
	decode.GetMetadata().Labels[AcceleratorClassLabel] = "different-accelerator"
	manager := NewLayoutManager()
	manager.Observe([]scheduling.Endpoint{prefill, decode})

	_, err := manager.State()
	require.ErrorContains(t, err, "inconsistent model or hardware identity")
}

func TestLayoutManagerStateExposesWorkerShape(t *testing.T) {
	endpoint := layoutTestEndpoint("mixed", RolePrefillDecode, "1")
	labels := endpoint.GetMetadata().Labels
	labels[EngineProfileRefLabel] = "profile-v1"
	labels[RoleCapabilitiesLabel] = "prefill, decode"
	labels[MaxModelLenLabel] = "32768"
	labels[MaxNumBatchedTokensLabel] = "8192"
	labels[MaxNumSeqsLabel] = "64"
	manager := NewLayoutManager()
	manager.Observe([]scheduling.Endpoint{endpoint})

	state, err := manager.State()
	require.NoError(t, err)
	require.Len(t, state.CurrentInstances["mixed"], 1)
	instance := state.CurrentInstances["mixed"][0]
	assert.Equal(t, "test-group", instance.TransferGroupID)
	assert.Equal(t, "test-model-v1", instance.ModelIdentity)
	assert.Equal(t, "test-accelerator", instance.AcceleratorClass)
	assert.Equal(t, "profile-v1", instance.EngineProfileRef)
	assert.Equal(t, []string{"prefill", "decode"}, instance.RoleCapabilities)
	assert.Equal(t, 32768, instance.MaxModelLen)
	assert.Equal(t, 8192, instance.MaxNumBatchedTokens)
	assert.Equal(t, 64, instance.MaxNumSeqs)
}

func TestLayoutEndpointWorkerShapeJSONRemainsFlat(t *testing.T) {
	payload, err := json.Marshal(LayoutEndpoint{
		Name:    "default/worker",
		Address: "10.0.0.1:8000",
		WorkerShape: WorkerShape{
			TransferGroupID:    "test-group",
			TensorParallelSize: 1,
			ModelIdentity:      "test-model-v1",
			AcceleratorClass:   "test-accelerator",
		},
	})
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"name":"default/worker",
		"address":"10.0.0.1:8000",
		"transfer_group_id":"test-group",
		"tensor_parallel_size":1,
		"model_identity":"test-model-v1",
		"accelerator_class":"test-accelerator"
	}`, string(payload))
}

func TestRoleFilterIgnoresEndpointWithoutTransferGroupForLayoutManagement(t *testing.T) {
	manager := NewLayoutManager()
	filter := NewPrefillRole().WithLayoutManager(manager)
	endpoint := scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: k8stypes.NamespacedName{Namespace: "default", Name: "prefill"},
		Labels: map[string]string{
			RoleLabel:               RolePrefill,
			TensorParallelSizeLabel: "1",
		},
	}, nil, nil)

	filtered := filter.Filter(context.Background(), nil, []scheduling.Endpoint{endpoint})
	require.Len(t, filtered, 1)
	_, err := manager.State()
	require.ErrorContains(t, err, RoleTransferGroupLabel)
}
