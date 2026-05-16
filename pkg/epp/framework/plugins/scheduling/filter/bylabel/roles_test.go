package bylabel

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

func TestRoleFilterDecodeRole(t *testing.T) {
	endpoints := []scheduling.Endpoint{
		createEndpoint(k8stypes.NamespacedName{Name: "decode-pod"}, "10.0.0.1",
			map[string]string{RoleLabel: RoleDecode}),
		createEndpoint(k8stypes.NamespacedName{Name: "prefill-pod"}, "10.0.0.2",
			map[string]string{RoleLabel: RolePrefill}),
		createEndpoint(k8stypes.NamespacedName{Name: "pd-pod"}, "10.0.0.3",
			map[string]string{RoleLabel: RolePrefillDecode}),
		createEndpoint(k8stypes.NamespacedName{Name: "both-pod"}, "10.0.0.4",
			map[string]string{RoleLabel: RoleBoth}),
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

	filtered := rf.Filter(ctx, nil, nil, endpoints)

	names := make([]string, len(filtered))
	for i, ep := range filtered {
		names[i] = ep.GetMetadata().NamespacedName.Name
	}

	assert.ElementsMatch(t, []string{
		"decode-pod", "pd-pod", "both-pod", "epd-pod",
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
		createEndpoint(k8stypes.NamespacedName{Name: "both-pod"}, "10.0.0.4",
			map[string]string{RoleLabel: RoleBoth}),
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

	filtered := rf.Filter(ctx, nil, nil, endpoints)

	names := make([]string, len(filtered))
	for i, ep := range filtered {
		names[i] = ep.GetMetadata().NamespacedName.Name
	}

	assert.ElementsMatch(t, []string{
		"prefill-pod", "pd-pod", "both-pod", "epd-pod", "ep-pod",
	}, names)
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

	filtered := rf.Filter(ctx, nil, nil, endpoints)

	names := make([]string, len(filtered))
	for i, ep := range filtered {
		names[i] = ep.GetMetadata().NamespacedName.Name
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
			name:         "DecodeRoleFactory returns ByLabel",
			roleName:     DecodeRoleType,
			expectedName: "test-decode",
		},
		{
			name:         "PrefillRoleFactory returns ByLabel",
			roleName:     PrefillRoleType,
			expectedName: "test-prefill",
		},
		{
			name:         "EncodeRoleFactory returns ByLabel",
			roleName:     EncodeRoleType,
			expectedName: "test-encode",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rf *ByLabel
			switch tt.roleName {
			case DecodeRoleType:
				p, err := DecodeRoleFactory("test-decode", nil, nil)
				require.NoError(t, err)
				var ok bool
				rf, ok = p.(*ByLabel)
				require.True(t, ok, "factory should return *ByLabel")
			case PrefillRoleType:
				p, err := PrefillRoleFactory("test-prefill", nil, nil)
				require.NoError(t, err)
				var ok bool
				rf, ok = p.(*ByLabel)
				require.True(t, ok, "factory should return *ByLabel")
			case EncodeRoleType:
				p, err := EncodeRoleFactory("test-encode", nil, nil)
				require.NoError(t, err)
				var ok bool
				rf, ok = p.(*ByLabel)
				require.True(t, ok, "factory should return *ByLabel")
			}

			assert.Equal(t, ByLabelType, rf.TypedName().Type)
			assert.Equal(t, tt.expectedName, rf.TypedName().Name)
		})
	}
}

func TestRoleFilterWithName(t *testing.T) {
	rf := NewDecodeRole()
	assert.Equal(t, DecodeRoleType, rf.TypedName().Name)
	assert.Equal(t, ByLabelType, rf.TypedName().Type)

	rf.WithName("my-custom-name")
	assert.Equal(t, "my-custom-name", rf.TypedName().Name)
	assert.Equal(t, ByLabelType, rf.TypedName().Type)
}

func TestRoleFilterEmptyEndpoints(t *testing.T) {
	ctx := utils.NewTestContext(t)
	rf := NewDecodeRole()

	result := rf.Filter(ctx, nil, nil, []scheduling.Endpoint{})
	assert.Empty(t, result)

	result = rf.Filter(ctx, nil, nil, nil)
	assert.Empty(t, result)
}
