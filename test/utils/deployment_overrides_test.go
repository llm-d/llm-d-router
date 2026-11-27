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

package utils

import (
	"testing"

	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

func int64Ptr(v int64) *int64 { return &v }

func makeDeployment(t *testing.T, grace *int64, containers, initContainers []corev1.Container) *unstructured.Unstructured {
	t.Helper()
	deploy := &appsv1.Deployment{
		TypeMeta:   metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{Name: "dep"},
		Spec: appsv1.DeploymentSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					TerminationGracePeriodSeconds: grace,
					Containers:                    containers,
					InitContainers:                initContainers,
				},
			},
		},
	}
	raw, err := runtime.DefaultUnstructuredConverter.ToUnstructured(deploy)
	require.NoError(t, err)
	return &unstructured.Unstructured{Object: raw}
}

func decodeDeployment(t *testing.T, obj *unstructured.Unstructured) appsv1.Deployment {
	t.Helper()
	_ = obj.DeepCopy()
	var deploy appsv1.Deployment
	require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &deploy))
	return deploy
}

func makeServiceObj() *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "v1",
		"kind":       "Service",
		"metadata":   map[string]any{"name": "svc"},
	}}
}

func makeMalformedDeploymentObj() *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "apps/v1",
		"kind":       "Deployment",
		"metadata":   map[string]any{"name": "bad"},
		"spec":       "not-an-object",
	}}
}

func TestApplyDeploymentOverrides_NoOps(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		obj       *unstructured.Unstructured
		overrides DeploymentOverrides
	}{
		{
			name:      "nil_object",
			obj:       nil,
			overrides: DeploymentOverrides{TerminationGracePeriodSeconds: int64Ptr(1), ProbePeriodSeconds: 1},
		},
		{
			name:      "non_deployment_service",
			obj:       makeServiceObj(),
			overrides: DeploymentOverrides{TerminationGracePeriodSeconds: int64Ptr(1), ProbePeriodSeconds: 1},
		},
		{
			name: "zero_overrides",
			obj: makeDeployment(t, int64Ptr(130), []corev1.Container{{
				Name:           "app",
				ReadinessProbe: &corev1.Probe{PeriodSeconds: 10, FailureThreshold: 3},
			}}, nil),
			overrides: DeploymentOverrides{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var before *unstructured.Unstructured
			if tc.obj != nil {
				before = tc.obj.DeepCopy()
			}
			require.NoError(t, ApplyDeploymentOverrides(tc.obj, tc.overrides))
			require.Equal(t, before, tc.obj)
		})
	}
}

func TestApplyDeploymentOverrides_GracePeriod(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		initialGrace *int64
		override     int64
	}{
		{name: "overwrites_130_with_1", initialGrace: int64Ptr(130), override: 1},
		{name: "overwrites_30_with_1", initialGrace: int64Ptr(30), override: 1},
		{name: "injects_when_omitted", initialGrace: nil, override: 1},
		{name: "allows_zero", initialGrace: int64Ptr(30), override: 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			obj := makeDeployment(t, tc.initialGrace, nil, nil)
			grace := tc.override
			require.NoError(t, ApplyDeploymentOverrides(obj, DeploymentOverrides{TerminationGracePeriodSeconds: &grace}))
			grace = 999

			got := decodeDeployment(t, obj)
			require.NotNil(t, got.Spec.Template.Spec.TerminationGracePeriodSeconds)
			require.Equal(t, tc.override, *got.Spec.Template.Spec.TerminationGracePeriodSeconds)
		})
	}
}

func TestApplyDeploymentOverrides_Probes(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name              string
		targetPeriod      int32
		containers        []corev1.Container
		initContainers    []corev1.Container
		wantReadiness     *corev1.Probe
		wantStartup       *corev1.Probe
		wantLiveness      *corev1.Probe
		wantInitReadiness *corev1.Probe
	}{
		{
			name:         "scales_explicit_threshold_and_preserves_liveness_and_timeout",
			targetPeriod: 1,
			containers: []corev1.Container{{
				Name:           "envoy",
				ReadinessProbe: &corev1.Probe{PeriodSeconds: 5, FailureThreshold: 1, TimeoutSeconds: 2, SuccessThreshold: 1},
				LivenessProbe:  &corev1.Probe{PeriodSeconds: 10, FailureThreshold: 3, TimeoutSeconds: 5},
			}},
			wantReadiness: &corev1.Probe{PeriodSeconds: 1, FailureThreshold: 5, TimeoutSeconds: 2, SuccessThreshold: 1},
			wantLiveness:  &corev1.Probe{PeriodSeconds: 10, FailureThreshold: 3, TimeoutSeconds: 5},
		},
		{
			name:         "resolves_omitted_failure_threshold_default_3",
			targetPeriod: 1,
			containers: []corev1.Container{{
				Name:           "epp",
				StartupProbe:   &corev1.Probe{PeriodSeconds: 5, FailureThreshold: 60, TimeoutSeconds: 5},
				ReadinessProbe: &corev1.Probe{PeriodSeconds: 10},
				LivenessProbe:  &corev1.Probe{PeriodSeconds: 10},
			}},
			wantStartup:   &corev1.Probe{PeriodSeconds: 1, FailureThreshold: 300, TimeoutSeconds: 5},
			wantReadiness: &corev1.Probe{PeriodSeconds: 1, FailureThreshold: 30},
			wantLiveness:  &corev1.Probe{PeriodSeconds: 10},
		},
		{
			name:          "resolves_omitted_period_default_10",
			targetPeriod:  1,
			containers:    []corev1.Container{{Name: "app", ReadinessProbe: &corev1.Probe{}}},
			wantReadiness: &corev1.Probe{PeriodSeconds: 1, FailureThreshold: 30},
		},
		{
			name:         "leaves_already_fast_probe_unchanged",
			targetPeriod: 1,
			containers: []corev1.Container{{
				Name:           "agentgateway",
				StartupProbe:   &corev1.Probe{PeriodSeconds: 1, FailureThreshold: 60, TimeoutSeconds: 2},
				ReadinessProbe: &corev1.Probe{PeriodSeconds: 5, FailureThreshold: 1, TimeoutSeconds: 1},
			}},
			wantStartup:   &corev1.Probe{PeriodSeconds: 1, FailureThreshold: 60, TimeoutSeconds: 2},
			wantReadiness: &corev1.Probe{PeriodSeconds: 1, FailureThreshold: 5, TimeoutSeconds: 1},
		},
		{
			name:          "uses_ceiling_division_for_non_divisible_period",
			targetPeriod:  4,
			containers:    []corev1.Container{{Name: "app", ReadinessProbe: &corev1.Probe{PeriodSeconds: 10, FailureThreshold: 3}}},
			wantReadiness: &corev1.Probe{PeriodSeconds: 4, FailureThreshold: 8},
		},
		{
			name:              "tightens_init_container_probes",
			targetPeriod:      1,
			containers:        []corev1.Container{{Name: "vllm"}},
			initContainers:    []corev1.Container{{Name: "routing-sidecar", ReadinessProbe: &corev1.Probe{PeriodSeconds: 10, FailureThreshold: 2, TimeoutSeconds: 3}}},
			wantInitReadiness: &corev1.Probe{PeriodSeconds: 1, FailureThreshold: 20, TimeoutSeconds: 3},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			obj := makeDeployment(t, nil, tc.containers, tc.initContainers)
			require.NoError(t, ApplyDeploymentOverrides(obj, DeploymentOverrides{ProbePeriodSeconds: tc.targetPeriod}))

			got := decodeDeployment(t, obj)
			require.Equal(t, tc.wantReadiness, got.Spec.Template.Spec.Containers[0].ReadinessProbe)
			require.Equal(t, tc.wantStartup, got.Spec.Template.Spec.Containers[0].StartupProbe)
			require.Equal(t, tc.wantLiveness, got.Spec.Template.Spec.Containers[0].LivenessProbe)
			if len(tc.initContainers) > 0 {
				require.Equal(t, tc.wantInitReadiness, got.Spec.Template.Spec.InitContainers[0].ReadinessProbe)
			}
		})
	}
}

func TestApplyDeploymentOverrides_Errors(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		obj       *unstructured.Unstructured
		overrides DeploymentOverrides
	}{
		{name: "negative_grace_period", obj: makeDeployment(t, nil, nil, nil), overrides: DeploymentOverrides{TerminationGracePeriodSeconds: int64Ptr(-1)}},
		{name: "negative_probe_period", obj: makeDeployment(t, nil, nil, nil), overrides: DeploymentOverrides{ProbePeriodSeconds: -1}},
		{name: "malformed_deployment_spec", obj: makeMalformedDeploymentObj(), overrides: DeploymentOverrides{TerminationGracePeriodSeconds: int64Ptr(1)}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			require.Error(t, ApplyDeploymentOverrides(tc.obj, tc.overrides))
		})
	}
}

func TestAppendDeploymentContainerArgs(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name           string
		containers     []corev1.Container
		initContainers []corev1.Container
		target         string
		args           []string
		wantMatched    bool
		wantArgs       []string
		wantInitArgs   []string
	}{
		{
			name:        "appends_and_deduplicates_container_args",
			containers:  []corev1.Container{{Name: "epp", Args: []string{"--v", "4", "--drain-timeout=0"}}},
			target:      "epp",
			args:        []string{"--drain-timeout=0", "--extra=1"},
			wantMatched: true,
			wantArgs:    []string{"--v", "4", "--drain-timeout=0", "--extra=1"},
		},
		{
			name:           "appends_to_init_container",
			containers:     []corev1.Container{{Name: "vllm", Args: []string{"--port=8200"}}},
			initContainers: []corev1.Container{{Name: "routing-sidecar", Args: []string{"--port=8000"}}},
			target:         "routing-sidecar",
			args:           []string{"--drain-timeout=0"},
			wantMatched:    true,
			wantArgs:       []string{"--port=8200"},
			wantInitArgs:   []string{"--port=8000", "--drain-timeout=0"},
		},
		{
			name:        "returns_false_when_container_absent",
			containers:  []corev1.Container{{Name: "envoy", Args: []string{"-c", "/etc/envoy/envoy.yaml"}}},
			target:      "epp",
			args:        []string{"--drain-timeout=0"},
			wantMatched: false,
			wantArgs:    []string{"-c", "/etc/envoy/envoy.yaml"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			obj := makeDeployment(t, nil, tc.containers, tc.initContainers)
			matched, err := AppendDeploymentContainerArgs(obj, tc.target, tc.args...)
			require.NoError(t, err)
			require.Equal(t, tc.wantMatched, matched)

			got := decodeDeployment(t, obj)
			require.Equal(t, tc.wantArgs, got.Spec.Template.Spec.Containers[0].Args)
			if len(tc.initContainers) > 0 {
				require.Equal(t, tc.wantInitArgs, got.Spec.Template.Spec.InitContainers[0].Args)
			}
		})
	}
}

func TestAppendDeploymentContainerArgs_NoOps(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		obj  *unstructured.Unstructured
		args []string
	}{
		{name: "nil_object", obj: nil, args: []string{"--drain-timeout=0"}},
		{name: "non_deployment_service", obj: makeServiceObj(), args: []string{"--drain-timeout=0"}},
		{name: "empty_args", obj: makeDeployment(t, nil, []corev1.Container{{Name: "epp"}}, nil), args: nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			matched, err := AppendDeploymentContainerArgs(tc.obj, "epp", tc.args...)
			require.NoError(t, err)
			require.False(t, matched)
		})
	}
}

func TestAppendDeploymentContainerArgs_Errors(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		obj           *unstructured.Unstructured
		containerName string
	}{
		{name: "empty_container_name", obj: makeDeployment(t, nil, nil, nil), containerName: ""},
		{name: "malformed_deployment_spec", obj: makeMalformedDeploymentObj(), containerName: "epp"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			matched, err := AppendDeploymentContainerArgs(tc.obj, tc.containerName, "--drain-timeout=0")
			require.Error(t, err)
			require.False(t, matched)
		})
	}
}
