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
	"errors"
	"fmt"
	"math"
	"slices"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

const (
	// Kubernetes corev1.Probe defaults applied when a manifest omits the field.
	defaultProbePeriodSeconds    int32 = 10
	defaultProbeFailureThreshold int32 = 3
)

// DeploymentOverrides configures test-only mutations applied to Deployment
// objects before creation so E2E suites do not modify manifests shared with dev
// environments. Zero-valued fields leave the corresponding Deployment fields
// unchanged.
type DeploymentOverrides struct {
	// TerminationGracePeriodSeconds, when non-nil, sets
	// spec.template.spec.terminationGracePeriodSeconds.
	TerminationGracePeriodSeconds *int64

	// ProbePeriodSeconds, when positive, lowers periodSeconds on existing
	// readinessProbe and startupProbe definitions across containers and
	// initContainers whose effective period is larger, scaling failureThreshold
	// with ceiling division so period * failureThreshold does not shrink.
	// livenessProbe and timeoutSeconds are left unchanged.
	ProbePeriodSeconds int32
}

// ApplyDeploymentOverrides mutates obj in place when obj is a Deployment.
// Non-Deployment objects and nil inputs are left unchanged.
func ApplyDeploymentOverrides(obj *unstructured.Unstructured, overrides DeploymentOverrides) error {
	if overrides.TerminationGracePeriodSeconds != nil && *overrides.TerminationGracePeriodSeconds < 0 {
		return fmt.Errorf("terminationGracePeriodSeconds must be non-negative, got %d", *overrides.TerminationGracePeriodSeconds)
	}
	if overrides.ProbePeriodSeconds < 0 {
		return fmt.Errorf("probePeriodSeconds must be non-negative, got %d", overrides.ProbePeriodSeconds)
	}
	if overrides.TerminationGracePeriodSeconds == nil && overrides.ProbePeriodSeconds == 0 {
		return nil
	}
	return mutateDeployment(obj, func(deploy *appsv1.Deployment) bool {
		if overrides.TerminationGracePeriodSeconds != nil {
			grace := *overrides.TerminationGracePeriodSeconds
			deploy.Spec.Template.Spec.TerminationGracePeriodSeconds = &grace
		}
		if overrides.ProbePeriodSeconds > 0 {
			tightenContainerProbes(deploy.Spec.Template.Spec.Containers, overrides.ProbePeriodSeconds)
			tightenContainerProbes(deploy.Spec.Template.Spec.InitContainers, overrides.ProbePeriodSeconds)
		}
		return true
	})
}

// AppendDeploymentContainerArgs appends args (skipping exact duplicates) to
// every container or initContainer named containerName on a Deployment.
// It reports whether at least one matching container was found. Non-Deployment
// objects, nil inputs, and empty args slices return (false, nil).
func AppendDeploymentContainerArgs(obj *unstructured.Unstructured, containerName string, args ...string) (bool, error) {
	if containerName == "" {
		return false, errors.New("container name must not be empty")
	}
	if len(args) == 0 {
		return false, nil
	}
	matched := false
	err := mutateDeployment(obj, func(deploy *appsv1.Deployment) bool {
		appendArgs := func(containers []corev1.Container) {
			for i := range containers {
				if containers[i].Name != containerName {
					continue
				}
				matched = true
				for _, arg := range args {
					if !slices.Contains(containers[i].Args, arg) {
						containers[i].Args = append(containers[i].Args, arg)
					}
				}
			}
		}
		appendArgs(deploy.Spec.Template.Spec.Containers)
		appendArgs(deploy.Spec.Template.Spec.InitContainers)
		return matched
	})
	return matched, err
}

func mutateDeployment(obj *unstructured.Unstructured, fn func(*appsv1.Deployment) bool) error {
	if obj == nil || obj.GetKind() != "Deployment" {
		return nil
	}
	var deploy appsv1.Deployment
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &deploy); err != nil {
		return fmt.Errorf("decode deployment %q: %w", obj.GetName(), err)
	}
	if !fn(&deploy) {
		return nil
	}
	updated, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&deploy)
	if err != nil {
		return fmt.Errorf("encode deployment %q: %w", obj.GetName(), err)
	}
	obj.Object = updated
	return nil
}

func tightenContainerProbes(containers []corev1.Container, targetPeriod int32) {
	for i := range containers {
		tightenProbe(containers[i].ReadinessProbe, targetPeriod)
		tightenProbe(containers[i].StartupProbe, targetPeriod)
	}
}

func tightenProbe(probe *corev1.Probe, targetPeriod int32) {
	if probe == nil {
		return
	}
	oldPeriod := probe.PeriodSeconds
	if oldPeriod <= 0 {
		oldPeriod = defaultProbePeriodSeconds
	}
	if oldPeriod <= targetPeriod {
		return
	}
	oldThreshold := probe.FailureThreshold
	if oldThreshold <= 0 {
		oldThreshold = defaultProbeFailureThreshold
	}
	scaled := (int64(oldPeriod)*int64(oldThreshold) + int64(targetPeriod) - 1) / int64(targetPeriod)
	if scaled > math.MaxInt32 {
		scaled = math.MaxInt32
	}
	probe.PeriodSeconds = targetPeriod
	probe.FailureThreshold = int32(scaled)
}
