/*
Copyright 2025 The Kubernetes Authors.
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

// Package apix provides builders for llm-d apix inference CRDs used in tests.
package apix

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	apixv1 "github.com/llm-d/llm-d-router/apix/v1"
	"github.com/llm-d/llm-d-router/apix/v1alpha2"
)

// InferenceObjectiveWrapper wraps an InferenceObjective.
type InferenceObjectiveWrapper struct {
	v1alpha2.InferenceObjective
}

// MakeInferenceObjective creates a wrapper for a InferenceObjective.
func MakeInferenceObjective(name string) *InferenceObjectiveWrapper {
	return &InferenceObjectiveWrapper{
		v1alpha2.InferenceObjective{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1alpha2.InferenceObjectiveSpec{},
		},
	}
}

func (m *InferenceObjectiveWrapper) Namespace(ns string) *InferenceObjectiveWrapper {
	m.ObjectMeta.Namespace = ns
	return m
}

// Obj returns the wrapped InferenceObjective.
func (m *InferenceObjectiveWrapper) ObjRef() *v1alpha2.InferenceObjective {
	return &m.InferenceObjective
}

func (m *InferenceObjectiveWrapper) PoolName(poolName string) *InferenceObjectiveWrapper {
	m.Spec.PoolRef.Name = v1alpha2.ObjectName(poolName)
	return m
}

func (m *InferenceObjectiveWrapper) PoolGroup(poolGroup string) *InferenceObjectiveWrapper {
	m.Spec.PoolRef.Group = v1alpha2.Group(poolGroup)
	return m
}

func (m *InferenceObjectiveWrapper) Priority(priority int32) *InferenceObjectiveWrapper {
	m.Spec.Priority = &priority
	return m
}

func (m *InferenceObjectiveWrapper) DeletionTimestamp() *InferenceObjectiveWrapper {
	now := metav1.Now()
	m.ObjectMeta.DeletionTimestamp = &now
	m.Finalizers = []string{"finalizer"}
	return m
}

func (m *InferenceObjectiveWrapper) CreationTimestamp(t metav1.Time) *InferenceObjectiveWrapper {
	m.ObjectMeta.CreationTimestamp = t
	return m
}

// V1InferenceObjectiveWrapper wraps a v1 InferenceObjective for tests that
// exercise datastore-direct behavior. Cluster-shape fixtures stay on the
// v1alpha2 wrapper above and convert at the reconciler edge.
type V1InferenceObjectiveWrapper struct {
	apixv1.InferenceObjective
}

// MakeV1InferenceObjective creates a wrapper for a v1 InferenceObjective.
func MakeV1InferenceObjective(name string) *V1InferenceObjectiveWrapper {
	return &V1InferenceObjectiveWrapper{
		apixv1.InferenceObjective{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: apixv1.InferenceObjectiveSpec{},
		},
	}
}

func (m *V1InferenceObjectiveWrapper) Namespace(ns string) *V1InferenceObjectiveWrapper {
	m.ObjectMeta.Namespace = ns
	return m
}

// Obj returns the wrapped v1 InferenceObjective.
func (m *V1InferenceObjectiveWrapper) ObjRef() *apixv1.InferenceObjective {
	return &m.InferenceObjective
}

func (m *V1InferenceObjectiveWrapper) Priority(priority int32) *V1InferenceObjectiveWrapper {
	m.Spec.Priority = &priority
	return m
}

func (m *V1InferenceObjectiveWrapper) PoolRefs(refs ...apixv1.PoolObjectReference) *V1InferenceObjectiveWrapper {
	m.Spec.PoolRefs = refs
	return m
}

func (m *V1InferenceObjectiveWrapper) PoolSelector(sel *metav1.LabelSelector) *V1InferenceObjectiveWrapper {
	m.Spec.PoolSelector = sel
	return m
}

func (m *V1InferenceObjectiveWrapper) CreationTimestamp(t metav1.Time) *V1InferenceObjectiveWrapper {
	m.ObjectMeta.CreationTimestamp = t
	return m
}
