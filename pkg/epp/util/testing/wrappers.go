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

package testing

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
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

// InferencePoolWrapper wraps an group "inference.networking.k8s.io" InferencePool.
type InferencePoolWrapper struct {
	v1.InferencePool
}

// MakeInferencePool creates a wrapper for a InferencePool.
func MakeInferencePool(name string) *InferencePoolWrapper {
	return &InferencePoolWrapper{
		v1.InferencePool{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			TypeMeta: metav1.TypeMeta{
				APIVersion: routing.InferencePoolAPIGroup + "/v1",
				Kind:       "InferencePool",
			},
			Spec: v1.InferencePoolSpec{
				TargetPorts: []v1.Port{
					{Number: 8000},
				},
			},
		},
	}
}

func (m *InferencePoolWrapper) Namespace(ns string) *InferencePoolWrapper {
	m.ObjectMeta.Namespace = ns
	return m
}

func (m *InferencePoolWrapper) Selector(selector map[string]string) *InferencePoolWrapper {
	s := make(map[v1.LabelKey]v1.LabelValue)
	for k, v := range selector {
		s[v1.LabelKey(k)] = v1.LabelValue(v)
	}
	m.Spec.Selector = v1.LabelSelector{
		MatchLabels: s,
	}
	return m
}

func (m *InferencePoolWrapper) TargetPorts(p int32) *InferencePoolWrapper {
	m.Spec.TargetPorts = []v1.Port{{Number: v1.PortNumber(p)}}
	return m
}

func (m *InferencePoolWrapper) EndpointPickerRef(name string) *InferencePoolWrapper {
	m.Spec.EndpointPickerRef = v1.EndpointPickerRef{Name: v1.ObjectName(name)}
	return m
}

// Obj returns the wrapped InferencePool.
func (m *InferencePoolWrapper) ObjRef() *v1.InferencePool {
	return &m.InferencePool
}
