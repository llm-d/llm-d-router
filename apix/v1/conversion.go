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

package v1

import (
	"cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	giev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
)

// ConvertFromV1Alpha2 converts a v1alpha2 InferenceObjective to v1. The
// single pool reference becomes the sole list entry. Empty group/kind fall
// back to the CRD defaults so undefaulted objects still match. An empty
// reference name yields no list entries; the objective then targets no pool.
func ConvertFromV1Alpha2(in *v1alpha2.InferenceObjective) *InferenceObjective {
	if in == nil {
		return nil
	}
	out := &InferenceObjective{}
	out.TypeMeta = metav1.TypeMeta{APIVersion: GroupVersion.String(), Kind: "InferenceObjective"}
	out.ObjectMeta = *in.ObjectMeta.DeepCopy()
	if len(in.Status.Conditions) > 0 {
		out.Status.Conditions = append([]metav1.Condition{}, in.Status.Conditions...)
	}
	out.Spec.Priority = nil
	if in.Spec.Priority != nil {
		priority := *in.Spec.Priority
		out.Spec.Priority = &priority
	}
	if in.Spec.PoolRef.Name != "" {
		out.Spec.PoolRefs = []PoolObjectReference{
			{
				Group: Group(cmp.Or(string(in.Spec.PoolRef.Group), giev1.GroupName)),
				Kind:  Kind(cmp.Or(string(in.Spec.PoolRef.Kind), "InferencePool")),
				Name:  ObjectName(in.Spec.PoolRef.Name),
			},
		}
	}
	return out
}

// ConvertToV1Alpha2 converts a v1 InferenceObjective to v1alpha2. Only the
// first named list entry survives; additional entries and the pool selector
// have no v1alpha2 equivalent and are dropped. Kept for tests only; the
// controller never downgrades served objects.
func ConvertToV1Alpha2(in *InferenceObjective) *v1alpha2.InferenceObjective {
	if in == nil {
		return nil
	}
	out := &v1alpha2.InferenceObjective{}
	out.TypeMeta = metav1.TypeMeta{APIVersion: v1alpha2.GroupVersion.String(), Kind: "InferenceObjective"}
	out.ObjectMeta = *in.ObjectMeta.DeepCopy()
	if len(in.Status.Conditions) > 0 {
		out.Status.Conditions = append([]metav1.Condition{}, in.Status.Conditions...)
	}
	out.Spec.Priority = nil
	if in.Spec.Priority != nil {
		priority := *in.Spec.Priority
		out.Spec.Priority = &priority
	}
	for _, ref := range in.Spec.PoolRefs {
		if ref.Name == "" {
			continue
		}
		out.Spec.PoolRef = v1alpha2.PoolObjectReference{
			Group: v1alpha2.Group(cmp.Or(string(ref.Group), giev1.GroupName)),
			Kind:  v1alpha2.Kind(cmp.Or(string(ref.Kind), "InferencePool")),
			Name:  v1alpha2.ObjectName(ref.Name),
		}
		break
	}
	return out
}
