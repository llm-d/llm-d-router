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
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
)

func TestConvertFromV1Alpha2(t *testing.T) {
	priority := int32(10)
	tests := []struct {
		name string
		in   *v1alpha2.InferenceObjective
		want *InferenceObjective
	}{
		{
			name: "nil converts to nil",
			in:   nil,
			want: nil,
		},
		{
			name: "single reference becomes the list",
			in: &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{Name: "tier", Namespace: "ns"},
				Spec: v1alpha2.InferenceObjectiveSpec{
					Priority: &priority,
					PoolRef:  v1alpha2.PoolObjectReference{Name: "pool1", Group: "inference.networking.k8s.io"},
				},
			},
			want: &InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{Name: "tier", Namespace: "ns"},
				Spec: InferenceObjectiveSpec{
					Priority: &priority,
					PoolRefs: []PoolObjectReference{{Name: "pool1", Group: "inference.networking.k8s.io"}},
				},
			},
		},
		{
			name: "nil priority stays nil",
			in: &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{Name: "tier", Namespace: "ns"},
				Spec: v1alpha2.InferenceObjectiveSpec{
					PoolRef: v1alpha2.PoolObjectReference{Name: "pool1", Group: "inference.networking.k8s.io"},
				},
			},
			want: &InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{Name: "tier", Namespace: "ns"},
				Spec: InferenceObjectiveSpec{
					PoolRefs: []PoolObjectReference{{Name: "pool1", Group: "inference.networking.k8s.io"}},
				},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if diff := cmp.Diff(test.want, ConvertFromV1Alpha2(test.in)); diff != "" {
				t.Errorf("ConvertFromV1Alpha2() diff (-want/+got): %s", diff)
			}
		})
	}
}

func TestConvertToV1Alpha2(t *testing.T) {
	priority := int32(10)
	tests := []struct {
		name string
		in   *InferenceObjective
		want *v1alpha2.InferenceObjective
	}{
		{
			name: "nil converts to nil",
			in:   nil,
			want: nil,
		},
		{
			name: "empty list yields zero single reference without inventing data",
			in: &InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{Name: "tier", Namespace: "ns"},
				Spec:       InferenceObjectiveSpec{},
			},
			want: &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{Name: "tier", Namespace: "ns"},
				Spec:       v1alpha2.InferenceObjectiveSpec{},
			},
		},
		{
			name: "first list entry populates the single reference, rest and selector dropped",
			in: &InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{Name: "tier", Namespace: "ns"},
				Spec: InferenceObjectiveSpec{
					Priority: &priority,
					PoolRefs: []PoolObjectReference{
						{Name: "pool1", Group: "inference.networking.k8s.io"},
						{Name: "pool2", Group: "inference.networking.k8s.io"},
					},
					PoolSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}},
				},
			},
			want: &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{Name: "tier", Namespace: "ns"},
				Spec: v1alpha2.InferenceObjectiveSpec{
					Priority: &priority,
					PoolRef:  v1alpha2.PoolObjectReference{Name: "pool1", Group: "inference.networking.k8s.io"},
				},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if diff := cmp.Diff(test.want, ConvertToV1Alpha2(test.in)); diff != "" {
				t.Errorf("ConvertToV1Alpha2() diff (-want/+got): %s", diff)
			}
		})
	}
}

func TestConversionDoesNotAliasInput(t *testing.T) {
	const mutated = "mutated"
	priority := int32(10)
	in := &v1alpha2.InferenceObjective{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "tier",
			Namespace:   "ns",
			Labels:      map[string]string{"a": "b"},
			Annotations: map[string]string{"c": "d"},
		},
		Spec: v1alpha2.InferenceObjectiveSpec{
			Priority: &priority,
			PoolRef:  v1alpha2.PoolObjectReference{Name: "pool1", Group: "inference.networking.k8s.io"},
		},
		Status: v1alpha2.InferenceObjectiveStatus{
			Conditions: []metav1.Condition{{Type: "Accepted", Status: metav1.ConditionTrue}},
		},
	}
	out := ConvertFromV1Alpha2(in)
	out.Labels["a"] = mutated
	out.Annotations["c"] = mutated
	*out.Spec.Priority = 99
	out.Spec.PoolRefs[0].Name = ObjectName(mutated)
	out.Status.Conditions[0].Status = metav1.ConditionFalse
	if in.Labels["a"] != "b" || in.Annotations["c"] != "d" {
		t.Error("conversion aliases input metadata maps")
	}
	if *in.Spec.Priority != 10 {
		t.Error("conversion aliases input priority")
	}
	if in.Spec.PoolRef.Name != "pool1" {
		t.Error("conversion aliases input pool reference")
	}
	if in.Status.Conditions[0].Status != metav1.ConditionTrue {
		t.Error("conversion aliases input status")
	}
}

func TestConvertToDoesNotAliasInput(t *testing.T) {
	const mutated = "mutated"
	priority := int32(10)
	in := &InferenceObjective{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "tier",
			Namespace:   "ns",
			Labels:      map[string]string{"a": "b"},
			Annotations: map[string]string{"c": "d"},
		},
		Spec: InferenceObjectiveSpec{
			Priority: &priority,
			PoolRefs: []PoolObjectReference{{Name: "pool1", Group: "inference.networking.k8s.io"}},
		},
		Status: InferenceObjectiveStatus{
			Conditions: []metav1.Condition{{Type: "Accepted", Status: metav1.ConditionTrue}},
		},
	}
	out := ConvertToV1Alpha2(in)
	out.Labels["a"] = mutated
	out.Annotations["c"] = mutated
	*out.Spec.Priority = 99
	out.Spec.PoolRef.Name = v1alpha2.ObjectName(mutated)
	out.Status.Conditions[0].Status = metav1.ConditionFalse
	if in.Labels["a"] != "b" || in.Annotations["c"] != "d" {
		t.Error("conversion aliases input metadata maps")
	}
	if *in.Spec.Priority != 10 {
		t.Error("conversion aliases input priority")
	}
	if in.Spec.PoolRefs[0].Name != "pool1" {
		t.Error("conversion aliases input pool references")
	}
	if in.Status.Conditions[0].Status != metav1.ConditionTrue {
		t.Error("conversion aliases input status")
	}
}

func TestConversionRoundTrip(t *testing.T) {
	priority := int32(10)
	original := &InferenceObjective{
		ObjectMeta: metav1.ObjectMeta{Name: "tier", Namespace: "ns"},
		Spec: InferenceObjectiveSpec{
			Priority: &priority,
			PoolRefs: []PoolObjectReference{
				{Name: "pool1", Group: "inference.networking.k8s.io"},
			},
		},
	}
	if diff := cmp.Diff(original, ConvertFromV1Alpha2(ConvertToV1Alpha2(original))); diff != "" {
		t.Errorf("round trip diff (-want/+got): %s", diff)
	}
}
