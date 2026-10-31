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

package apix

import (
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

const (
	objName  = "objective-1"
	poolName = "pool-1"
	testNS   = "ns-1"
)

func TestInferenceObjectiveWrapper(t *testing.T) {
	tests := []struct {
		name  string
		build func() *v1alpha2.InferenceObjective
		check func(t *testing.T, obj *v1alpha2.InferenceObjective)
	}{
		{
			// MakeInferenceObjective does no defaulting: the CRD defaults for
			// PoolRef.Group/Kind are applied by the apiserver, not the builder.
			name:  "MakeInferenceObjective sets only the name",
			build: func() *v1alpha2.InferenceObjective { return MakeInferenceObjective(objName).ObjRef() },
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				if obj.Name != objName {
					t.Errorf("Name = %q, want %q", obj.Name, objName)
				}
				if obj.Namespace != "" {
					t.Errorf("Namespace = %q, want empty", obj.Namespace)
				}
				if obj.Spec.PoolRef.Name != "" || obj.Spec.PoolRef.Group != "" || obj.Spec.PoolRef.Kind != "" {
					t.Errorf("Spec.PoolRef = %+v, want zero value", obj.Spec.PoolRef)
				}
				if obj.Spec.Priority != nil {
					t.Errorf("Spec.Priority = %d, want nil", *obj.Spec.Priority)
				}
				if obj.DeletionTimestamp != nil {
					t.Errorf("DeletionTimestamp = %v, want nil", obj.DeletionTimestamp)
				}
				if !obj.CreationTimestamp.IsZero() {
					t.Errorf("CreationTimestamp = %v, want zero", obj.CreationTimestamp)
				}
			},
		},
		{
			name:  "empty name is preserved",
			build: func() *v1alpha2.InferenceObjective { return MakeInferenceObjective("").ObjRef() },
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				if obj.Name != "" {
					t.Errorf("Name = %q, want empty", obj.Name)
				}
			},
		},
		{
			name:  "Namespace sets the namespace",
			build: func() *v1alpha2.InferenceObjective { return MakeInferenceObjective(objName).Namespace(testNS).ObjRef() },
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				if obj.Namespace != testNS {
					t.Errorf("Namespace = %q, want %q", obj.Namespace, testNS)
				}
			},
		},
		{
			name: "PoolName sets the pool reference name only",
			build: func() *v1alpha2.InferenceObjective {
				return MakeInferenceObjective(objName).PoolName(poolName).ObjRef()
			},
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				if obj.Spec.PoolRef.Name != v1alpha2.ObjectName(poolName) {
					t.Errorf("Spec.PoolRef.Name = %q, want %q", obj.Spec.PoolRef.Name, poolName)
				}
				if obj.Spec.PoolRef.Group != "" {
					t.Errorf("Spec.PoolRef.Group = %q, want empty", obj.Spec.PoolRef.Group)
				}
			},
		},
		{
			name: "PoolGroup sets the pool reference group only",
			build: func() *v1alpha2.InferenceObjective {
				return MakeInferenceObjective(objName).PoolGroup(routing.InferencePoolAPIGroup).ObjRef()
			},
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				if obj.Spec.PoolRef.Group != v1alpha2.Group(routing.InferencePoolAPIGroup) {
					t.Errorf("Spec.PoolRef.Group = %q, want %q", obj.Spec.PoolRef.Group, routing.InferencePoolAPIGroup)
				}
				if obj.Spec.PoolRef.Name != "" {
					t.Errorf("Spec.PoolRef.Name = %q, want empty", obj.Spec.PoolRef.Name)
				}
			},
		},
		{
			// Priority takes the address of its parameter, so each call must own a
			// distinct value rather than aliasing a shared one.
			name: "Priority stores the last value set",
			build: func() *v1alpha2.InferenceObjective {
				return MakeInferenceObjective(objName).Priority(1).Priority(-10).ObjRef()
			},
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				if obj.Spec.Priority == nil {
					t.Fatal("Spec.Priority is nil, want a value")
				}
				if *obj.Spec.Priority != -10 {
					t.Errorf("Spec.Priority = %d, want %d", *obj.Spec.Priority, -10)
				}
			},
		},
		{
			// Zero is a meaningful priority and must be distinguishable from unset.
			name:  "Priority zero is set, not left nil",
			build: func() *v1alpha2.InferenceObjective { return MakeInferenceObjective(objName).Priority(0).ObjRef() },
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				if obj.Spec.Priority == nil {
					t.Fatal("Spec.Priority is nil, want a pointer to 0")
				}
				if *obj.Spec.Priority != 0 {
					t.Errorf("Spec.Priority = %d, want 0", *obj.Spec.Priority)
				}
			},
		},
		{
			// A deletion timestamp without a finalizer is dropped by the apiserver, so
			// the builder must set both for the object to read as terminating.
			name: "DeletionTimestamp marks the objective terminating",
			build: func() *v1alpha2.InferenceObjective {
				return MakeInferenceObjective(objName).DeletionTimestamp().ObjRef()
			},
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				if obj.DeletionTimestamp == nil {
					t.Fatal("DeletionTimestamp is nil, want a timestamp")
				}
				if obj.DeletionTimestamp.IsZero() {
					t.Error("DeletionTimestamp is the zero time, want a real time")
				}
				if len(obj.Finalizers) == 0 {
					t.Error("Finalizers is empty, want a finalizer so the timestamp is retained")
				}
			},
		},
		{
			name: "CreationTimestamp sets the supplied time",
			build: func() *v1alpha2.InferenceObjective {
				ts := metav1.NewTime(time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC))
				return MakeInferenceObjective(objName).CreationTimestamp(ts).ObjRef()
			},
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				want := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
				if !obj.CreationTimestamp.Time.Equal(want) {
					t.Errorf("CreationTimestamp = %v, want %v", obj.CreationTimestamp.Time, want)
				}
			},
		},
		{
			name: "CreationTimestamp accepts the zero time",
			build: func() *v1alpha2.InferenceObjective {
				return MakeInferenceObjective(objName).CreationTimestamp(metav1.Time{}).ObjRef()
			},
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				if !obj.CreationTimestamp.IsZero() {
					t.Errorf("CreationTimestamp = %v, want zero", obj.CreationTimestamp)
				}
			},
		},
		{
			name: "full fluent chain applies every setter",
			build: func() *v1alpha2.InferenceObjective {
				return MakeInferenceObjective(objName).
					Namespace(testNS).
					PoolName(poolName).
					PoolGroup(routing.InferencePoolAPIGroup).
					Priority(5).
					DeletionTimestamp().
					CreationTimestamp(metav1.NewTime(time.Date(2026, time.March, 1, 0, 0, 0, 0, time.UTC))).
					ObjRef()
			},
			check: func(t *testing.T, obj *v1alpha2.InferenceObjective) {
				if obj.Name != objName || obj.Namespace != testNS {
					t.Errorf("Name/Namespace = %q/%q, want %q/%q", obj.Name, obj.Namespace, objName, testNS)
				}
				if obj.Spec.PoolRef.Name != v1alpha2.ObjectName(poolName) ||
					obj.Spec.PoolRef.Group != v1alpha2.Group(routing.InferencePoolAPIGroup) {
					t.Errorf("Spec.PoolRef = %+v, want name %q group %q", obj.Spec.PoolRef, poolName, routing.InferencePoolAPIGroup)
				}
				if obj.Spec.Priority == nil || *obj.Spec.Priority != 5 {
					t.Errorf("Spec.Priority = %v, want 5", obj.Spec.Priority)
				}
				if obj.DeletionTimestamp == nil {
					t.Error("DeletionTimestamp is nil, want a timestamp")
				}
				if obj.CreationTimestamp.IsZero() {
					t.Error("CreationTimestamp is zero, want the supplied time")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.check(t, tt.build())
		})
	}
}

// ObjRef must hand back the wrapper's own object, not a copy: callers pass the
// result to fake clients and expect later builder calls to be visible.
func TestObjRefAliasesTheWrappedObject(t *testing.T) {
	t.Run("InferenceObjectiveWrapper", func(t *testing.T) {
		w := MakeInferenceObjective(objName)
		ref := w.ObjRef()

		if ref != &w.InferenceObjective {
			t.Fatalf("ObjRef() = %p, want &wrapper.InferenceObjective (%p)", ref, &w.InferenceObjective)
		}

		w.Namespace(testNS)
		if ref.Namespace != testNS {
			t.Errorf("Namespace via earlier ObjRef() = %q, want %q", ref.Namespace, testNS)
		}
	})
}
