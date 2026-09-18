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

package controller

import (
	"context"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	poolutil "github.com/llm-d/llm-d-router/pkg/epp/util/pool"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
)

var (
	inferencePool        = testutil.MakeInferencePool("test-pool1").Namespace("ns1").ObjRef()
	inferencePoolLabeled = testutil.MakeInferencePool("test-pool1").Namespace("ns1").ObjRef()
	infObjective1        = testutil.MakeInferenceObjective("model1").
				Namespace(inferencePool.Namespace).
				Priority(int32(1)).
				CreationTimestamp(metav1.Unix(1000, 0)).
				PoolName(inferencePool.Name).
				PoolGroup(routing.InferencePoolAPIGroup).ObjRef()
	infObjective1Pool2 = testutil.MakeInferenceObjective(infObjective1.Name).
				Namespace(infObjective1.Namespace).
				Priority(*infObjective1.Spec.Priority).
				CreationTimestamp(metav1.Unix(1001, 0)).
				PoolName("test-pool2").
				PoolGroup(routing.InferencePoolAPIGroup).ObjRef()
	infObjective1Critical = testutil.MakeInferenceObjective(infObjective1.Name).
				Namespace(infObjective1.Namespace).
				Priority(int32(2)).
				CreationTimestamp(metav1.Unix(1003, 0)).
				PoolName(inferencePool.Name).
				PoolGroup(routing.InferencePoolAPIGroup).ObjRef()
	infObjective1Deleted = testutil.MakeInferenceObjective(infObjective1.Name).
				Namespace(infObjective1.Namespace).
				CreationTimestamp(metav1.Unix(1004, 0)).
				DeletionTimestamp().
				PoolName(inferencePool.Name).
				PoolGroup(routing.InferencePoolAPIGroup).ObjRef()
	infObjective1DiffGroup = testutil.MakeInferenceObjective(infObjective1.Name).
				Namespace(inferencePool.Namespace).
				Priority(int32(1)).
				CreationTimestamp(metav1.Unix(1005, 0)).
				PoolName(inferencePool.Name).
				PoolGroup(v1alpha2.GroupName).ObjRef()
	infObjective2 = testutil.MakeInferenceObjective("model2").
			Namespace(inferencePool.Namespace).
			CreationTimestamp(metav1.Unix(1000, 0)).
			PoolName(inferencePool.Name).
			PoolGroup(routing.InferencePoolAPIGroup).ObjRef()
	infObjectiveShared = testutil.MakeInferenceObjective("shared").
				Namespace(inferencePool.Namespace).
				Priority(int32(1)).
				CreationTimestamp(metav1.Unix(1000, 0)).
				PoolRefs(
			v1alpha2.PoolObjectReference{Name: v1alpha2.ObjectName(inferencePool.Name), Group: v1alpha2.Group(routing.InferencePoolAPIGroup)},
			v1alpha2.PoolObjectReference{Name: "test-pool2", Group: v1alpha2.Group(routing.InferencePoolAPIGroup)},
		).ObjRef()
	infObjectiveSharedMiss = testutil.MakeInferenceObjective("shared-miss").
				Namespace(inferencePool.Namespace).
				Priority(int32(1)).
				CreationTimestamp(metav1.Unix(1000, 0)).
				PoolRefs(
			v1alpha2.PoolObjectReference{Name: "test-pool2", Group: v1alpha2.Group(routing.InferencePoolAPIGroup)},
		).ObjRef()
	infObjectiveUnion = testutil.MakeInferenceObjective("union").
				Namespace(inferencePool.Namespace).
				Priority(int32(1)).
				CreationTimestamp(metav1.Unix(1000, 0)).
				PoolName("test-pool2").
				PoolGroup(routing.InferencePoolAPIGroup).
				PoolRefs(
			v1alpha2.PoolObjectReference{Name: v1alpha2.ObjectName(inferencePool.Name), Group: v1alpha2.Group(routing.InferencePoolAPIGroup)},
		).ObjRef()
	infObjectiveNoPool = testutil.MakeInferenceObjective("no-pool").
				Namespace(inferencePool.Namespace).
				Priority(int32(1)).
				CreationTimestamp(metav1.Unix(1000, 0)).
				ObjRef()
	infObjectiveSelector = testutil.MakeInferenceObjective("selector").
				Namespace(inferencePool.Namespace).
				Priority(int32(1)).
				CreationTimestamp(metav1.Unix(1000, 0)).
				PoolSelector(&metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}).
				ObjRef()
	infObjectiveSelectorMiss = testutil.MakeInferenceObjective("selector-miss").
					Namespace(inferencePool.Namespace).
					Priority(int32(1)).
					CreationTimestamp(metav1.Unix(1000, 0)).
					PoolSelector(&metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "other"}}).
					ObjRef()
	infObjectiveSelectorEmpty = testutil.MakeInferenceObjective("selector-empty").
					Namespace(inferencePool.Namespace).
					Priority(int32(1)).
					CreationTimestamp(metav1.Unix(1000, 0)).
					PoolSelector(&metav1.LabelSelector{}).
					ObjRef()
	infObjectiveSelectorUnion = testutil.MakeInferenceObjective("selector-union").
					Namespace(inferencePool.Namespace).
					Priority(int32(1)).
					CreationTimestamp(metav1.Unix(1000, 0)).
					PoolRefs(
			v1alpha2.PoolObjectReference{Name: "test-pool2", Group: v1alpha2.Group(routing.InferencePoolAPIGroup)},
		).
		PoolSelector(&metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}).
		ObjRef()
)

func init() {
	inferencePoolLabeled.Labels = map[string]string{"tiers": "shared"}
}

func TestInferenceObjectiveReconciler(t *testing.T) {
	tests := []struct {
		name                  string
		objectivessInStore    []*v1alpha2.InferenceObjective
		objectivesInAPIServer []*v1alpha2.InferenceObjective
		poolInAPIServer       *v1.InferencePool
		objective             *v1alpha2.InferenceObjective
		incomingReq           *types.NamespacedName
		wantObjectives        []*v1alpha2.InferenceObjective
		wantResult            ctrl.Result
	}{
		{
			name:           "Empty store, add new objective",
			objective:      infObjective1,
			wantObjectives: []*v1alpha2.InferenceObjective{infObjective1},
		},
		{
			name:               "Existing objective changed pools",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			objective:          infObjective1Pool2,
			wantObjectives:     []*v1alpha2.InferenceObjective{},
		},
		{
			name:               "Not found, delete existing objective",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			incomingReq:        &types.NamespacedName{Name: infObjective1.Name, Namespace: infObjective1.Namespace},
			wantObjectives:     []*v1alpha2.InferenceObjective{},
		},
		{
			name:               "Deletion timestamp set, delete existing objective",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			objective:          infObjective1Deleted,
			wantObjectives:     []*v1alpha2.InferenceObjective{},
		},
		{
			name:               "Objective changed priority",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			objective:          infObjective1Critical,
			wantObjectives:     []*v1alpha2.InferenceObjective{infObjective1Critical},
		},
		{
			name:               "Objective not found, no matching existing objective to delete",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			incomingReq:        &types.NamespacedName{Name: "non-existent-objective", Namespace: inferencePool.Namespace},
			wantObjectives:     []*v1alpha2.InferenceObjective{infObjective1},
		},
		{
			name:               "Add to existing",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			objective:          infObjective2,
			wantObjectives:     []*v1alpha2.InferenceObjective{infObjective1, infObjective2},
		},
		{
			name:               "Objective deleted due to group mismatch for the inference inferencePool",
			objectivessInStore: []*v1alpha2.InferenceObjective{infObjective1},
			objective:          infObjective1DiffGroup,
			wantObjectives:     []*v1alpha2.InferenceObjective{},
		},
		{
			name:           "Objective ignored due to group mismatch for the inference inferencePool",
			objective:      infObjective1DiffGroup,
			wantObjectives: []*v1alpha2.InferenceObjective{},
		},
		{
			name:           "Shared objective via poolRefs includes own pool",
			objective:      infObjectiveShared,
			wantObjectives: []*v1alpha2.InferenceObjective{infObjectiveShared},
		},
		{
			name:           "Shared objective via poolRefs excludes own pool",
			objective:      infObjectiveSharedMiss,
			wantObjectives: []*v1alpha2.InferenceObjective{},
		},
		{
			name:           "poolRefs matches when poolRef points elsewhere",
			objective:      infObjectiveUnion,
			wantObjectives: []*v1alpha2.InferenceObjective{infObjectiveUnion},
		},
		{
			name:           "No pool reference set, objective ignored",
			objective:      infObjectiveNoPool,
			wantObjectives: []*v1alpha2.InferenceObjective{},
		},
		{
			name:            "Selector matches own pool labels",
			poolInAPIServer: inferencePoolLabeled,
			objective:       infObjectiveSelector,
			wantObjectives:  []*v1alpha2.InferenceObjective{infObjectiveSelector},
		},
		{
			name:            "Selector misses own pool labels",
			poolInAPIServer: inferencePoolLabeled,
			objective:       infObjectiveSelectorMiss,
			wantObjectives:  []*v1alpha2.InferenceObjective{},
		},
		{
			name:           "Empty selector matches without pool object",
			objective:      infObjectiveSelectorEmpty,
			wantObjectives: []*v1alpha2.InferenceObjective{infObjectiveSelectorEmpty},
		},
		{
			name:           "Selector with requirements and no pool object is ignored",
			objective:      infObjectiveSelector,
			wantObjectives: []*v1alpha2.InferenceObjective{},
		},
		{
			name:            "poolRefs miss with selector hit loads",
			poolInAPIServer: inferencePoolLabeled,
			objective:       infObjectiveSelectorUnion,
			wantObjectives:  []*v1alpha2.InferenceObjective{infObjectiveSelectorUnion},
		},
	}
	for _, test := range tests {
		period := time.Second
		factories := []datalayer.EndpointFactory{
			datalayer.NewTestRuntime(t, period),
		}
		for _, epf := range factories {
			t.Run(test.name, func(t *testing.T) {
				// Create a fake client with no InferenceObjective objects.
				scheme := runtime.NewScheme()
				_ = clientgoscheme.AddToScheme(scheme)
				_ = v1alpha2.Install(scheme)
				_ = v1.Install(scheme)
				initObjs := []client.Object{}
				if test.objective != nil {
					initObjs = append(initObjs, test.objective)
				}
				for _, m := range test.objectivesInAPIServer {
					initObjs = append(initObjs, m)
				}
				if test.poolInAPIServer != nil {
					initObjs = append(initObjs, test.poolInAPIServer)
				}
				fakeClient := fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(initObjs...).
					Build()
				ds := datastore.NewDatastore(t.Context(), epf)
				for _, m := range test.objectivessInStore {
					ds.ObjectiveSet(m)
				}
				endpointPool := poolutil.InferencePoolToEndpointPool(inferencePool)
				_ = ds.PoolSet(context.Background(), fakeClient, endpointPool)
				reconciler := &InferenceObjectiveReconciler{
					Reader:    fakeClient,
					Datastore: ds,
					PoolGKNN: common.GKNN{
						NamespacedName: types.NamespacedName{Name: inferencePool.Name, Namespace: inferencePool.Namespace},
						GroupKind:      schema.GroupKind{Group: inferencePool.GroupVersionKind().Group, Kind: inferencePool.GroupVersionKind().Kind},
					},
				}
				if test.incomingReq == nil {
					test.incomingReq = &types.NamespacedName{Name: test.objective.Name, Namespace: test.objective.Namespace}
				}

				// Call Reconcile.
				result, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: *test.incomingReq})
				if err != nil {
					t.Fatalf("expected no error when resource is not found, got %v", err)
				}

				if diff := cmp.Diff(result, test.wantResult); diff != "" {
					t.Errorf("Unexpected result diff (+got/-want): %s", diff)
				}

				if len(test.wantObjectives) != len(ds.ObjectiveGetAll()) {
					t.Errorf("Unexpected; want: %d, got:%d", len(test.wantObjectives), len(ds.ObjectiveGetAll()))
				}
				if diff := diffStore(ds, diffStoreParams{wantPool: endpointPool, wantObjectives: test.wantObjectives}); diff != "" {
					t.Errorf("Unexpected diff (+got/-want): %s", diff)
				}

			})
		}
	}
}

func TestInferenceObjectiveLabelChange(t *testing.T) {
	period := time.Second
	factories := []datalayer.EndpointFactory{
		datalayer.NewTestRuntime(t, period),
	}
	for _, epf := range factories {
		t.Run("unlabelled pool unloads selector objective", func(t *testing.T) {
			scheme := runtime.NewScheme()
			_ = clientgoscheme.AddToScheme(scheme)
			_ = v1alpha2.Install(scheme)
			_ = v1.Install(scheme)
			pool := testutil.MakeInferencePool("test-pool1").Namespace("ns1").ObjRef()
			pool.Labels = map[string]string{"tiers": "shared"}
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(infObjectiveSelector, pool).
				Build()
			ds := datastore.NewDatastore(t.Context(), epf)
			endpointPool := poolutil.InferencePoolToEndpointPool(inferencePool)
			_ = ds.PoolSet(context.Background(), fakeClient, endpointPool)
			reconciler := &InferenceObjectiveReconciler{
				Reader:    fakeClient,
				Datastore: ds,
				PoolGKNN: common.GKNN{
					NamespacedName: types.NamespacedName{Name: inferencePool.Name, Namespace: inferencePool.Namespace},
					GroupKind:      schema.GroupKind{Group: inferencePool.GroupVersionKind().Group, Kind: inferencePool.GroupVersionKind().Kind},
				},
			}
			req := ctrl.Request{NamespacedName: types.NamespacedName{Name: infObjectiveSelector.Name, Namespace: infObjectiveSelector.Namespace}}
			if _, err := reconciler.Reconcile(context.Background(), req); err != nil {
				t.Fatalf("expected no error, got %v", err)
			}
			if len(ds.ObjectiveGetAll()) != 1 {
				t.Fatalf("expected labeled pool to load objective, got %d", len(ds.ObjectiveGetAll()))
			}
			pool.Labels = nil
			if err := fakeClient.Update(context.Background(), pool); err != nil {
				t.Fatalf("expected no error updating pool, got %v", err)
			}
			if _, err := reconciler.Reconcile(context.Background(), req); err != nil {
				t.Fatalf("expected no error, got %v", err)
			}
			if len(ds.ObjectiveGetAll()) != 0 {
				t.Fatalf("expected unlabelled pool to unload objective, got %d", len(ds.ObjectiveGetAll()))
			}
		})
	}
}

func TestMatchesPool(t *testing.T) {
	poolName := inferencePool.Name
	poolGroup := routing.InferencePoolAPIGroup
	ref := func(name, group string) v1alpha2.PoolObjectReference {
		return v1alpha2.PoolObjectReference{Name: v1alpha2.ObjectName(name), Group: v1alpha2.Group(group)}
	}
	tests := []struct {
		name       string
		spec       v1alpha2.InferenceObjectiveSpec
		poolLabels map[string]string
		want       bool
	}{
		{
			name: "poolRef hit",
			spec: v1alpha2.InferenceObjectiveSpec{PoolRef: &v1alpha2.PoolObjectReference{Name: v1alpha2.ObjectName(poolName), Group: v1alpha2.Group(poolGroup)}},
			want: true,
		},
		{
			name: "poolRef name miss",
			spec: v1alpha2.InferenceObjectiveSpec{PoolRef: &v1alpha2.PoolObjectReference{Name: "other", Group: v1alpha2.Group(poolGroup)}},
			want: false,
		},
		{
			name: "poolRef group miss",
			spec: v1alpha2.InferenceObjectiveSpec{PoolRef: &v1alpha2.PoolObjectReference{Name: v1alpha2.ObjectName(poolName), Group: "other.example.io"}},
			want: false,
		},
		{
			name: "poolRefs hit",
			spec: v1alpha2.InferenceObjectiveSpec{PoolRefs: []v1alpha2.PoolObjectReference{ref("other", poolGroup), ref(poolName, poolGroup)}},
			want: true,
		},
		{
			name: "poolRefs miss",
			spec: v1alpha2.InferenceObjectiveSpec{PoolRefs: []v1alpha2.PoolObjectReference{ref("other", poolGroup)}},
			want: false,
		},
		{
			name:       "selector hit",
			spec:       v1alpha2.InferenceObjectiveSpec{PoolSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}},
			poolLabels: map[string]string{"tiers": "shared"},
			want:       true,
		},
		{
			name:       "selector miss",
			spec:       v1alpha2.InferenceObjectiveSpec{PoolSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}},
			poolLabels: map[string]string{"tiers": "other"},
			want:       false,
		},
		{
			name:       "selector with requirements and no labels misses",
			spec:       v1alpha2.InferenceObjectiveSpec{PoolSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}},
			poolLabels: nil,
			want:       false,
		},
		{
			name:       "empty selector matches without labels",
			spec:       v1alpha2.InferenceObjectiveSpec{PoolSelector: &metav1.LabelSelector{}},
			poolLabels: nil,
			want:       true,
		},
		{
			name:       "invalid selector fails closed",
			spec:       v1alpha2.InferenceObjectiveSpec{PoolSelector: &metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{{Key: "tiers", Operator: "Bogus", Values: []string{"shared"}}}}},
			poolLabels: map[string]string{"tiers": "shared"},
			want:       false,
		},
		{
			name:       "union of miss and hit loads",
			spec:       v1alpha2.InferenceObjectiveSpec{PoolRefs: []v1alpha2.PoolObjectReference{ref("other", poolGroup)}, PoolSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}},
			poolLabels: map[string]string{"tiers": "shared"},
			want:       true,
		},
		{
			name:       "empty spec matches nothing",
			spec:       v1alpha2.InferenceObjectiveSpec{},
			poolLabels: map[string]string{"tiers": "shared"},
			want:       false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := matchesPool(test.spec, poolName, poolGroup, test.poolLabels); got != test.want {
				t.Errorf("matchesPool() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestInferenceObjectiveEventPredicate(t *testing.T) {
	reconciler := &InferenceObjectiveReconciler{
		PoolGKNN: common.GKNN{
			NamespacedName: types.NamespacedName{Name: inferencePool.Name, Namespace: inferencePool.Namespace},
			GroupKind:      schema.GroupKind{Group: inferencePool.GroupVersionKind().Group, Kind: inferencePool.GroupVersionKind().Kind},
		},
	}
	if !reconciler.eventPredicate(infObjectiveSelector) {
		t.Error("selector objective should pass the event predicate")
	}
	if !reconciler.eventPredicate(infObjective1) {
		t.Error("poolRef objective for own pool should pass the event predicate")
	}
	if reconciler.eventPredicate(infObjective1Pool2) {
		t.Error("poolRef objective for another pool should not pass the event predicate")
	}
}

func TestInferenceObjectiveSelectorMapping(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = v1alpha2.Install(scheme)
	_ = v1.Install(scheme)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(infObjective1, infObjectiveSelector).
		Build()
	reconciler := &InferenceObjectiveReconciler{
		Reader: fakeClient,
		PoolGKNN: common.GKNN{
			NamespacedName: types.NamespacedName{Name: inferencePool.Name, Namespace: inferencePool.Namespace},
			GroupKind:      schema.GroupKind{Group: inferencePool.GroupVersionKind().Group, Kind: inferencePool.GroupVersionKind().Kind},
		},
	}
	reqs := reconciler.objectivesWithSelector(context.Background())
	if len(reqs) != 1 {
		t.Fatalf("expected 1 selector objective, got %d", len(reqs))
	}
	if reqs[0].Name != infObjectiveSelector.Name {
		t.Errorf("expected %q, got %q", infObjectiveSelector.Name, reqs[0].Name)
	}
}
