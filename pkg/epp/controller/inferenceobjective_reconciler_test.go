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
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	apixv1 "github.com/llm-d/llm-d-router/apix/v1"
	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	poolutil "github.com/llm-d/llm-d-router/pkg/epp/util/pool"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
)

var (
	inferencePool = testutil.MakeInferencePool("test-pool1").Namespace("ns1").ObjRef()
	infObjective1 = testutil.MakeInferenceObjective("model1").
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
	v1ObjectiveShared = testutil.MakeV1InferenceObjective("shared").
				Namespace(inferencePool.Namespace).
				Priority(int32(1)).
				CreationTimestamp(metav1.Unix(1000, 0)).
				PoolRefs(
			apixv1.PoolObjectReference{Name: apixv1.ObjectName(inferencePool.Name), Group: apixv1.Group(routing.InferencePoolAPIGroup)},
			apixv1.PoolObjectReference{Name: "test-pool2", Group: apixv1.Group(routing.InferencePoolAPIGroup)},
		).ObjRef()
	v1ObjectiveSelector = testutil.MakeV1InferenceObjective("selector").
				Namespace(inferencePool.Namespace).
				Priority(int32(1)).
				CreationTimestamp(metav1.Unix(1000, 0)).
				PoolSelector(&metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}).
				ObjRef()
	v1ObjectiveSelectorMiss = testutil.MakeV1InferenceObjective("selector-miss").
				Namespace(inferencePool.Namespace).
				Priority(int32(1)).
				CreationTimestamp(metav1.Unix(1000, 0)).
				PoolSelector(&metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "other"}}).
				ObjRef()
	v1ObjectiveSelectorEmpty = testutil.MakeV1InferenceObjective("selector-empty").
					Namespace(inferencePool.Namespace).
					Priority(int32(1)).
					CreationTimestamp(metav1.Unix(1000, 0)).
					PoolSelector(&metav1.LabelSelector{}).
					ObjRef()
)

// toV1 converts v1alpha2 fixtures to the datastore-normalized v1 shape.
func toV1(objs ...*v1alpha2.InferenceObjective) []*apixv1.InferenceObjective {
	out := make([]*apixv1.InferenceObjective, 0, len(objs))
	for _, obj := range objs {
		out = append(out, apixv1.ConvertFromV1Alpha2(obj))
	}
	return out
}

// defaultedV1 converts a fixture without priority the way the reconciler
// stores it: the API default of 0 applied.
func defaultedV1(obj *v1alpha2.InferenceObjective) *apixv1.InferenceObjective {
	converted := apixv1.ConvertFromV1Alpha2(obj)
	priority := int32(0)
	converted.Spec.Priority = &priority
	return converted
}

func infObjective2DefaultedV1() *apixv1.InferenceObjective {
	return defaultedV1(infObjective2)
}

func labeledPool() *v1.InferencePool {
	pool := testutil.MakeInferencePool("test-pool1").Namespace("ns1").ObjRef()
	pool.Labels = map[string]string{"tiers": "shared"}
	return pool
}

func TestInferenceObjectiveReconciler(t *testing.T) {
	tests := []struct {
		name                  string
		objectivessInStore    []*apixv1.InferenceObjective
		objectivesInAPIServer []client.Object
		poolInAPIServer       *v1.InferencePool
		primaryV1             bool
		objectiveV1Alpha2     *v1alpha2.InferenceObjective
		objectiveV1           *apixv1.InferenceObjective
		incomingReq           *types.NamespacedName
		wantObjectives        []*apixv1.InferenceObjective
		wantResult            ctrl.Result
	}{
		{
			name:              "Empty store, add new objective",
			objectiveV1Alpha2: infObjective1,
			wantObjectives:    toV1(infObjective1),
		},
		{
			name:               "Existing objective changed pools",
			objectivessInStore: toV1(infObjective1),
			objectiveV1Alpha2:  infObjective1Pool2,
			wantObjectives:     []*apixv1.InferenceObjective{},
		},
		{
			name:               "Not found, delete existing objective",
			objectivessInStore: toV1(infObjective1),
			incomingReq:        &types.NamespacedName{Name: infObjective1.Name, Namespace: infObjective1.Namespace},
			wantObjectives:     []*apixv1.InferenceObjective{},
		},
		{
			name:               "Deletion timestamp set, delete existing objective",
			objectivessInStore: toV1(infObjective1),
			objectiveV1Alpha2:  infObjective1Deleted,
			wantObjectives:     []*apixv1.InferenceObjective{},
		},
		{
			name:               "Objective changed priority",
			objectivessInStore: toV1(infObjective1),
			objectiveV1Alpha2:  infObjective1Critical,
			wantObjectives:     toV1(infObjective1Critical),
		},
		{
			name:               "Objective not found, no matching existing objective to delete",
			objectivessInStore: toV1(infObjective1),
			incomingReq:        &types.NamespacedName{Name: "non-existent-objective", Namespace: inferencePool.Namespace},
			wantObjectives:     toV1(infObjective1),
		},
		{
			name:               "Add to existing",
			objectivessInStore: toV1(infObjective1),
			objectiveV1Alpha2:  infObjective2,
			wantObjectives:     []*apixv1.InferenceObjective{toV1(infObjective1)[0], infObjective2DefaultedV1()},
		},
		{
			name:              "Objective without priority is stored with priority 0",
			objectiveV1Alpha2: infObjective2,
			wantObjectives:    []*apixv1.InferenceObjective{infObjective2DefaultedV1()},
		},
		{
			name:               "Objective deleted due to group mismatch for the inference inferencePool",
			objectivessInStore: toV1(infObjective1),
			objectiveV1Alpha2:  infObjective1DiffGroup,
			wantObjectives:     []*apixv1.InferenceObjective{},
		},
		{
			name:              "Objective ignored due to group mismatch for the inference inferencePool",
			objectiveV1Alpha2: infObjective1DiffGroup,
			wantObjectives:    []*apixv1.InferenceObjective{},
		},
		{
			name:           "v1 list hit loads without pool object",
			primaryV1:      true,
			objectiveV1:    v1ObjectiveShared,
			wantObjectives: []*apixv1.InferenceObjective{v1ObjectiveShared},
		},
		{
			name:            "v1 selector hit loads with labeled pool",
			primaryV1:       true,
			poolInAPIServer: labeledPool(),
			objectiveV1:     v1ObjectiveSelector,
			wantObjectives:  []*apixv1.InferenceObjective{v1ObjectiveSelector},
		},
		{
			name:            "v1 selector miss is ignored",
			primaryV1:       true,
			poolInAPIServer: labeledPool(),
			objectiveV1:     v1ObjectiveSelectorMiss,
			wantObjectives:  []*apixv1.InferenceObjective{},
		},
		{
			name:           "v1 empty selector matches without pool object",
			primaryV1:      true,
			objectiveV1:    v1ObjectiveSelectorEmpty,
			wantObjectives: []*apixv1.InferenceObjective{v1ObjectiveSelectorEmpty},
		},
		{
			name:           "v1 selector with requirements and no pool object is ignored",
			primaryV1:      true,
			objectiveV1:    v1ObjectiveSelector,
			wantObjectives: []*apixv1.InferenceObjective{},
		},
		{
			name:              "v1alpha2 namesake ignored when v1 primary",
			primaryV1:         true,
			objectiveV1Alpha2: infObjective1,
			wantObjectives:    []*apixv1.InferenceObjective{},
		},
		{
			name:           "v1 object ignored when v1alpha2 primary",
			objectiveV1:    v1ObjectiveShared,
			wantObjectives: []*apixv1.InferenceObjective{},
		},
		{
			name:              "v1alpha2 converts at the edge",
			poolInAPIServer:   labeledPool(),
			objectiveV1Alpha2: infObjective1,
			wantObjectives:    toV1(infObjective1),
		},
	}
	for _, test := range tests {
		period := time.Second
		factories := []datalayer.EndpointFactory{
			datalayer.NewTestRuntime(t, period),
		}
		for _, epf := range factories {
			t.Run(test.name, func(t *testing.T) {
				scheme := runtime.NewScheme()
				_ = clientgoscheme.AddToScheme(scheme)
				_ = v1alpha2.Install(scheme)
				_ = apixv1.Install(scheme)
				_ = v1.Install(scheme)
				initObjs := []client.Object{}
				initObjs = append(initObjs, test.objectivesInAPIServer...)
				if test.objectiveV1Alpha2 != nil {
					initObjs = append(initObjs, test.objectiveV1Alpha2)
				}
				if test.objectiveV1 != nil {
					initObjs = append(initObjs, test.objectiveV1)
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
					PrimaryV1: test.primaryV1,
				}
				if test.incomingReq == nil {
					name, namespace := "", ""
					switch {
					case test.objectiveV1 != nil:
						name, namespace = test.objectiveV1.Name, test.objectiveV1.Namespace
					case test.objectiveV1Alpha2 != nil:
						name, namespace = test.objectiveV1Alpha2.Name, test.objectiveV1Alpha2.Namespace
					}
					test.incomingReq = &types.NamespacedName{Name: name, Namespace: namespace}
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
			_ = apixv1.Install(scheme)
			_ = v1.Install(scheme)
			pool := labeledPool()
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(v1ObjectiveSelector, pool).
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
				PrimaryV1: true,
			}
			req := ctrl.Request{NamespacedName: types.NamespacedName{Name: v1ObjectiveSelector.Name, Namespace: v1ObjectiveSelector.Namespace}}
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

func TestInferenceObjectiveSelectorMapping(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = v1alpha2.Install(scheme)
	_ = apixv1.Install(scheme)
	_ = v1.Install(scheme)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(infObjective1, v1ObjectiveSelector).
		Build()
	reconciler := &InferenceObjectiveReconciler{
		Reader: fakeClient,
		PoolGKNN: common.GKNN{
			NamespacedName: types.NamespacedName{Name: inferencePool.Name, Namespace: inferencePool.Namespace},
			GroupKind:      schema.GroupKind{Group: inferencePool.GroupVersionKind().Group, Kind: inferencePool.GroupVersionKind().Kind},
		},
		PrimaryV1: true,
	}
	reqs := reconciler.objectivesWithSelector(context.Background())
	if len(reqs) != 1 {
		t.Fatalf("expected 1 selector objective, got %d", len(reqs))
	}
	if reqs[0].Name != v1ObjectiveSelector.Name {
		t.Errorf("expected %q, got %q", v1ObjectiveSelector.Name, reqs[0].Name)
	}
}

func TestMatchesPool(t *testing.T) {
	poolName := inferencePool.Name
	poolGroup := routing.InferencePoolAPIGroup
	ref := func(name, group string) apixv1.PoolObjectReference {
		return apixv1.PoolObjectReference{Name: apixv1.ObjectName(name), Group: apixv1.Group(group)}
	}
	tests := []struct {
		name       string
		spec       apixv1.InferenceObjectiveSpec
		poolLabels map[string]string
		want       bool
	}{
		{
			name: "poolRefs hit",
			spec: apixv1.InferenceObjectiveSpec{PoolRefs: []apixv1.PoolObjectReference{ref("other", poolGroup), ref(poolName, poolGroup)}},
			want: true,
		},
		{
			name: "poolRefs miss",
			spec: apixv1.InferenceObjectiveSpec{PoolRefs: []apixv1.PoolObjectReference{ref("other", poolGroup)}},
			want: false,
		},
		{
			name:       "selector hit",
			spec:       apixv1.InferenceObjectiveSpec{PoolSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}},
			poolLabels: map[string]string{"tiers": "shared"},
			want:       true,
		},
		{
			name:       "selector miss",
			spec:       apixv1.InferenceObjectiveSpec{PoolSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}},
			poolLabels: map[string]string{"tiers": "other"},
			want:       false,
		},
		{
			name:       "selector with requirements and no labels misses",
			spec:       apixv1.InferenceObjectiveSpec{PoolSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}},
			poolLabels: nil,
			want:       false,
		},
		{
			name:       "empty selector matches without labels",
			spec:       apixv1.InferenceObjectiveSpec{PoolSelector: &metav1.LabelSelector{}},
			poolLabels: nil,
			want:       true,
		},
		{
			name:       "invalid selector fails closed",
			spec:       apixv1.InferenceObjectiveSpec{PoolSelector: &metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{{Key: "tiers", Operator: "Bogus", Values: []string{"shared"}}}}},
			poolLabels: map[string]string{"tiers": "shared"},
			want:       false,
		},
		{
			name:       "union of miss and hit loads",
			spec:       apixv1.InferenceObjectiveSpec{PoolRefs: []apixv1.PoolObjectReference{ref("other", poolGroup)}, PoolSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}},
			poolLabels: map[string]string{"tiers": "shared"},
			want:       true,
		},
		{
			name:       "empty spec matches nothing",
			spec:       apixv1.InferenceObjectiveSpec{},
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
	if !reconciler.eventPredicate(infObjective1) {
		t.Error("poolRef objective for own pool should pass the event predicate")
	}
	if reconciler.eventPredicate(infObjective1Pool2) {
		t.Error("poolRef objective for another pool should not pass the event predicate")
	}
	if !reconciler.eventPredicateV1(v1ObjectiveSelector) {
		t.Error("selector objective should pass the v1 event predicate without label lookup")
	}
	if !reconciler.eventPredicateV1(v1ObjectiveShared) {
		t.Error("poolRefs objective for own pool should pass the v1 event predicate")
	}
	miss := testutil.MakeV1InferenceObjective("miss").
		Namespace(inferencePool.Namespace).
		PoolRefs(
			apixv1.PoolObjectReference{Name: "test-pool2", Group: apixv1.Group(routing.InferencePoolAPIGroup)},
		).ObjRef()
	if reconciler.eventPredicateV1(miss) {
		t.Error("poolRefs objective for another pool should not pass the v1 event predicate")
	}
}

// errorPoolReader fails pool reads to exercise the requeue path.
type errorPoolReader struct {
	client.Reader
}

func (r errorPoolReader) Get(ctx context.Context, nn types.NamespacedName, obj client.Object, opts ...client.GetOption) error {
	if _, ok := obj.(*v1.InferencePool); ok {
		return errors.NewServiceUnavailable("injected pool read error")
	}
	return r.Reader.Get(ctx, nn, obj, opts...)
}

func testReconciler(ds datastore.Datastore, reader client.Reader, watchV1 bool) *InferenceObjectiveReconciler {
	return &InferenceObjectiveReconciler{
		Reader:    reader,
		Datastore: ds,
		PoolGKNN: common.GKNN{
			NamespacedName: types.NamespacedName{Name: inferencePool.Name, Namespace: inferencePool.Namespace},
			GroupKind:      schema.GroupKind{Group: inferencePool.GroupVersionKind().Group, Kind: inferencePool.GroupVersionKind().Kind},
		},
		PrimaryV1: watchV1,
	}
}

func TestInferenceObjectiveV1Deletion(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = v1alpha2.Install(scheme)
	_ = apixv1.Install(scheme)
	_ = v1.Install(scheme)
	now := metav1.Now()
	deleting := testutil.MakeV1InferenceObjective("deleting").
		Namespace(inferencePool.Namespace).
		Priority(int32(1)).
		PoolRefs(
			apixv1.PoolObjectReference{Name: apixv1.ObjectName(inferencePool.Name), Group: apixv1.Group(routing.InferencePoolAPIGroup)},
		).ObjRef()
	deleting.DeletionTimestamp = &now
	deleting.Finalizers = []string{"finalizer"}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(deleting).
		Build()
	ds := datastore.NewDatastore(t.Context(), datalayer.NewTestRuntime(t, time.Second))
	ds.ObjectiveSet(v1ObjectiveShared)
	reconciler := testReconciler(ds, fakeClient, true)
	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: deleting.Name, Namespace: deleting.Namespace},
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got := ds.ObjectiveGet("deleting"); got != nil {
		t.Errorf("expected deleted objective removed, got %v", got)
	}
	if got := ds.ObjectiveGet(v1ObjectiveShared.Name); got == nil {
		t.Error("expected unrelated objective retained")
	}
}

func TestInferenceObjectivePoolReadError(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = v1alpha2.Install(scheme)
	_ = apixv1.Install(scheme)
	_ = v1.Install(scheme)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(v1ObjectiveSelector).
		Build()
	ds := datastore.NewDatastore(t.Context(), datalayer.NewTestRuntime(t, time.Second))
	reconciler := testReconciler(ds, errorPoolReader{Reader: fakeClient}, true)
	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: v1ObjectiveSelector.Name, Namespace: v1ObjectiveSelector.Namespace},
	})
	if err == nil {
		t.Error("expected pool read error to propagate for requeue")
	}
}

type recordingBands struct {
	submitted []map[int]struct{}
}

func (r *recordingBands) SubmitDesiredPriorities(desired map[int]struct{}) {
	r.submitted = append(r.submitted, desired)
}

func TestInferenceObjectiveBandsDefaultUnsetPriority(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = v1alpha2.Install(scheme)
	_ = apixv1.Install(scheme)
	_ = v1.Install(scheme)
	plain := testutil.MakeV1InferenceObjective("plain").
		Namespace(inferencePool.Namespace).
		PoolRefs(
			apixv1.PoolObjectReference{Name: apixv1.ObjectName(inferencePool.Name), Group: apixv1.Group(routing.InferencePoolAPIGroup)},
		).ObjRef()
	prioritized := testutil.MakeV1InferenceObjective("prioritized").
		Namespace(inferencePool.Namespace).
		Priority(int32(7)).
		CreationTimestamp(metav1.Unix(1000, 0)).
		PoolRefs(
			apixv1.PoolObjectReference{Name: apixv1.ObjectName(inferencePool.Name), Group: apixv1.Group(routing.InferencePoolAPIGroup)},
		).ObjRef()
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(plain, prioritized).
		Build()
	ds := datastore.NewDatastore(t.Context(), datalayer.NewTestRuntime(t, time.Second))
	bands := &recordingBands{}
	reconciler := testReconciler(ds, fakeClient, true)
	reconciler.PriorityBandControlPlane = bands
	for _, name := range []string{"plain", "prioritized"} {
		_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
			NamespacedName: types.NamespacedName{Name: name, Namespace: inferencePool.Namespace},
		})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
	}
	if len(bands.submitted) == 0 {
		t.Fatal("expected band submissions")
	}
	last := bands.submitted[len(bands.submitted)-1]
	if _, ok := last[7]; !ok {
		t.Errorf("expected submitted bands %v to contain 7", last)
	}
	// The unset priority defaults to band 0 at store time.
	if _, ok := last[0]; !ok {
		t.Errorf("expected submitted bands %v to contain defaulted 0", last)
	}
	if len(last) != 2 {
		t.Errorf("expected exactly the defaulted and set priorities submitted, got %v", last)
	}
}
