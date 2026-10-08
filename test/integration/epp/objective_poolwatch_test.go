/*
Copyright 2026 The Kubernetes Authors.
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

package epp

import (
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	crconfig "sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	apixv1 "github.com/llm-d/llm-d-router/apix/v1"
	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
)

// startObjectiveEnv boots an envtest environment serving the gaie v1
// InferencePool CRD and an InferenceObjective CRD from the given testdata
// subdirectory: "crd" serves v1 only, "crd-dual" serves v1alpha2 and v1
// with None conversion. The shipped CRD serves v1alpha2 only.
func startObjectiveEnv(t *testing.T, crdDir string) *rest.Config {
	t.Helper()
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}",
		"sigs.k8s.io/gateway-api-inference-extension").Output()
	require.NoError(t, err, "failed to locate gateway-api-inference-extension module")
	gaieModulePath := strings.TrimSpace(string(out))

	env := &envtest.Environment{
		CRDDirectoryPaths: []string{
			filepath.Join(gaieModulePath, "config", "crd", "bases"),
			filepath.Join(repoRootPath, "test", "integration", "epp", "testdata", crdDir),
		},
		ErrorIfCRDPathMissing: true,
	}
	cfg, err := env.Start()
	require.NoError(t, err, "failed to start envtest environment")
	t.Cleanup(func() { _ = env.Stop() })
	return cfg
}

// TestInferenceObjectivePoolWatchWiring verifies against a real API server
// that pool events requeue selector-bearing InferenceObjectives: a pool label
// change or pool creation must re-reconcile objectives whose poolSelector
// matches the pool, so datastore entries follow the pool's labels.
func TestInferenceObjectivePoolWatchWiring(t *testing.T) {
	cfg := startObjectiveEnv(t, "crd")

	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(apixv1.Install(scheme))
	utilruntime.Must(v1.Install(scheme))

	skipNameValidation := true
	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme:     scheme,
		Metrics:    metricsserver.Options{BindAddress: "0"},
		Controller: crconfig.Controller{SkipNameValidation: &skipNameValidation},
	})
	require.NoError(t, err, "failed to create manager")

	poolName := "pool-watch-pool"
	objectiveName := "selector-objective"
	namespace := "pool-watch-" + uuid.NewString()[:8]

	ds := datastore.NewDatastore(t.Context(), datalayer.NewTestRuntime(t, time.Second))
	reconciler := &controller.InferenceObjectiveReconciler{
		Datastore: ds,
		Reader:    mgr.GetClient(),
		PoolGKNN: common.GKNN{
			NamespacedName: types.NamespacedName{Name: poolName, Namespace: namespace},
			GroupKind:      schema.GroupKind{Group: routing.InferencePoolAPIGroup, Kind: "InferencePool"},
		},
		RunOnNonLeaders: true,
		PrimaryV1:       true,
	}
	require.NoError(t, reconciler.SetupWithManager(mgr), "failed to set up reconciler")

	ctx, cancel := context.WithCancel(t.Context())
	mgrErr := make(chan error, 1)
	go func() { mgrErr <- mgr.Start(ctx) }()
	t.Cleanup(func() {
		cancel()
		if err := <-mgrErr; err != nil && !strings.Contains(err.Error(), "context canceled") {
			t.Errorf("manager stopped unexpectedly: %v", err)
		}
	})

	directClient, err := client.New(cfg, client.Options{Scheme: scheme})
	require.NoError(t, err, "failed to create direct client")
	require.NoError(t, directClient.Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: namespace},
	}))

	createPool := func(tier string) {
		pool := testutil.MakeInferencePool(poolName).
			Namespace(namespace).
			Selector(map[string]string{"app": poolName}).
			EndpointPickerRef("epp").
			TargetPorts(8000).
			ObjRef()
		pool.Spec.EndpointPickerRef.Port = &v1.Port{Number: v1.PortNumber(9002)}
		pool.Labels = map[string]string{"tiers": tier}
		require.NoError(t, directClient.Create(ctx, pool), "failed to create pool")
	}
	poolKey := types.NamespacedName{Name: poolName, Namespace: namespace}
	updatePoolLabels := func(tier string) {
		pool := &v1.InferencePool{}
		require.NoError(t, directClient.Get(ctx, poolKey, pool), "failed to get pool")
		pool.Labels = map[string]string{"tiers": tier}
		require.NoError(t, directClient.Update(ctx, pool), "failed to update pool labels")
	}
	objectiveRegistered := func() bool { return ds.ObjectiveGet(objectiveName) != nil }

	createPool("shared")
	objective := testutil.MakeV1InferenceObjective(objectiveName).
		Namespace(namespace).
		Priority(int32(1)).
		PoolSelector(&metav1.LabelSelector{MatchLabels: map[string]string{"tiers": "shared"}}).
		ObjRef()
	require.NoError(t, directClient.Create(ctx, objective), "failed to create objective")

	// The objective's own create event registers it.
	require.Eventually(t, objectiveRegistered, 15*time.Second, 50*time.Millisecond,
		"objective was not registered from its create event")

	// A pool label change must requeue the objective; the selector no longer
	// matches, so the datastore entry is dropped.
	updatePoolLabels("dedicated")
	require.Eventually(t, func() bool { return !objectiveRegistered() }, 15*time.Second, 50*time.Millisecond,
		"pool label change did not requeue the selector objective")

	// Changing the labels back re-registers it.
	updatePoolLabels("shared")
	require.Eventually(t, objectiveRegistered, 15*time.Second, 50*time.Millisecond,
		"pool label change back did not requeue the selector objective")

	// Pool deletion does not requeue (the pool reconciler owns that path),
	// so the entry survives. Recreating the pool with non-matching labels
	// fires the create event: the requeue must drop the stale entry.
	require.NoError(t, directClient.Delete(ctx, &v1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{Name: poolName, Namespace: namespace},
	}))
	require.Eventually(t, func() bool {
		return errors.IsNotFound(directClient.Get(ctx, poolKey, &v1.InferencePool{}))
	}, 15*time.Second, 50*time.Millisecond, "pool was not deleted")

	createPool("dedicated")
	require.Eventually(t, func() bool { return !objectiveRegistered() }, 15*time.Second, 50*time.Millisecond,
		"pool create event did not requeue the selector objective")
}

// TestInferenceObjectiveEmptySelectorRejected verifies the CRD validation
// rule rejects an empty poolSelector at admission, so an objective cannot
// blanket-match every pool in the namespace. The reconciler does not
// re-check this; admission is the single enforcement point.
func TestInferenceObjectiveEmptySelectorRejected(t *testing.T) {
	cfg := startObjectiveEnv(t, "crd")

	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(apixv1.Install(scheme))
	c, err := client.New(cfg, client.Options{Scheme: scheme})
	require.NoError(t, err, "failed to create client")

	cases := []struct {
		name     string
		selector *metav1.LabelSelector
	}{
		{name: "empty selector", selector: &metav1.LabelSelector{}},
		{name: "empty matchLabels map", selector: &metav1.LabelSelector{MatchLabels: map[string]string{}}},
		{name: "empty matchExpressions list", selector: &metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			objective := testutil.MakeV1InferenceObjective(tc.name).
				Namespace("default").
				Priority(int32(1)).
				PoolSelector(tc.selector).
				ObjRef()
			err := c.Create(t.Context(), objective)
			require.Error(t, err, "empty poolSelector must be rejected at admission")
			require.Contains(t, err.Error(), "poolSelector must not be empty")
			err = c.Get(t.Context(), types.NamespacedName{Name: tc.name, Namespace: "default"}, &apixv1.InferenceObjective{})
			require.True(t, errors.IsNotFound(err), "rejected objective must not be stored")
		})
	}
}

// TestInferenceObjectiveDualVersionWiring verifies the WatchV1Alpha2 path
// against a real API server serving both versions with None conversion and
// v1alpha2 storage. v1alpha2-written objectives load through the secondary
// watch and are converted at the edge. v1-written objectives cannot work in
// this state: with None conversion the API server prunes stored objects
// against the storage version's schema, stripping poolRefs and poolSelector,
// so a v1 create silently stores an objective that targets nothing. This is
// why dual serving requires a conversion webhook (or a storage-version
// switch with object recreation) before v1 can serve.
func TestInferenceObjectiveDualVersionWiring(t *testing.T) {
	cfg := startObjectiveEnv(t, "crd-dual")

	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha2.Install(scheme))
	utilruntime.Must(apixv1.Install(scheme))
	utilruntime.Must(v1.Install(scheme))

	logs := &capturingLogSink{}
	skipNameValidation := true
	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme:     scheme,
		Logger:     logr.New(logs),
		Metrics:    metricsserver.Options{BindAddress: "0"},
		Controller: crconfig.Controller{SkipNameValidation: &skipNameValidation},
	})
	require.NoError(t, err, "failed to create manager")

	poolName := "dual-watch-pool"
	namespace := "dual-watch-" + uuid.NewString()[:8]

	ds := datastore.NewDatastore(t.Context(), datalayer.NewTestRuntime(t, time.Second))
	reconciler := &controller.InferenceObjectiveReconciler{
		Datastore: ds,
		Reader:    mgr.GetClient(),
		PoolGKNN: common.GKNN{
			NamespacedName: types.NamespacedName{Name: poolName, Namespace: namespace},
			GroupKind:      schema.GroupKind{Group: routing.InferencePoolAPIGroup, Kind: "InferencePool"},
		},
		RunOnNonLeaders: true,
		PrimaryV1:       true,
		WatchV1Alpha2:   true,
	}
	require.NoError(t, reconciler.SetupWithManager(mgr), "failed to set up reconciler")

	ctx, cancel := context.WithCancel(t.Context())
	mgrErr := make(chan error, 1)
	go func() { mgrErr <- mgr.Start(ctx) }()
	t.Cleanup(func() {
		cancel()
		if err := <-mgrErr; err != nil && !strings.Contains(err.Error(), "context canceled") {
			t.Errorf("manager stopped unexpectedly: %v", err)
		}
	})

	directClient, err := client.New(cfg, client.Options{Scheme: scheme})
	require.NoError(t, err, "failed to create direct client")
	require.NoError(t, directClient.Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: namespace},
	}))

	pool := testutil.MakeInferencePool(poolName).
		Namespace(namespace).
		Selector(map[string]string{"app": poolName}).
		EndpointPickerRef("epp").
		TargetPorts(8000).
		ObjRef()
	pool.Spec.EndpointPickerRef.Port = &v1.Port{Number: v1.PortNumber(9002)}
	require.NoError(t, directClient.Create(ctx, pool), "failed to create pool")

	// A v1-written objective stores losslessly: the v1alpha2 storage schema
	// is a superset of v1's targeting fields, so None conversion prunes
	// nothing and the primary watch loads it.
	v1Objective := testutil.MakeV1InferenceObjective("v1-objective").
		Namespace(namespace).
		Priority(int32(3)).
		PoolRefs(apixv1.PoolObjectReference{Name: apixv1.ObjectName(poolName), Group: apixv1.Group(routing.InferencePoolAPIGroup)}).
		ObjRef()
	require.NoError(t, directClient.Create(ctx, v1Objective), "failed to create v1 objective")
	stored := &apixv1.InferenceObjective{}
	require.NoError(t, directClient.Get(ctx, types.NamespacedName{Name: "v1-objective", Namespace: namespace}, stored))
	require.Len(t, stored.Spec.PoolRefs, 1, "v1 targeting must survive storage under the superset v1alpha2 schema")
	require.Equal(t, apixv1.ObjectName(poolName), stored.Spec.PoolRefs[0].Name)
	require.Eventually(t, func() bool { return ds.ObjectiveGet("v1-objective") != nil },
		15*time.Second, 50*time.Millisecond, "v1 objective was not loaded through the primary watch")

	// Under None conversion the secondary Get reads the same stored object
	// with an empty poolRef; that view must not log a deprecation or add a
	// conversion candidate for a pure v1 objective.
	require.Zero(t, logs.deprecations(), "pure v1 objective must not log a v1alpha2 deprecation")

	// A typed client can author a poolRefs-only v1alpha2 objective: omitzero
	// keeps the zero-value poolRef out of the serialized object. The v1 view
	// carries the targeting, so the primary watch loads it.
	refsOnly := testutil.MakeInferenceObjective("refs-only-objective").
		Namespace(namespace).
		Priority(int32(4)).
		ObjRef()
	refsOnly.Spec.PoolRefs = []v1alpha2.PoolObjectReference{{Name: v1alpha2.ObjectName(poolName)}}
	require.NoError(t, directClient.Create(ctx, refsOnly), "typed client must be able to author a poolRefs-only v1alpha2 objective")
	require.Eventually(t, func() bool { return ds.ObjectiveGet("refs-only-objective") != nil },
		15*time.Second, 50*time.Millisecond, "superset-authored v1alpha2 objective was not loaded through the primary watch")
	require.Zero(t, logs.deprecations(), "superset-authored v1alpha2 objective must not log a deprecation")

	// A v1alpha2-written objective reaches the datastore through the
	// secondary watch, converted to the v1 shape with the poolRef mapped
	// to a single poolRefs entry, and logs the deprecation.
	legacyObjective := testutil.MakeInferenceObjective("legacy-objective").
		Namespace(namespace).
		Priority(int32(2)).
		PoolName(poolName).
		PoolGroup(routing.InferencePoolAPIGroup).
		ObjRef()
	require.NoError(t, directClient.Create(ctx, legacyObjective), "failed to create v1alpha2 objective")
	require.Eventually(t, func() bool { return ds.ObjectiveGet("legacy-objective") != nil },
		15*time.Second, 50*time.Millisecond, "v1alpha2 objective was not loaded through the secondary watch")
	got := ds.ObjectiveGet("legacy-objective")
	require.NotNil(t, got.Spec.Priority)
	require.Equal(t, int32(2), *got.Spec.Priority, "converted objective must keep its priority")
	require.Len(t, got.Spec.PoolRefs, 1, "converted objective must target one pool")
	require.NotZero(t, logs.deprecations(), "objective using the legacy poolRef must log the deprecation")

	require.Len(t, ds.ObjectiveGetAll(), 3, "one datastore entry per objective, got %d", len(ds.ObjectiveGetAll()))

	// poolRef cannot be combined with the v1 targeting fields.
	mixed := testutil.MakeInferenceObjective("mixed-objective").
		Namespace(namespace).
		Priority(int32(1)).
		PoolName(poolName).
		PoolGroup(routing.InferencePoolAPIGroup).
		ObjRef()
	mixed.Spec.PoolRefs = []v1alpha2.PoolObjectReference{{Name: v1alpha2.ObjectName(poolName)}}
	err = directClient.Create(ctx, mixed)
	require.ErrorContains(t, err, "poolRef cannot be combined", "poolRef plus poolRefs must be rejected at admission")

	// An objective with no targeting at all is rejected; the unstructured
	// object exercises the kubectl-style authoring path.
	untargeted := &unstructured.Unstructured{}
	untargeted.SetGroupVersionKind(schema.GroupVersion{Group: v1alpha2.GroupVersion.Group, Version: v1alpha2.GroupVersion.Version}.WithKind("InferenceObjective"))
	untargeted.SetName("untargeted-objective")
	untargeted.SetNamespace(namespace)
	require.NoError(t, unstructured.SetNestedField(untargeted.Object, int64(1), "spec", "priority"))
	err = directClient.Create(ctx, untargeted)
	require.ErrorContains(t, err, "must be set", "objective without any targeting field must be rejected at admission")

	// The v1alpha2 copy of the empty-selector rule must reject on its own,
	// independently of the v1 version of the same rule.
	emptySelector := &unstructured.Unstructured{}
	emptySelector.SetGroupVersionKind(schema.GroupVersion{Group: v1alpha2.GroupVersion.Group, Version: v1alpha2.GroupVersion.Version}.WithKind("InferenceObjective"))
	emptySelector.SetName("empty-selector-objective")
	emptySelector.SetNamespace(namespace)
	require.NoError(t, unstructured.SetNestedMap(emptySelector.Object, map[string]interface{}{}, "spec", "poolSelector", "matchLabels"))
	err = directClient.Create(ctx, emptySelector)
	require.ErrorContains(t, err, "must not be empty", "v1alpha2 objective with an empty poolSelector must be rejected at admission")

	require.NoError(t, directClient.Delete(ctx, legacyObjective))
	require.Eventually(t, func() bool { return ds.ObjectiveGet("legacy-objective") == nil },
		15*time.Second, 50*time.Millisecond, "deleted v1alpha2 objective was not removed")
}

// TestInferenceObjectiveDualVersionV1StorageWiring verifies the same
// dual-serving setup with v1 as the storage version: v1-written objectives
// are fully functional through the primary watch, and the v1alpha2 view of
// them decodes without targeting so the secondary watch stays silent.
// A v1alpha2-written objective is stripped at storage in this state, which
// is why the storage version must switch exactly when users recreate their
// objects in v1.
func TestInferenceObjectiveDualVersionV1StorageWiring(t *testing.T) {
	cfg := startObjectiveEnv(t, "crd-dual-v1storage")

	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha2.Install(scheme))
	utilruntime.Must(apixv1.Install(scheme))
	utilruntime.Must(v1.Install(scheme))

	skipNameValidation := true
	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme:     scheme,
		Metrics:    metricsserver.Options{BindAddress: "0"},
		Controller: crconfig.Controller{SkipNameValidation: &skipNameValidation},
	})
	require.NoError(t, err, "failed to create manager")

	poolName := "dual-v1storage-pool"
	namespace := "dual-v1storage-" + uuid.NewString()[:8]

	ds := datastore.NewDatastore(t.Context(), datalayer.NewTestRuntime(t, time.Second))
	reconciler := &controller.InferenceObjectiveReconciler{
		Datastore: ds,
		Reader:    mgr.GetClient(),
		PoolGKNN: common.GKNN{
			NamespacedName: types.NamespacedName{Name: poolName, Namespace: namespace},
			GroupKind:      schema.GroupKind{Group: routing.InferencePoolAPIGroup, Kind: "InferencePool"},
		},
		RunOnNonLeaders: true,
		PrimaryV1:       true,
		WatchV1Alpha2:   true,
	}
	require.NoError(t, reconciler.SetupWithManager(mgr), "failed to set up reconciler")

	ctx, cancel := context.WithCancel(t.Context())
	mgrErr := make(chan error, 1)
	go func() { mgrErr <- mgr.Start(ctx) }()
	t.Cleanup(func() {
		cancel()
		if err := <-mgrErr; err != nil && !strings.Contains(err.Error(), "context canceled") {
			t.Errorf("manager stopped unexpectedly: %v", err)
		}
	})

	directClient, err := client.New(cfg, client.Options{Scheme: scheme})
	require.NoError(t, err, "failed to create direct client")
	require.NoError(t, directClient.Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: namespace},
	}))

	pool := testutil.MakeInferencePool(poolName).
		Namespace(namespace).
		Selector(map[string]string{"app": poolName}).
		EndpointPickerRef("epp").
		TargetPorts(8000).
		ObjRef()
	pool.Spec.EndpointPickerRef.Port = &v1.Port{Number: v1.PortNumber(9002)}
	require.NoError(t, directClient.Create(ctx, pool), "failed to create pool")

	// A v1-written objective loads natively through the primary watch.
	v1Objective := testutil.MakeV1InferenceObjective("v1-objective").
		Namespace(namespace).
		Priority(int32(3)).
		PoolRefs(apixv1.PoolObjectReference{Name: apixv1.ObjectName(poolName), Group: apixv1.Group(routing.InferencePoolAPIGroup)}).
		ObjRef()
	require.NoError(t, directClient.Create(ctx, v1Objective), "failed to create v1 objective")
	require.Eventually(t, func() bool { return ds.ObjectiveGet("v1-objective") != nil },
		15*time.Second, 50*time.Millisecond, "v1 objective was not loaded through the primary watch")
	require.Eventually(t, func() bool { return len(ds.ObjectiveGetAll()) == 1 },
		15*time.Second, 50*time.Millisecond, "expected exactly one datastore entry, got %d", len(ds.ObjectiveGetAll()))

	// A v1alpha2-written objective is stripped at storage in this state.
	legacyObjective := testutil.MakeInferenceObjective("legacy-objective").
		Namespace(namespace).
		Priority(int32(2)).
		PoolName(poolName).
		PoolGroup(routing.InferencePoolAPIGroup).
		ObjRef()
	require.NoError(t, directClient.Create(ctx, legacyObjective), "failed to create v1alpha2 objective")
	legacyStored := &v1alpha2.InferenceObjective{}
	require.NoError(t, directClient.Get(ctx, types.NamespacedName{Name: "legacy-objective", Namespace: namespace}, legacyStored))
	require.Empty(t, legacyStored.Spec.PoolRef.Name, "None conversion must strip v1alpha2 fields under v1 storage")
	require.Never(t, func() bool { return ds.ObjectiveGet("legacy-objective") != nil },
		3*time.Second, 100*time.Millisecond, "stripped v1alpha2 objective must not enter the datastore")

	require.NoError(t, directClient.Delete(ctx, v1Objective))
	require.Eventually(t, func() bool { return ds.ObjectiveGet("v1-objective") == nil },
		15*time.Second, 50*time.Millisecond, "deleted v1 objective was not removed")
}

// capturingLogSink records every log message the manager emits so tests can
// assert on reconciler log output.
type capturingLogSink struct {
	mu    sync.Mutex
	lines []string
}

func (s *capturingLogSink) Init(logr.RuntimeInfo) {}

func (s *capturingLogSink) Enabled(int) bool { return true }

func (s *capturingLogSink) Info(_ int, msg string, _ ...interface{}) { s.record(msg) }

func (s *capturingLogSink) Error(_ error, msg string, _ ...interface{}) { s.record(msg) }

func (s *capturingLogSink) WithValues(_ ...interface{}) logr.LogSink { return s }

func (s *capturingLogSink) WithName(_ string) logr.LogSink { return s }

func (s *capturingLogSink) record(msg string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lines = append(s.lines, msg)
}

func (s *capturingLogSink) deprecations() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, line := range s.lines {
		if strings.Contains(line, "DEPRECATION") {
			n++
		}
	}
	return n
}
