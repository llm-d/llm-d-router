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

package epp

import (
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	crconfig "sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkapix "github.com/llm-d/llm-d-router/test/framework/apix"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
	fwkgaie "github.com/llm-d/llm-d-router/test/framework/gaie"
)

// TestPoolRecreateKeepsRewritesAndObjectives verifies against a real API
// server with the shipped CRDs that deleting and recreating the InferencePool
// leaves InferenceModelRewrites and InferenceObjectives in the datastore.
// Neither reconciler requeues on pool creation for these objects.
func TestPoolRecreateKeepsRewritesAndObjectives(t *testing.T) {
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}",
		"sigs.k8s.io/gateway-api-inference-extension").Output()
	require.NoError(t, err, "failed to locate gateway-api-inference-extension module")
	env := &envtest.Environment{
		CRDDirectoryPaths: []string{
			filepath.Join(strings.TrimSpace(string(out)), "config", "crd", "bases"),
			filepath.Join(eppharness.RepoRoot(), "config", "crd", "bases"),
		},
		ErrorIfCRDPathMissing: true,
	}
	cfg, err := env.Start()
	require.NoError(t, err, "failed to start envtest environment")
	t.Cleanup(func() { _ = env.Stop() })

	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha2.Install(scheme))
	utilruntime.Must(v1.Install(scheme))

	skipNameValidation := true
	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme:     scheme,
		Metrics:    metricsserver.Options{BindAddress: "0"},
		Controller: crconfig.Controller{SkipNameValidation: &skipNameValidation},
	})
	require.NoError(t, err, "failed to create manager")

	poolName := "recreate-pool"
	namespace := "recreate-" + uuid.NewString()[:8]
	poolGKNN := common.GKNN{
		NamespacedName: types.NamespacedName{Name: poolName, Namespace: namespace},
		GroupKind:      schema.GroupKind{Group: routing.InferencePoolAPIGroup, Kind: "InferencePool"},
	}

	ds := datastore.NewDatastore(t.Context(), datalayer.NewTestRuntime(t, time.Second))
	require.NoError(t, (&controller.InferencePoolReconciler{
		Datastore: ds, Reader: mgr.GetClient(), RunOnNonLeaders: true,
	}).SetupWithManager(mgr), "failed to set up pool reconciler")
	require.NoError(t, (&controller.InferenceModelRewriteReconciler{
		Datastore: ds, Reader: mgr.GetClient(), PoolGKNN: poolGKNN, RunOnNonLeaders: true,
	}).SetupWithManager(mgr), "failed to set up rewrite reconciler")
	require.NoError(t, (&controller.InferenceObjectiveReconciler{
		Datastore: ds, Reader: mgr.GetClient(), PoolGKNN: poolGKNN, RunOnNonLeaders: true,
	}).SetupWithManager(mgr), "failed to set up objective reconciler")

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

	createPool := func() {
		pool := fwkgaie.MakeInferencePool(poolName).
			Namespace(namespace).
			Selector(map[string]string{"app": poolName}).
			EndpointPickerRef("epp").
			TargetPorts(8000).
			ObjRef()
		pool.Spec.EndpointPickerRef.Port = &v1.Port{Number: v1.PortNumber(9002)}
		require.NoError(t, directClient.Create(ctx, pool), "failed to create pool")
	}
	createPool()
	require.Eventually(t, ds.PoolHasSynced, 15*time.Second, 50*time.Millisecond, "pool was not synced")

	require.NoError(t, directClient.Create(ctx, &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{Name: "rewrite", Namespace: namespace},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			PoolRef: &v1alpha2.PoolObjectReference{
				Name:  v1alpha2.ObjectName(poolName),
				Group: v1alpha2.Group(routing.InferencePoolAPIGroup),
				Kind:  "InferencePool",
			},
			Rules: []v1alpha2.InferenceModelRewriteRule{{
				Matches: []v1alpha2.Match{{Model: &v1alpha2.ModelMatch{Value: "public-model"}}},
				Targets: []v1alpha2.TargetModel{{ModelRewrite: "served-model"}},
			}},
		},
	}), "failed to create rewrite")
	require.NoError(t, directClient.Create(ctx, fwkapix.MakeInferenceObjective("objective").
		Namespace(namespace).
		Priority(int32(5)).
		PoolName(poolName).
		PoolGroup(routing.InferencePoolAPIGroup).
		ObjRef()), "failed to create objective")

	rewriteLoaded := func() bool {
		rewrite, _ := ds.ModelRewriteGet("public-model")
		return rewrite != nil
	}
	objectiveLoaded := func() bool { return ds.ObjectiveGet("objective") != nil }
	require.Eventually(t, rewriteLoaded, 15*time.Second, 50*time.Millisecond, "rewrite was not loaded")
	require.Eventually(t, objectiveLoaded, 15*time.Second, 50*time.Millisecond, "objective was not loaded")

	poolKey := types.NamespacedName{Name: poolName, Namespace: namespace}
	require.NoError(t, directClient.Delete(ctx, &v1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{Name: poolName, Namespace: namespace},
	}))
	require.Eventually(t, func() bool {
		return errors.IsNotFound(directClient.Get(ctx, poolKey, &v1.InferencePool{})) && !ds.PoolHasSynced()
	}, 15*time.Second, 50*time.Millisecond, "pool deletion did not clear the datastore pool")

	createPool()
	require.Eventually(t, ds.PoolHasSynced, 15*time.Second, 50*time.Millisecond, "recreated pool was not synced")
	require.True(t, rewriteLoaded(), "rewrite missing from the datastore after pool recreate")
	require.True(t, objectiveLoaded(), "objective missing from the datastore after pool recreate")
}
