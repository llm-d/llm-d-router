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

package controller

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clocktesting "k8s.io/utils/clock/testing"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	poolutil "github.com/llm-d/llm-d-router/pkg/epp/util/pool"
	fwkgaie "github.com/llm-d/llm-d-router/test/framework/gaie"
	fwkk8s "github.com/llm-d/llm-d-router/test/framework/k8s"
)

func TestDiscoveryFreshnessExpiresWhenMembershipReadsFail(t *testing.T) {
	for _, failure := range []string{"pool", "pods"} {
		t.Run(failure, func(t *testing.T) {
			ctx := t.Context()
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			require.NoError(t, v1.Install(scheme))
			pool := fwkgaie.MakeInferencePool("pool").Namespace("default").
				Selector(map[string]string{"app": "model"}).TargetPorts(8000).ObjRef()
			pod := fwkk8s.MakePod("model").Namespace("default").
				Labels(map[string]string{"app": "model"}).ReadyCondition().ObjRef()
			api := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pool, pod).Build()
			ds := datastore.NewDatastore(ctx, datalayer.NewTestRuntime(t, time.Second))
			require.NoError(t, ds.PoolSet(ctx, api, poolutil.InferencePoolToEndpointPool(pool)))
			unavailable := false
			var readDelay time.Duration
			clock := clocktesting.NewFakeClock(time.Now())
			reader := interceptor.NewClient(api, interceptor.Funcs{
				Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
					if unavailable && failure == "pool" {
						return errors.New("discovery unavailable")
					}
					return c.Get(ctx, key, obj, opts...)
				},
				List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					if unavailable && failure == "pods" {
						return errors.New("discovery unavailable")
					}
					clock.Step(readDelay)
					return c.List(ctx, list, opts...)
				},
			})
			d := NewDiscoveryFreshness(reader, ds, types.NamespacedName{Namespace: pool.Namespace, Name: pool.Name}, 10*time.Second)
			d.clock = clock
			require.Error(t, d.CheckReady())
			require.NoError(t, d.refresh(ctx))
			require.NoError(t, d.CheckReady())
			require.NoError(t, d.Admit(ctx, nil, nil))

			unavailable = true
			clock.Step(9 * time.Second)
			require.Error(t, d.refresh(ctx))
			require.NoError(t, d.CheckReady(), "a failed read retains the bounded grace period")
			clock.Step(time.Second)
			require.Error(t, d.CheckReady(), "lease renewal cannot extend membership freshness")
			var rejection errcommon.Error
			require.ErrorAs(t, d.Admit(ctx, nil, nil), &rejection)
			require.Equal(t, errcommon.ServiceUnavailable, rejection.Code)
			require.ErrorAs(t, d.Admit(ctx, nil, nil), &rejection)

			// The API recovers before the watch-driven routing store catches up.
			require.NoError(t, api.Delete(ctx, pod))
			unavailable = false
			require.Error(t, d.refresh(ctx))
			require.Error(t, d.CheckReady())
			ds.PodDelete(pod.Name)
			require.NoError(t, d.refresh(ctx))
			require.NoError(t, d.CheckReady())
			require.NoError(t, d.Admit(ctx, nil, nil))
			readDelay = 10 * time.Second
			require.NoError(t, d.refresh(ctx))
			require.Error(t, d.CheckReady(), "slow API reads must not extend the freshness budget")
		})
	}
}

func TestDiscoveryFreshnessInvalidatesKnownMembershipChanges(t *testing.T) {
	for _, change := range []string{"selector", "not-ready", "deleted-pool", "selector-list-failure", "ports-list-failure", "protocol-list-failure"} {
		t.Run(change, func(t *testing.T) {
			ctx := t.Context()
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			require.NoError(t, v1.Install(scheme))
			pool := fwkgaie.MakeInferencePool("pool").Namespace("default").
				Selector(map[string]string{"app": "model"}).TargetPorts(8000).ObjRef()
			pod := fwkk8s.MakePod("model").Namespace("default").
				Labels(map[string]string{"app": "model"}).ReadyCondition().ObjRef()
			api := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pool, pod).WithStatusSubresource(pod).Build()
			ds := datastore.NewDatastore(ctx, datalayer.NewTestRuntime(t, time.Second))
			require.NoError(t, ds.PoolSet(ctx, api, poolutil.InferencePoolToEndpointPool(pool)))
			d := NewDiscoveryFreshness(api, ds, client.ObjectKeyFromObject(pool), time.Minute)
			require.NoError(t, d.refresh(ctx))
			require.NoError(t, d.CheckReady())
			if strings.HasSuffix(change, "list-failure") {
				d.reader = interceptor.NewClient(api, interceptor.Funcs{
					List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
						return errors.New("Pod list unavailable")
					},
				})
			}
			switch change {
			case "selector", "selector-list-failure":
				pool.Spec.Selector.MatchLabels = map[v1.LabelKey]v1.LabelValue{"app": "withdrawn"}
				require.NoError(t, api.Update(ctx, pool))
			case "ports-list-failure":
				pool.Spec.TargetPorts = []v1.Port{{Number: 9000}}
				require.NoError(t, api.Update(ctx, pool))
			case "protocol-list-failure":
				pool.Spec.AppProtocol = "kubernetes.io/h2c"
				require.NoError(t, api.Update(ctx, pool))
			case "not-ready":
				pod.Status.Conditions = nil
				require.NoError(t, api.Status().Update(ctx, pod))
			case "deleted-pool":
				require.NoError(t, api.Delete(ctx, pool))
			}
			require.Error(t, d.refresh(ctx))
			require.Error(t, d.CheckReady(), "known stale membership must not use the discovery-error grace period")
		})
	}
}

func TestDiscoveryFreshnessLoopKeepsDelayedSuccessfulReadsReady(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, v1.Install(scheme))
	pool := fwkgaie.MakeInferencePool("pool").Namespace("default").Selector(map[string]string{"app": "model"}).TargetPorts(8000).ObjRef()
	api := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pool).Build()
	ds := datastore.NewDatastore(ctx, datalayer.NewTestRuntime(t, time.Second))
	require.NoError(t, ds.PoolSet(ctx, api, poolutil.InferencePoolToEndpointPool(pool)))
	started := make(chan struct{})
	finish := make(chan struct{})
	reader := interceptor.NewClient(api, interceptor.Funcs{
		List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			select {
			case started <- struct{}{}:
			case <-ctx.Done():
				return ctx.Err()
			}
			select {
			case <-finish:
			case <-ctx.Done():
				return ctx.Err()
			}
			return c.List(ctx, list, opts...)
		},
	})
	d := NewDiscoveryFreshness(reader, ds, client.ObjectKeyFromObject(pool), 10*time.Second)
	clock := clocktesting.NewFakeClock(time.Now())
	d.clock = clock
	done := make(chan error, 1)
	go func() { done <- d.Start(ctx) }()
	t.Cleanup(func() { cancel(); require.NoError(t, <-done) })
	receive := func() {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("verification did not start within its cadence")
		}
	}
	receive()
	for cycle := range 4 {
		snapshotStarted := clock.Now()
		clock.Step(3 * time.Second)
		if cycle > 0 {
			require.NoError(t, d.CheckReady(), "previous snapshot must cover the next successful read")
		}
		finish <- struct{}{}
		require.Eventually(t, func() bool {
			validated := d.validatedAt.Load()
			return validated != nil && validated.Equal(snapshotStarted)
		}, time.Second, time.Millisecond)
		require.NoError(t, d.CheckReady())
		clock.Step(time.Second)
		receive()
	}
}
