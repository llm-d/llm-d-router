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

package runner

import (
	"context"
	"net/http"
	"slices"
	"testing"
	"time"

	"github.com/spf13/pflag"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

func TestSetup_AppliesDataParallelStartRankLabel(t *testing.T) {
	const startRankLabel = "leaderworkerset.sigs.k8s.io/worker-index"

	opts := runserver.NewOptions()
	fs := pflag.NewFlagSet(t.Name(), pflag.ContinueOnError)
	opts.AddFlags(fs)
	require.NoError(t, fs.Parse([]string{"--data-parallel-start-rank-label=" + startRankLabel}))
	opts.PoolName = testPoolName
	opts.PoolNamespace = "test-ns"
	opts.EndpointSelector = labels.Everything()
	opts.EndpointTargetPorts = []int{8000}
	opts.SecureServing = false
	opts.GRPCHealthPort = 0
	opts.MetricsPort = 0
	opts.RefreshMetricsInterval = time.Hour

	ctx := t.Context()
	r := NewRunner()
	servingCtx, cancelServing := context.WithCancel(context.WithoutCancel(ctx))
	r.cancelServing = cancelServing
	t.Cleanup(r.stopServing)

	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{corev1.SchemeGroupVersion})
	mapper.Add(corev1.SchemeGroupVersion.WithKind("Pod"), meta.RESTScopeNamespace)
	_, ds, err := r.setup(ctx, servingCtx, &rest.Config{Host: "http://127.0.0.1"}, opts, []func(*ctrl.Options){
		func(o *ctrl.Options) {
			skipNameValidation := true
			o.Controller.SkipNameValidation = &skipNameValidation
			o.Metrics.BindAddress = "0"
			o.MapperProvider = func(*rest.Config, *http.Client) (meta.RESTMapper, error) {
				return mapper, nil
			}
		},
	})
	require.NoError(t, err)
	t.Cleanup(ds.Clear)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-1",
			Namespace: opts.PoolNamespace,
			Labels: map[string]string{
				startRankLabel: "1",
			},
			Annotations: map[string]string{
				"llm-d.ai/data-parallel-size-local": "2",
			},
		},
		Status: corev1.PodStatus{
			PodIP: "10.0.0.2",
		},
	}
	require.NoError(t, ds.PodUpdateOrAddIfNotExist(ctx, pod))

	endpoints := ds.PodList(datastore.AllPodsPredicate)
	require.Len(t, endpoints, 2)
	ranks := make([]int, 0, len(endpoints))
	for _, ep := range endpoints {
		require.True(t, ep.GetMetadata().RankByHeader)
		ranks = append(ranks, ep.GetMetadata().RankIndex)
	}
	slices.Sort(ranks)
	require.Equal(t, []int{2, 3}, ranks)
}
