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
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	"github.com/llm-d/llm-d-router/internal/runnable"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/discovery/k8speer"
	"github.com/llm-d/llm-d-router/pkg/epp/statesync"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
)

func readyPeerPod(name, ns, ip string) *corev1.Pod {
	return testutil.MakePod(name).
		Namespace(ns).
		ReadyCondition().
		IP(ip).
		Labels(map[string]string{"app": "epp-peer-test"}).
		Complete().
		ObjRef()
}

// createPodWithStatus creates a pod and then updates its status, because the
// API server strips Status fields on create.
func createPodWithStatus(ctx context.Context, t *testing.T, c client.Client, pod *corev1.Pod) {
	t.Helper()
	status := pod.Status.DeepCopy()
	require.NoError(t, c.Create(ctx, pod))
	pod.Status = *status
	require.NoError(t, c.Status().Update(ctx, pod))
}

// TestIntegrationPeerPlugin runs the k8s-peer-discovery plugin end to end
// against a live API server. The datalayer runtime binds the Pod watch and
// the manager runs Start, so the manager drives the full lifecycle:
//
//   - peers reach the store through the notifier passed to Start
//   - a peer that is deleted is removed from the store
func TestIntegrationPeerPlugin(t *testing.T) {
	const (
		selfIP  = "10.0.0.1"
		peer1IP = "10.0.0.2"
		peer2IP = "10.0.0.3"
	)

	nsName := "epp-peer-plugin-" + uuid.New().String()[:8]

	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: nsName}}
	ctx, cancel := context.WithTimeout(context.Background(), testContextTimeout)
	defer cancel()

	require.NoError(t, k8sClient.Create(ctx, ns))
	t.Cleanup(func() {
		_ = k8sClient.Delete(context.Background(), ns)
	})

	// The plugin reads its own address from the environment to exclude itself.
	t.Setenv("POD_IP", selfIP)

	params := fmt.Sprintf(`{"selector":"app=epp-peer-test","port":"9002","namespace":%q}`, nsName)
	plugin, err := k8speer.Factory("peer-disc", json.NewDecoder(strings.NewReader(params)), nil)
	require.NoError(t, err)

	peerDisc, ok := plugin.(*k8speer.Plugin)
	require.True(t, ok, "factory returned %T, want *k8speer.Plugin", plugin)

	mgr, mgrClient := setupTestManager(t, testEnv.Config, nsName)
	runtime := datalayer.NewRuntime(0)
	require.NoError(t, peerDisc.RegisterDependencies(runtime))
	require.NoError(t, runtime.Configure(nil, logr.Discard()))
	require.NoError(t, runtime.Start(ctx, mgr))

	store := statesync.NewMemoryPeerStore()
	require.NoError(t, mgr.Add(runnable.NoLeaderElection(manager.RunnableFunc(func(ctx context.Context) error {
		return peerDisc.Start(ctx, fwkdl.NewPeerNotifier(store))
	}))))

	startManagerAndWaitForSync(ctx, t, mgr)

	// Create pods. The datalayer notification runtime feeds the store.
	createPodWithStatus(ctx, t, mgrClient, readyPeerPod("epp-0", nsName, selfIP))
	createPodWithStatus(ctx, t, mgrClient, readyPeerPod("epp-1", nsName, peer1IP))

	select {
	case <-peerDisc.Ready():
	case <-ctx.Done():
		t.Fatal("plugin did not become ready")
	}

	require.Eventually(t, func() bool {
		peers := store.Peers()
		return len(peers) == 1 && peers[0].Address == peer1IP
	}, eventWaitTimeout, eventPollInterval, "expected 1 peer")

	// Discovery keeps flowing.
	peer2 := readyPeerPod("epp-2", nsName, peer2IP)
	createPodWithStatus(ctx, t, mgrClient, peer2)

	require.Eventually(t, func() bool {
		return len(store.Peers()) == 2
	}, eventWaitTimeout, eventPollInterval, "expected 2 peers")

	require.NoError(t, mgrClient.Delete(ctx, peer2))

	require.Eventually(t, func() bool {
		peers := store.Peers()
		return len(peers) == 1 && peers[0].Address == peer1IP
	}, eventWaitTimeout, eventPollInterval, "expected epp-2 to be removed")
}
