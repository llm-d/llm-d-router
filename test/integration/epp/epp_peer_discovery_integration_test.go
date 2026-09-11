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

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/llm-d/llm-d-router/pkg/epp/controller"
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

// TestIntegrationEPPPeerDiscovery runs EPPPeerReconciler against a live API
// server, with the notifier set at construction. It covers:
//
//   - a ready pod matching the selector becomes a peer
//   - this replica's own pod is excluded by IP
//   - a peer that goes unready is removed
//   - a deleted peer is removed
func TestIntegrationEPPPeerDiscovery(t *testing.T) {
	nsName := "epp-peer-test-" + uuid.New().String()[:8]

	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: nsName}}
	ctx, cancel := context.WithTimeout(context.Background(), testContextTimeout)
	defer cancel()

	require.NoError(t, k8sClient.Create(ctx, ns))
	t.Cleanup(func() {
		_ = k8sClient.Delete(context.Background(), ns)
	})

	mgr, mgrClient := setupTestManager(t, testEnv.Config, nsName)

	store := statesync.NewMemoryPeerStore()
	selector := labels.SelectorFromSet(map[string]string{"app": "epp-peer-test"})

	r := &controller.EPPPeerReconciler{
		Reader:      mgr.GetClient(),
		Notifier:    fwkdl.NewPeerNotifier(store),
		Selector:    selector,
		Namespace:   nsName,
		Port:        "9002",
		SelfAddress: "10.0.0.1",
	}
	require.NoError(t, r.SetupWithManager(mgr))

	startManagerAndWaitForSync(ctx, t, mgr)

	// Create self pod (excluded) and one peer pod.
	selfPod := readyPeerPod("epp-0", nsName, "10.0.0.1")
	peer1 := readyPeerPod("epp-1", nsName, "10.0.0.2")
	createPodWithStatus(ctx, t, mgrClient, selfPod)
	createPodWithStatus(ctx, t, mgrClient, peer1)

	require.Eventually(t, func() bool {
		peers := store.Peers()
		return len(peers) == 1 && peers[0].Address == "10.0.0.2"
	}, eventWaitTimeout, eventPollInterval, "expected 1 peer (10.0.0.2)")

	// Add a second peer.
	peer2 := readyPeerPod("epp-2", nsName, "10.0.0.3")
	createPodWithStatus(ctx, t, mgrClient, peer2)

	require.Eventually(t, func() bool {
		return len(store.Peers()) == 2
	}, eventWaitTimeout, eventPollInterval, "expected 2 peers")

	// Mark peer1 as not ready.
	require.NoError(t, mgrClient.Get(ctx, client.ObjectKeyFromObject(peer1), peer1))
	peer1.Status.Conditions = []corev1.PodCondition{{
		Type:   corev1.PodReady,
		Status: corev1.ConditionFalse,
	}}
	require.NoError(t, mgrClient.Status().Update(ctx, peer1))

	require.Eventually(t, func() bool {
		peers := store.Peers()
		return len(peers) == 1 && peers[0].Address == "10.0.0.3"
	}, eventWaitTimeout, eventPollInterval, "expected 1 peer (10.0.0.3) after marking epp-1 not ready")

	// Delete peer2.
	require.NoError(t, mgrClient.Delete(ctx, peer2))

	require.Eventually(t, func() bool {
		return len(store.Peers()) == 0
	}, eventWaitTimeout, eventPollInterval, "expected 0 peers after deleting epp-2")
}

// TestIntegrationPeerPlugin runs the k8s-peer-discovery plugin end to end
// against a live API server. SetupWithManager registers both the reconciler
// and a Start runnable, so the manager drives the full lifecycle:
//
//   - peers flow directly into the store from the first reconcile
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
	require.NoError(t, peerDisc.SetupWithManager(mgr))

	startManagerAndWaitForSync(ctx, t, mgr)

	// Create pods. The manager-registered Start runnable binds the buffer to
	// the store, so peers flow through to Store() once the manager is up.
	createPodWithStatus(ctx, t, mgrClient, readyPeerPod("epp-0", nsName, selfIP))
	createPodWithStatus(ctx, t, mgrClient, readyPeerPod("epp-1", nsName, peer1IP))

	select {
	case <-peerDisc.Ready():
	case <-ctx.Done():
		t.Fatal("plugin did not become ready")
	}

	require.Eventually(t, func() bool {
		peers := peerDisc.Store().Peers()
		return len(peers) == 1 && peers[0].Address == peer1IP
	}, eventWaitTimeout, eventPollInterval, "expected 1 peer")

	// Discovery keeps flowing.
	peer2 := readyPeerPod("epp-2", nsName, peer2IP)
	createPodWithStatus(ctx, t, mgrClient, peer2)

	require.Eventually(t, func() bool {
		return len(peerDisc.Store().Peers()) == 2
	}, eventWaitTimeout, eventPollInterval, "expected 2 peers")

	require.NoError(t, mgrClient.Delete(ctx, peer2))

	require.Eventually(t, func() bool {
		peers := peerDisc.Store().Peers()
		return len(peers) == 1 && peers[0].Address == peer1IP
	}, eventWaitTimeout, eventPollInterval, "expected epp-2 to be removed")
}
