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
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/llm-d/llm-d-router/pkg/epp/controller"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
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
		Selector:    selector,
		Namespace:   nsName,
		Port:        "9002",
		SelfAddress: "10.0.0.1",
	}
	require.NoError(t, r.BindNotifier(fwkdl.NewPeerNotifier(store)))
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
