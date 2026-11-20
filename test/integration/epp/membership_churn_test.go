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
	"net"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

// These tests change Pod membership with a healthy API server and require every
// request sent meanwhile to be routed to a current member.

const (
	churnFirstAddr  = "192.168.1.1:8000"
	churnSecondAddr = "192.168.1.2:8000"

	// churnSettleTimeout bounds how long a membership change may take to
	// reach the datastore.
	churnSettleTimeout = 10 * time.Second
	// churnWindow is how long requests keep flowing after the change is in
	// the datastore.
	churnWindow = 2 * time.Second
)

// setupChurnHarness starts an EPP over one ready Pod; any member that has
// received metrics is a routing candidate.
func setupChurnHarness(t *testing.T) *eppharness.TestHarness {
	t.Helper()
	h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithStandardMode(),
		eppharness.WithConfigText(endpointHealthConfig(""))).WithBaseResources()
	h.WithPods([]eppharness.PodState{eppharness.P(0, 0, 0, modelMyModelTarget)}).
		WaitForSync(1, modelMyModel).WaitForMetricsDelivery()
	return h
}

// routeThrough sends requests until settled holds, then for churnWindow more,
// and returns the count per address; each must go to an address in allowed.
func routeThrough(t *testing.T, h *eppharness.TestHarness, settled func() bool, allowed ...string) map[string]int {
	t.Helper()
	routed := map[string]int{}
	start := time.Now()
	var settledAt time.Time
	for settledAt.IsZero() || time.Since(settledAt) < churnWindow {
		endpoint := routeAndRespond(t, h, respondOK)
		require.Contains(t, allowed, endpoint, "request routed outside the pool members")
		routed[endpoint]++
		if settledAt.IsZero() && settled() {
			settledAt = time.Now()
		}
		require.False(t, settledAt.IsZero() && time.Since(start) > churnSettleTimeout,
			"the membership change did not reach the datastore")
	}
	return routed
}

// datastoreHolds reports whether the datastore holds exactly the endpoints at
// addrs and each of them has received metrics.
func datastoreHolds(h *eppharness.TestHarness, addrs ...string) func() bool {
	return func() bool {
		pods := h.Datastore.PodList(datastore.AllPodsPredicate)
		if len(pods) != len(addrs) {
			return false
		}
		for _, pod := range pods {
			meta := pod.GetMetadata()
			if !slices.Contains(addrs, net.JoinHostPort(meta.Address, meta.Port)) {
				return false
			}
			if m := pod.GetMetrics(); m == nil || m.UpdateTime.IsZero() {
				return false
			}
		}
		return true
	}
}

// TestMembershipChurn_ScaleUp adds a ready Pod; both members receive traffic
// once it is in the datastore.
func TestMembershipChurn_ScaleUp(t *testing.T) {
	h := setupChurnHarness(t)

	h.WithPods([]eppharness.PodState{eppharness.P(1, 0, 0, modelMyModelTarget)})
	routed := routeThrough(t, h, datastoreHolds(h, churnFirstAddr, churnSecondAddr), churnFirstAddr, churnSecondAddr)
	assert.Positive(t, routed[churnSecondAddr], "the added Pod received no requests")
}

// TestMembershipChurn_LabelOutsideSelector adds a label outside the pool
// selector to the only member, which keeps every request.
func TestMembershipChurn_LabelOutsideSelector(t *testing.T) {
	h := setupChurnHarness(t)
	api := eppharness.K8sClient()

	pod := &corev1.Pod{}
	require.NoError(t, api.Get(t.Context(), types.NamespacedName{Namespace: h.Namespace, Name: "pod-0"}, pod))
	pod.Labels["version"] = "v2"
	require.NoError(t, api.Update(t.Context(), pod))

	relabeled := func() bool {
		pods := h.Datastore.PodList(datastore.AllPodsPredicate)
		return len(pods) == 1 && pods[0].GetMetadata().Labels["version"] == "v2"
	}
	routeThrough(t, h, relabeled, churnFirstAddr)
}

// TestMembershipChurn_Replace adds a replacement Pod, then deletes the original;
// no request reaches it once it has left the datastore.
func TestMembershipChurn_Replace(t *testing.T) {
	h := setupChurnHarness(t)

	h.WithPods([]eppharness.PodState{eppharness.P(1, 0, 0, modelMyModelTarget)})
	routeThrough(t, h, datastoreHolds(h, churnFirstAddr, churnSecondAddr), churnFirstAddr, churnSecondAddr)

	original := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: h.Namespace, Name: "pod-0"}}
	require.NoError(t, eppharness.K8sClient().Delete(t.Context(), original, client.GracePeriodSeconds(0)))
	routeThrough(t, h, datastoreHolds(h, churnSecondAddr), churnFirstAddr, churnSecondAddr)

	routed := routeThrough(t, h, func() bool { return true }, churnSecondAddr)
	assert.Positive(t, routed[churnSecondAddr])
}
