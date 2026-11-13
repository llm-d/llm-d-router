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

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
)

type wantEndpoint struct {
	Name      string            `json:"name"`
	Namespace string            `json:"namespace"`
	Address   string            `json:"address"`
	Port      string            `json:"port"`
	Rank      int               `json:"rank"`
	Labels    map[string]string `json:"labels"`
}

type wantEndpointsBody struct {
	Endpoints []wantEndpoint `json:"endpoints"`
}

func newEndpointsAPITestPod(namespace, name, ip string, podLabels map[string]string, ports ...int32) *corev1.Pod {
	containerPorts := make([]corev1.ContainerPort, 0, len(ports))
	for _, p := range ports {
		containerPorts = append(containerPorts, corev1.ContainerPort{ContainerPort: p})
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, Labels: podLabels},
		Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "vllm", Ports: containerPorts}}},
		Status: corev1.PodStatus{
			PodIP:      ip,
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	}
}

func newEndpointsAPITestDatastore(t *testing.T, targetPorts []int, pods ...*corev1.Pod) datastore.Datastore {
	t.Helper()
	pool := &datalayer.EndpointPool{
		Selector:    labels.Everything(),
		TargetPorts: targetPorts,
		Namespace:   "default",
		Name:        "pool",
	}
	ds := datastore.NewDatastore(t.Context(), datalayer.NewTestRuntime(t, time.Second)).WithEndpointPool(pool)
	for _, pod := range pods {
		if err := ds.PodUpdateOrAddIfNotExist(t.Context(), pod); err != nil {
			t.Fatalf("PodUpdateOrAddIfNotExist(%s/%s): %v", pod.Namespace, pod.Name, err)
		}
	}
	return ds
}

func TestEndpointsAPI_GET(t *testing.T) {
	appLabels := map[string]string{"app": "vllm", "tier": "gpu"}

	tests := []struct {
		name        string
		targetPorts []int
		pods        []*corev1.Pod
		want        []wantEndpoint
	}{
		{
			name:        "empty datastore lists no endpoints",
			targetPorts: []int{8000},
			want:        []wantEndpoint{},
		},
		{
			name:        "single pod reports its address port rank and labels",
			targetPorts: []int{8000},
			pods:        []*corev1.Pod{newEndpointsAPITestPod("default", "pod-a", "10.0.0.1", appLabels, 8000)},
			want: []wantEndpoint{
				{Name: "pod-a", Namespace: "default", Address: "10.0.0.1", Port: "8000", Rank: 0, Labels: appLabels},
			},
		},
		{
			name:        "multi-port pod yields one entry per rank",
			targetPorts: []int{8000, 8001},
			pods:        []*corev1.Pod{newEndpointsAPITestPod("default", "pod-a", "10.0.0.1", appLabels, 8000, 8001)},
			want: []wantEndpoint{
				{Name: "pod-a", Namespace: "default", Address: "10.0.0.1", Port: "8000", Rank: 0, Labels: appLabels},
				{Name: "pod-a", Namespace: "default", Address: "10.0.0.1", Port: "8001", Rank: 1, Labels: appLabels},
			},
		},
		{
			name:        "entries sorted by namespace then name then rank",
			targetPorts: []int{8000, 8001},
			pods: []*corev1.Pod{
				newEndpointsAPITestPod("zeta", "pod-a", "10.0.0.4", appLabels, 8000, 8001),
				newEndpointsAPITestPod("alpha", "pod-b", "10.0.0.3", appLabels, 8000, 8001),
				newEndpointsAPITestPod("alpha", "pod-a", "10.0.0.2", appLabels, 8000, 8001),
			},
			want: []wantEndpoint{
				{Name: "pod-a", Namespace: "alpha", Address: "10.0.0.2", Port: "8000", Rank: 0, Labels: appLabels},
				{Name: "pod-a", Namespace: "alpha", Address: "10.0.0.2", Port: "8001", Rank: 1, Labels: appLabels},
				{Name: "pod-b", Namespace: "alpha", Address: "10.0.0.3", Port: "8000", Rank: 0, Labels: appLabels},
				{Name: "pod-b", Namespace: "alpha", Address: "10.0.0.3", Port: "8001", Rank: 1, Labels: appLabels},
				{Name: "pod-a", Namespace: "zeta", Address: "10.0.0.4", Port: "8000", Rank: 0, Labels: appLabels},
				{Name: "pod-a", Namespace: "zeta", Address: "10.0.0.4", Port: "8001", Rank: 1, Labels: appLabels},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ds := newEndpointsAPITestDatastore(t, tc.targetPorts, tc.pods...)
			rec := httptest.NewRecorder()
			NewEndpointsAPIHandler(ds, func() bool { return true }).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, EndpointsAPIPath, nil))

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d; body: %s", rec.Code, http.StatusOK, rec.Body.String())
			}
			if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
				t.Errorf("Content-Type = %q, want %q", ct, "application/json")
			}
			if len(tc.want) == 0 {
				// The empty list must serialize as [] so clients never see null.
				if got := strings.TrimSpace(rec.Body.String()); got != `{"endpoints":[]}` {
					t.Errorf("body = %s, want {\"endpoints\":[]}", got)
				}
			}
			var got wantEndpointsBody
			if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
				t.Fatalf("decode body %q: %v", rec.Body.String(), err)
			}
			if diff := cmp.Diff(tc.want, got.Endpoints); diff != "" {
				t.Errorf("endpoints mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestEndpointsAPI_RejectsNonGET(t *testing.T) {
	ds := newEndpointsAPITestDatastore(t, []int{8000},
		newEndpointsAPITestPod("default", "pod-a", "10.0.0.1", nil, 8000))

	for _, method := range []string{http.MethodPost, http.MethodDelete} {
		t.Run(method, func(t *testing.T) {
			rec := httptest.NewRecorder()
			NewEndpointsAPIHandler(ds, func() bool { return true }).ServeHTTP(rec, httptest.NewRequest(method, EndpointsAPIPath, nil))

			if rec.Code != http.StatusMethodNotAllowed {
				t.Errorf("status = %d, want %d", rec.Code, http.StatusMethodNotAllowed)
			}
			if allow := rec.Header().Get("Allow"); allow != http.MethodGet {
				t.Errorf("Allow = %q, want %q", allow, http.MethodGet)
			}
		})
	}
}

// Before discovery's initial sync the datastore can be empty, so an empty 200
// would tell clients that configured peers do not exist.
func TestEndpointsAPI_UnavailableUntilDiscoveryReady(t *testing.T) {
	ds := newEndpointsAPITestDatastore(t, []int{8000},
		newEndpointsAPITestPod("default", "pod-a", "10.0.0.1", nil, 8000))

	tests := []struct {
		name       string
		ready      bool
		wantStatus int
	}{
		{name: "not ready returns 503", ready: false, wantStatus: http.StatusServiceUnavailable},
		{name: "ready returns 200", ready: true, wantStatus: http.StatusOK},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			NewEndpointsAPIHandler(ds, func() bool { return tc.ready }).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, EndpointsAPIPath, nil))
			if rec.Code != tc.wantStatus {
				t.Errorf("status = %d, want %d; body: %s", rec.Code, tc.wantStatus, rec.Body.String())
			}
		})
	}
}
