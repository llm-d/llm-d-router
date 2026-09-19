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

package standalone

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
)

const (
	poolName = "food-review-inference-pool"
	eppName  = poolName + "-epp"

	simpleConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
  parameters:
    maxPrefixTokensToMatch: 16384
    lruCapacityPerServer: 256
- type: prefix-cache-scorer
- type: decode-filter
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
`
)

var podSelector = map[string]string{"app": poolName}

func testConfig(namespace, image string) Config {
	return Config{Namespace: namespace, EPPImage: image, PodSelector: podSelector, ReleaseName: poolName}
}

func TestStandaloneChart(t *testing.T) {
	for _, tc := range []struct {
		name      string
		namespace string
		replicas  int
		ports     []int32
	}{
		{"single", "e2e-1", 1, []int32{8000}},
		{"leader-election", "e2e-2", 3, []int32{8000}},
		{"data-parallel", "e2e-3", 1, []int32{8000, 8001}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			image := "registry.example:5000/team/router:dev"
			router, err := renderRouter(context.Background(), testConfig(tc.namespace, image), simpleConfig, tc.replicas, tc.ports)
			if err != nil {
				t.Fatal(err)
			}
			objects := make(map[string]*unstructured.Unstructured)
			for _, obj := range router.objects {
				if obj.GetNamespace() != tc.namespace {
					t.Fatalf("%s/%s has namespace %q", obj.GetKind(), obj.GetName(), obj.GetNamespace())
				}
				objects[obj.GetKind()+"/"+obj.GetName()] = obj
			}
			for _, name := range []string{
				"Deployment/" + eppName, "Service/" + eppName,
				"ServiceAccount/" + eppName, "ConfigMap/" + eppName,
				"ConfigMap/envoy", "InferencePool/" + poolName,
				"Role/" + eppName + "-sa", "RoleBinding/" + eppName + "-sa",
				"Role/" + eppName + "-non-sa", "RoleBinding/" + eppName + "-non-sa",
			} {
				if objects[name] == nil {
					t.Fatalf("chart did not render %s", name)
				}
			}
			deploy := objects["Deployment/"+eppName]
			replicas, _, _ := unstructured.NestedInt64(deploy.Object, "spec", "replicas")
			if replicas != int64(tc.replicas) {
				t.Fatalf("replicas = %d, want %d", replicas, tc.replicas)
			}
			containers, _, err := unstructured.NestedSlice(deploy.Object, "spec", "template", "spec", "containers")
			if err != nil || len(containers) != 2 {
				t.Fatalf("expected Envoy and EPP containers, got %v (%v)", containers, err)
			}
			var epp map[string]any
			for _, container := range containers {
				c := container.(map[string]any)
				if c["name"] == "epp" {
					epp = c
				}
				if _, ok := c["readinessProbe"]; !ok {
					t.Fatalf("%s has no readiness probe", c["name"])
				}
			}
			if epp["image"] != image {
				t.Fatalf("image = %v, want %s", epp["image"], image)
			}
			if epp["imagePullPolicy"] != "IfNotPresent" {
				t.Fatalf("unexpected EPP pull policy: %v", epp["imagePullPolicy"])
			}
			args, _, _ := unstructured.NestedStringSlice(epp, "args")
			for _, flag := range []string{"--allow-experimental-plugins=true", "--drain-timeout=0", "--metrics-endpoint-auth=false"} {
				if !strings.Contains(strings.Join(args, "\n"), flag) {
					t.Errorf("missing EPP flag %s", flag)
				}
			}
			hasElection := strings.Contains(strings.Join(args, "\n"), "--ha-enable-leader-election")
			if hasElection != (tc.replicas > 1) {
				t.Fatalf("leader election enabled = %t for %d replicas", hasElection, tc.replicas)
			}
			if hasElection && objects["Role/"+eppName+"-leader-election"] == nil {
				t.Fatal("missing leader election RBAC")
			}
			config, _, _ := unstructured.NestedString(objects["ConfigMap/"+eppName].Object, "data", "epp-config.yaml")
			if strings.TrimSpace(config) != strings.TrimSpace(simpleConfig) {
				t.Fatal("plugin configuration changed during rendering")
			}
			envoy, _, _ := unstructured.NestedString(objects["ConfigMap/envoy"].Object, "data", "envoy.yaml")
			if !strings.Contains(envoy, "failure_mode_allow: false") || !strings.Contains(envoy, "address: 127.0.0.1") {
				t.Fatal("expected chart fail-closed sidecar proxy configuration")
			}
			pool := objects["InferencePool/"+poolName]
			ports, _, _ := unstructured.NestedSlice(pool.Object, "spec", "targetPorts")
			if len(ports) != len(tc.ports) {
				t.Fatalf("target ports = %v, want %v", ports, tc.ports)
			}
			for i, port := range ports {
				if port.(map[string]any)["number"] != int64(tc.ports[i]) {
					t.Fatalf("target port = %v, want %d", port, tc.ports[i])
				}
			}
			picker, _, _ := unstructured.NestedString(pool.Object, "spec", "endpointPickerRef", "name")
			if picker != eppName || router.PoolName != poolName {
				t.Fatalf("pool or EPP service identity changed: %s, %s", router.PoolName, picker)
			}
			service := router.accessService(tc.namespace, 30080, 32090)
			if !reflect.DeepEqual(service.Spec.Selector, router.Selector) || service.Spec.Ports[0].NodePort != 30080 || service.Spec.Ports[1].NodePort != 32090 {
				t.Fatalf("invalid test access service: %+v", service.Spec)
			}
			ports, _, _ = unstructured.NestedSlice(objects["Service/"+eppName].Object, "spec", "ports")
			for _, want := range []int64{8081, 5557, 9090} {
				found := false
				for _, port := range ports {
					found = found || port.(map[string]any)["port"] == want
				}
				if !found {
					t.Errorf("EPP service missing port %d", want)
				}
			}
		})
	}
}

func TestStandaloneImageValues(t *testing.T) {
	for _, tc := range []struct{ image, registry, repository, tag string }{
		{"registry.example:5000/team/router:dev", "registry.example:5000", "team/router", "dev"},
		{"llm-d/router:dev", "docker.io", "llm-d/router", "dev"},
		{"router", "docker.io", "library/router", "latest"},
		{"localhost:5000/router", "localhost:5000", "router", "latest"},
	} {
		t.Run(tc.image, func(t *testing.T) {
			image, err := imageValues(tc.image)
			if err != nil {
				t.Fatal(err)
			}
			if image["registry"] != tc.registry || image["repository"] != tc.repository || image["tag"] != tc.tag {
				t.Fatalf("image values = %v", image)
			}
		})
	}
	if _, err := imageValues("router@sha256:abcd"); err == nil {
		t.Fatal("digest images must fail explicitly because the chart requires a tag")
	}
}

func readyTestRouterPod(name string, ready bool) corev1.Pod {
	condition := corev1.ConditionFalse
	if ready {
		condition = corev1.ConditionTrue
	}
	return corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, UID: types.UID(name)},
		Status: corev1.PodStatus{
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: condition}},
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "envoy-proxy", Ready: true, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
				{Name: "epp", Ready: ready, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
			},
		},
	}
}

func TestStandaloneReadyLeader(t *testing.T) {
	leader := readyTestRouterPod("leader", true)
	standby := readyTestRouterPod("standby", false)
	for _, tc := range []struct {
		name     string
		pods     []corev1.Pod
		replicas int
		wantErr  bool
	}{
		{"single", []corev1.Pod{leader}, 1, false},
		{"three with standbys", []corev1.Pod{leader, standby, standby}, 3, false},
		{"missing standby", []corev1.Pod{leader, standby}, 3, true},
		{"two leaders", []corev1.Pod{leader, leader, standby}, 3, true},
		{"no leader", []corev1.Pod{standby, standby, standby}, 3, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := readyLeader(tc.pods, tc.replicas)
			if (err != nil) != tc.wantErr {
				t.Fatalf("readiness error = %v, wantErr = %t", err, tc.wantErr)
			}
		})
	}
	leader.DeletionTimestamp = &metav1.Time{Time: metav1.Now().Time}
	if _, err := readyLeader([]corev1.Pod{leader}, 1); err == nil {
		t.Fatal("terminating leader must not be Ready")
	}
	leader = readyTestRouterPod("leader", true)
	leader.Status.ContainerStatuses[0].Ready = false
	if _, err := readyLeader([]corev1.Pod{leader}, 1); err == nil {
		t.Fatal("unready Envoy must not be Ready")
	}
}

func TestStandalonePortForwardRecovery(t *testing.T) {
	ctx := context.Background()
	var events []string
	var exited chan struct{}
	forward := &routerPortForward{start: func(_ context.Context, pod *corev1.Pod) (*forwardProcess, error) {
		events = append(events, "start "+pod.Name)
		exited = make(chan struct{})
		done := exited
		return &forwardProcess{done: done, stop: func() {
			events = append(events, "stop "+pod.Name)
			select {
			case <-done:
			default:
				close(done)
			}
		}}, nil
	}}
	leader := readyTestRouterPod("leader", true)
	standby := readyTestRouterPod("standby", false)
	for _, pods := range [][]corev1.Pod{{standby}, {standby, leader}, {leader, standby}} {
		if err := forward.reconcile(ctx, pods); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(events, []string{"start leader"}) {
		t.Fatalf("standby or duplicate forwarding: %v", events)
	}
	close(exited)
	if err := forward.reconcile(ctx, []corev1.Pod{leader}); err != nil {
		t.Fatal(err)
	}
	leader.DeletionTimestamp = &metav1.Time{Time: metav1.Now().Time}
	newLeader := readyTestRouterPod("new-leader", true)
	if err := forward.reconcile(ctx, []corev1.Pod{leader, standby, newLeader}); err != nil {
		t.Fatal(err)
	}
	forward.close()
	want := []string{"start leader", "stop leader", "start leader", "stop leader", "start new-leader", "stop new-leader"}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("forward lifecycle = %v, want %v", events, want)
	}
	select {
	case <-exited:
	default:
		t.Fatal("cleanup returned before the process exited")
	}
	forward.start = func(context.Context, *corev1.Pod) (*forwardProcess, error) {
		return nil, errors.New("start failed")
	}
	if err := forward.reconcile(ctx, []corev1.Pod{newLeader}); err == nil {
		t.Fatal("expected start failure")
	}
}

func TestStandalonePortForwardWaitsForExit(t *testing.T) {
	exited := make(chan struct{})
	stopped := make(chan struct{})
	closed := make(chan struct{})
	forward := &routerPortForward{process: &forwardProcess{done: exited, stop: func() { close(stopped) }}}
	go func() {
		forward.close()
		close(closed)
	}()
	<-stopped
	select {
	case <-closed:
		t.Fatal("port-forward cleanup returned before process exit")
	default:
	}
	close(exited)
	<-closed
}
