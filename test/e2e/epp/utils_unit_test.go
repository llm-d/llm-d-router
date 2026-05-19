/*
Copyright 2026 The Kubernetes Authors.

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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	igwtestutils "github.com/llm-d/llm-d-router/test/utils/igw"
)

// TestGenerateTraffic pins generateTraffic's outcome for a set of exec responses.
// Fixtures mirror real curl output: headers from -i plus the HTTP_STATUS trailer from -w.
// The HTTP/2 case guards against the reason phrase being dropped on an HTTP/2 connection.
func TestGenerateTraffic(t *testing.T) {
	cases := []struct {
		name           string
		response       string
		expectedStatus string
		wantErr        bool
	}{
		{"non-200 fails", "HTTP/1.1 503 Service Unavailable\r\n\r\nHTTP_STATUS=503\n", statusOK, true},
		{"200 succeeds", "HTTP/1.1 200 OK\r\n\r\nHTTP_STATUS=200\n", statusOK, false},
		// HTTP/2 drops the reason phrase ("HTTP/2 200" vs "HTTP/1.1 200 OK"); the trailer handles it.
		{"HTTP/2 200 succeeds", "HTTP/2 200\r\n\r\nHTTP_STATUS=200\n", statusOK, false},
		// empty expectedStatus accepts any exec-success (used when generating intentional errors)
		{`"" accepts 503`, "HTTP/1.1 503 Service Unavailable\r\n\r\nHTTP_STATUS=503\n", "", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			execFn := func(_ []string) (string, error) { return tc.response, nil }
			semaphore := make(chan struct{}, 1)
			err := generateTraffic([]string{"curl"}, 1, semaphore, execFn, 0, tc.expectedStatus)
			if (err != nil) != tc.wantErr {
				t.Fatalf("generateTraffic: wantErr=%v, gotErr=%v", tc.wantErr, err)
			}
		})
	}
}

// TestDeploymentReadyCondition exercises deploymentReadyCondition with a fake k8s client.
func TestDeploymentReadyCondition(t *testing.T) {
	const (
		ns         = "test-ns"
		deployName = "test-deploy"
	)
	podLabels := map[string]string{"app": "test"}

	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}

	makeDeploy := func(replicas, ready int32) *appsv1.Deployment {
		return &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: deployName, Namespace: ns},
			Spec: appsv1.DeploymentSpec{
				Selector: &metav1.LabelSelector{MatchLabels: podLabels},
			},
			Status: appsv1.DeploymentStatus{
				Replicas: replicas, ReadyReplicas: ready,
			},
		}
	}
	makePod := func(name string, terminating bool) *corev1.Pod {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns, Labels: podLabels},
		}
		if terminating {
			now := metav1.Now()
			pod.DeletionTimestamp = &now
			pod.Finalizers = []string{"test/block-deletion"}
		}
		return pod
	}

	cases := []struct {
		name    string
		deploy  *appsv1.Deployment
		pods    []*corev1.Pod
		wantErr bool
	}{
		{
			name:    "ready, no terminating pods",
			deploy:  makeDeploy(1, 1),
			pods:    []*corev1.Pod{makePod("pod-ready", false)},
			wantErr: false,
		},
		{
			// The regression case: after DeleteAllOf, a terminating pod still counts
			// toward ReadyReplicas. deploymentReadyCondition must detect it.
			name:    "terminating pod still present",
			deploy:  makeDeploy(1, 1),
			pods:    []*corev1.Pod{makePod("pod-terminating", true)},
			wantErr: true,
		},
		{
			name:    "no ready replicas",
			deploy:  makeDeploy(1, 0),
			pods:    []*corev1.Pod{makePod("pod-starting", false)},
			wantErr: true,
		},
		{
			name:    "replicas mismatch",
			deploy:  makeDeploy(2, 1),
			pods:    []*corev1.Pod{makePod("pod-1", false)},
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			objs := make([]client.Object, 0, 1+len(tc.pods))
			objs = append(objs, tc.deploy)
			for _, p := range tc.pods {
				objs = append(objs, p)
			}
			cfg := &igwtestutils.TestConfig{
				Context:   context.Background(),
				K8sClient: fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build(),
			}
			err := deploymentReadyCondition(cfg, types.NamespacedName{Name: deployName, Namespace: ns})
			if (err != nil) != tc.wantErr {
				t.Fatalf("wantErr=%v, got=%v", tc.wantErr, err)
			}
		})
	}
}
