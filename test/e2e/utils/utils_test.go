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

package utils

import (
	"context"
	"errors"
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	infextv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

type failingCreateClient struct {
	client.Client
	failName string
	created  []string
	reads    int
}

func (c *failingCreateClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	c.reads++
	return c.Client.Get(ctx, key, obj, opts...)
}

func (c *failingCreateClient) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	c.reads++
	return c.Client.List(ctx, list, opts...)
}

func (c *failingCreateClient) Create(ctx context.Context, obj client.Object, opts ...client.CreateOption) error {
	if obj.GetName() == c.failName {
		return errors.New("injected create failure")
	}
	if err := c.Client.Create(ctx, obj, opts...); err != nil {
		return err
	}
	c.created = append(c.created, obj.GetName())
	return nil
}

func TestCreatesAllResourcesBeforeWaiting(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := infextv1.Install(scheme); err != nil {
		t.Fatal(err)
	}
	cli := &failingCreateClient{Client: fake.NewClientBuilder().WithScheme(scheme).Build()}
	resources := &CaseResources{Client: cli}
	objects, err := DecodeCaseObjects([]byte(`apiVersion: apps/v1
kind: Deployment
metadata:
  name: router
spec:
  replicas: 3
---
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: router-pool
`), "test")
	if err != nil {
		t.Fatal(err)
	}
	if err := resources.Create(ctx, objects); err != nil {
		t.Fatal(err)
	}
	if cli.reads != 0 || !reflect.DeepEqual(cli.created, []string{"router", "router-pool"}) {
		t.Fatalf("creation waited for the unready Deployment before creating dependencies: reads=%d created=%v", cli.reads, cli.created)
	}
}

func TestPartialCreationCleanup(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	foreign := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "foreign", Namespace: "test"}}
	for _, failure := range []string{"injected", "foreign"} {
		t.Run(failure, func(t *testing.T) {
			cli := &failingCreateClient{Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(foreign.DeepCopy()).Build(), failName: "injected"}
			resources := &CaseResources{Client: cli}
			obj := func(name string) *unstructured.Unstructured {
				return &unstructured.Unstructured{Object: map[string]any{
					"apiVersion": "v1", "kind": "ConfigMap", "metadata": map[string]any{"name": name, "namespace": "test"},
				}}
			}
			if err := resources.Create(ctx, []*unstructured.Unstructured{obj("owned"), obj(failure), obj("unattempted")}); err == nil {
				t.Fatal("expected create failure")
			}
			if !reflect.DeepEqual(cli.created, []string{"owned"}) || len(resources.created) != 1 {
				t.Fatalf("creation continued after failure: %v", cli.created)
			}
			if err := resources.Delete(ctx); err != nil {
				t.Fatal(err)
			}
			deleted, err := resources.Deleted(ctx)
			if err != nil || !deleted {
				t.Fatalf("created resources were not cleaned up: %t, %v", deleted, err)
			}
			if err := cli.Get(ctx, types.NamespacedName{Namespace: "test", Name: "foreign"}, &corev1.ConfigMap{}); err != nil {
				t.Fatalf("cleanup affected the preexisting object: %v", err)
			}
		})
	}
}

func TestCleanupWaitsForPods(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	deployment := &appsv1.Deployment{
		TypeMeta:   metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "test", UID: "deployment"},
		Spec:       appsv1.DeploymentSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"router": "owned"}}},
	}
	oldPod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Name: "old-router", Namespace: "test", Labels: map[string]string{"router": "owned"},
		DeletionTimestamp: &metav1.Time{Time: metav1.Now().Time}, Finalizers: []string{"test.example/hold"},
	}}
	cli := fake.NewClientBuilder().WithScheme(scheme).WithObjects(deployment, oldPod).Build()
	obj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(deployment)
	if err != nil {
		t.Fatal(err)
	}
	resources := &CaseResources{Client: cli, created: []*unstructured.Unstructured{{Object: obj}}}
	if err := resources.Delete(ctx); err != nil {
		t.Fatal(err)
	}
	if deleted, err := resources.Deleted(ctx); deleted || err != nil {
		t.Fatalf("cleanup must wait for terminating router Pods: %t, %v", deleted, err)
	}
	if err := cli.Get(ctx, client.ObjectKeyFromObject(oldPod), oldPod); err != nil {
		t.Fatal(err)
	}
	oldPod.Finalizers = nil
	if err := cli.Update(ctx, oldPod); err != nil {
		t.Fatal(err)
	}
	if deleted, err := resources.Deleted(ctx); !deleted || err != nil {
		t.Fatalf("cleanup did not finish after router Pod deletion: %t, %v", deleted, err)
	}
}
