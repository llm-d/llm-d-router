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

package lifecycle

import (
	"context"
	"errors"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// CaseResources tracks the objects one test case created so cleanup can delete
// exactly that set and wait for their Pods to terminate.
type CaseResources struct {
	Client  client.Client
	created []*unstructured.Unstructured
}

// Create creates objects in order and records each object that was created.
func (r *CaseResources) Create(ctx context.Context, objects []*unstructured.Unstructured) error {
	for _, obj := range objects {
		if err := r.Client.Create(ctx, obj); err != nil {
			return fmt.Errorf("create %s/%s: %w", obj.GetKind(), obj.GetName(), err)
		}
		r.created = append(r.created, obj.DeepCopy())
	}
	return nil
}

// Delete removes the recorded objects newest-first with UID preconditions so a
// same-named object created later is never deleted.
func (r *CaseResources) Delete(ctx context.Context) error {
	var errs []error
	for i := len(r.created) - 1; i >= 0; i-- {
		obj := r.created[i]
		uid := obj.GetUID()
		err := r.Client.Delete(ctx, obj, client.PropagationPolicy(metav1.DeletePropagationForeground),
			client.Preconditions{UID: &uid})
		if err != nil && !apierrors.IsNotFound(err) {
			errs = append(errs, fmt.Errorf("delete %s/%s: %w", obj.GetKind(), obj.GetName(), err))
		}
	}
	return errors.Join(errs...)
}

// Deleted reports whether every recorded object is gone and no Pod from a
// recorded Deployment still exists.
func (r *CaseResources) Deleted(ctx context.Context) (bool, error) {
	for _, obj := range r.created {
		current := &unstructured.Unstructured{}
		current.SetGroupVersionKind(obj.GroupVersionKind())
		err := r.Client.Get(ctx, client.ObjectKeyFromObject(obj), current)
		if err == nil && current.GetUID() == obj.GetUID() {
			return false, nil
		}
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		if obj.GetKind() == "Deployment" {
			selector, _, err := unstructured.NestedStringMap(obj.Object, "spec", "selector", "matchLabels")
			if err != nil {
				return false, err
			}
			pods := &corev1.PodList{}
			if err := r.Client.List(ctx, pods, client.InNamespace(obj.GetNamespace()), client.MatchingLabels(selector)); err != nil {
				return false, err
			}
			if len(pods.Items) != 0 {
				return false, nil
			}
		}
	}
	return true, nil
}

// DeferCaseCleanup registers case cleanup that runs stop, dumps diagnostics on
// failure when keepOnFailure is set, deletes the recorded resources, and waits
// until they are fully terminated.
func DeferCaseCleanup(cfg *testutils.TestConfig, keepOnFailure bool, resources *CaseResources, namespace string, stop func()) {
	ginkgo.DeferCleanup(func() {
		if stop != nil {
			stop()
		}
		if ginkgo.CurrentSpecReport().Failed() && keepOnFailure {
			testutils.DumpPodsAndLogs(cfg, namespace)
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), cfg.ReadyTimeout)
		defer cancel()
		gomega.Expect(resources.Delete(ctx)).To(gomega.Succeed())
		gomega.Eventually(func() (bool, error) {
			return resources.Deleted(ctx)
		}, cfg.ReadyTimeout, cfg.Interval).Should(gomega.BeTrue(), "case resources and their Pods must terminate before names are reused")
	})
}
