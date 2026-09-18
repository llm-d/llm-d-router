/*
Copyright 2025 The Kubernetes Authors.
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

package controller

import (
	"context"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
)

type InferenceObjectiveReconciler struct {
	client.Reader
	Datastore                datastore.Datastore
	PoolGKNN                 common.GKNN
	PriorityBandControlPlane contracts.PriorityBandControlPlane
	RunOnNonLeaders          bool
}

func (c *InferenceObjectiveReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).V(logutil.DEFAULT)
	ctx = ctrl.LoggerInto(ctx, logger)

	logger.Info("Reconciling InferenceObjective")

	infObjective := &v1alpha2.InferenceObjective{}
	notFound := false
	if err := c.Get(ctx, req.NamespacedName, infObjective); err != nil {
		if !errors.IsNotFound(err) {
			return ctrl.Result{}, fmt.Errorf("unable to get InferenceObjective - %w", err)
		}
		notFound = true
	}

	// Keep compatibility while surfacing migration guidance for legacy group users.
	if strings.HasPrefix(infObjective.APIVersion, "inference.networking.x-k8s.io/") {
		logger.Info("DEPRECATION: apiVersion inference.networking.x-k8s.io/v1alpha2/InferenceObjective is deprecated",
			"replacement", "llm-d.ai/v1alpha2/InferenceObjective")
	}

	if notFound || !infObjective.DeletionTimestamp.IsZero() {
		// InferenceObjective object got deleted.
		c.Datastore.ObjectiveDelete(req.NamespacedName)
		c.syncPriorityBands()
		return ctrl.Result{}, nil
	}

	poolLabels, err := c.ownPoolLabels(ctx)
	if err != nil {
		return ctrl.Result{}, err
	}
	if !matchesPool(infObjective.Spec, c.PoolGKNN.Name, c.PoolGKNN.Group, poolLabels) {
		// InferenceObjective object stopped targeting this inferencePool.
		c.Datastore.ObjectiveDelete(req.NamespacedName)
		c.syncPriorityBands()
		return ctrl.Result{}, nil
	}

	// Add or update if the InferenceObjective instance has a creation timestamp older than the existing entry of the model.
	logger = logger.WithValues("poolRef", infObjective.Spec.PoolRef, "poolRefs", infObjective.Spec.PoolRefs,
		"poolSelector", infObjective.Spec.PoolSelector)
	c.Datastore.ObjectiveSet(infObjective)
	c.syncPriorityBands()
	logger.Info("Added/Updated InferenceObjective")

	return ctrl.Result{}, nil
}

func (c *InferenceObjectiveReconciler) syncPriorityBands() {
	if c.PriorityBandControlPlane == nil {
		return
	}
	desired := make(map[int]struct{})
	for _, objective := range c.Datastore.ObjectiveGetAll() {
		if objective.Spec.Priority != nil {
			desired[int(*objective.Spec.Priority)] = struct{}{}
		}
	}
	c.PriorityBandControlPlane.SubmitDesiredPriorities(desired)
}

func (c *InferenceObjectiveReconciler) SetupWithManager(mgr ctrl.Manager) error {
	needLeaderElection := !c.RunOnNonLeaders
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha2.InferenceObjective{}, builder.WithPredicates(predicate.Funcs{
			CreateFunc: func(e event.CreateEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
			UpdateFunc: func(e event.UpdateEvent) bool {
				return c.eventPredicate(e.ObjectOld.(*v1alpha2.InferenceObjective)) || c.eventPredicate(e.ObjectNew.(*v1alpha2.InferenceObjective))
			},
			DeleteFunc:  func(e event.DeleteEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
			GenericFunc: func(e event.GenericEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
		})).
		Watches(
			&v1.InferencePool{},
			handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []ctrl.Request {
				if obj.GetName() != c.PoolGKNN.Name || obj.GetNamespace() != c.PoolGKNN.Namespace {
					return nil
				}
				return c.objectivesWithSelector(ctx)
			}),
		).
		WithOptions(controller.Options{NeedLeaderElection: &needLeaderElection}).
		Complete(c)
}

// eventPredicate is a coarse pre-filter on objective events. Selector
// bearing objectives always pass; Reconcile re-evaluates against the
// pool labels authoritatively.
func (c *InferenceObjectiveReconciler) eventPredicate(infObjective *v1alpha2.InferenceObjective) bool {
	if infObjective.Spec.PoolSelector != nil {
		return true
	}
	return matchesPoolRefs(infObjective.Spec, c.PoolGKNN.Name, c.PoolGKNN.Group)
}

func matchesPoolRefs(spec v1alpha2.InferenceObjectiveSpec, poolName, poolGroup string) bool {
	if spec.PoolRef != nil && string(spec.PoolRef.Name) == poolName && string(spec.PoolRef.Group) == poolGroup {
		return true
	}
	for _, ref := range spec.PoolRefs {
		if string(ref.Name) == poolName && string(ref.Group) == poolGroup {
			return true
		}
	}
	return false
}

func matchesPool(spec v1alpha2.InferenceObjectiveSpec, poolName, poolGroup string, poolLabels map[string]string) bool {
	if matchesPoolRefs(spec, poolName, poolGroup) {
		return true
	}
	if spec.PoolSelector == nil {
		return false
	}
	sel, err := metav1.LabelSelectorAsSelector(spec.PoolSelector)
	if err != nil {
		return false
	}
	return sel.Matches(labels.Set(poolLabels))
}

// ownPoolLabels returns the labels of this controller's pool. A missing
// pool yields empty labels; other errors propagate for requeue.
func (c *InferenceObjectiveReconciler) ownPoolLabels(ctx context.Context) (map[string]string, error) {
	pool := &v1.InferencePool{}
	if err := c.Get(ctx, types.NamespacedName{Name: c.PoolGKNN.Name, Namespace: c.PoolGKNN.Namespace}, pool); err != nil {
		if errors.IsNotFound(err) {
			return map[string]string{}, nil
		}
		return nil, fmt.Errorf("unable to get InferencePool - %w", err)
	}
	return pool.Labels, nil
}

// objectivesWithSelector lists namespaced objectives carrying a pool
// selector, for re-reconciliation when the own pool changes.
func (c *InferenceObjectiveReconciler) objectivesWithSelector(ctx context.Context) []ctrl.Request {
	list := &v1alpha2.InferenceObjectiveList{}
	if err := c.List(ctx, list, client.InNamespace(c.PoolGKNN.Namespace)); err != nil {
		log.FromContext(ctx).V(logutil.DEBUG).Info("Unable to list InferenceObjectives for pool requeue", "error", err)
		return nil
	}
	var reqs []ctrl.Request
	for _, obj := range list.Items {
		if obj.Spec.PoolSelector != nil {
			reqs = append(reqs, ctrl.Request{NamespacedName: types.NamespacedName{Name: obj.Name, Namespace: obj.Namespace}})
		}
	}
	return reqs
}
