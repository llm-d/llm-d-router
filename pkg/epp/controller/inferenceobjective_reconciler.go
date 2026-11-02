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
	"cmp"
	"context"
	"fmt"
	"maps"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	apixv1 "github.com/llm-d/llm-d-router/apix/v1"
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
	// PrimaryV1 selects v1 as the served primary. Otherwise v1alpha2 (or
	// the legacy group) is served.
	PrimaryV1 bool
	// WatchV1Alpha2 watches llm-d.ai/v1alpha2 alongside a v1 primary and
	// converts matches at the edge. Set only when both are served; v1 is
	// evaluated first.
	WatchV1Alpha2 bool
}

// Reconcile normalizes served versions to the v1 shape and evaluates
// each once, primary first. The first match wins; shared names across
// versions must not be dual-written (see SecondaryObjectiveGV).
func (c *InferenceObjectiveReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).V(logutil.DEFAULT)
	ctx = ctrl.LoggerInto(ctx, logger)

	logger.Info("Reconciling InferenceObjective")

	var candidates []*apixv1.InferenceObjective
	if c.PrimaryV1 {
		v1obj := &apixv1.InferenceObjective{}
		if err := c.Get(ctx, req.NamespacedName, v1obj); err != nil {
			if !errors.IsNotFound(err) {
				return ctrl.Result{}, fmt.Errorf("unable to get InferenceObjective - %w", err)
			}
		} else if v1obj.DeletionTimestamp.IsZero() {
			candidates = append(candidates, v1obj)
		}
		if c.WatchV1Alpha2 {
			legacy := &v1alpha2.InferenceObjective{}
			if err := c.Get(ctx, req.NamespacedName, legacy); err != nil {
				if !errors.IsNotFound(err) {
					return ctrl.Result{}, fmt.Errorf("unable to get InferenceObjective - %w", err)
				}
			} else if legacy.DeletionTimestamp.IsZero() && legacy.Spec.PoolRef.Name != "" {
				// Under None conversion both Gets read the same stored
				// object; an empty poolRef means the v1alpha2 view of an
				// object authored through v1, which the primary candidate
				// already carries.
				logger.Info("DEPRECATION: llm-d.ai/v1alpha2/InferenceObjective is deprecated",
					"replacement", "llm-d.ai/v1/InferenceObjective")
				candidates = append(candidates, apixv1.ConvertFromV1Alpha2(legacy))
			}
		}
	} else {
		legacy := &v1alpha2.InferenceObjective{}
		if err := c.Get(ctx, req.NamespacedName, legacy); err != nil {
			if !errors.IsNotFound(err) {
				return ctrl.Result{}, fmt.Errorf("unable to get InferenceObjective - %w", err)
			}
		} else if legacy.DeletionTimestamp.IsZero() {
			candidates = append(candidates, apixv1.ConvertFromV1Alpha2(legacy))
		}
	}

	if len(candidates) == 0 {
		// InferenceObjective object got deleted.
		c.Datastore.ObjectiveDelete(req.NamespacedName)
		c.syncPriorityBands()
		return ctrl.Result{}, nil
	}

	var poolLabels map[string]string
	for _, candidate := range candidates {
		if candidate.Spec.PoolSelector != nil {
			var err error
			poolLabels, _, err = c.ownPoolLabels(ctx)
			if err != nil {
				return ctrl.Result{}, err
			}
			break
		}
	}
	for _, current := range candidates {
		if !matchesPool(current.Spec, c.PoolGKNN, poolLabels) {
			continue
		}
		// Add or update the stored objective.
		logger = logger.WithValues("poolRefs", current.Spec.PoolRefs, "poolSelector", current.Spec.PoolSelector)
		if current.Spec.Priority == nil {
			// The API defines an unset priority as 0.
			current.Spec.Priority = ptr.To(int32(0))
		}
		c.Datastore.ObjectiveSet(current)
		c.syncPriorityBands()
		logger.Info("Added/Updated InferenceObjective")
		return ctrl.Result{}, nil
	}

	// No served version targets this inferencePool.
	logger.V(logutil.DEBUG).Info("Ignoring InferenceObjective without pool match",
		"candidates", len(candidates),
		"poolRefs", candidates[0].Spec.PoolRefs,
		"poolSelector", candidates[0].Spec.PoolSelector)
	c.Datastore.ObjectiveDelete(req.NamespacedName)
	c.syncPriorityBands()
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
	b := ctrl.NewControllerManagedBy(mgr)
	if c.PrimaryV1 {
		b = b.For(&apixv1.InferenceObjective{}, builder.WithPredicates(predicate.Funcs{
			CreateFunc: func(e event.CreateEvent) bool { return c.eventPredicateV1(e.Object.(*apixv1.InferenceObjective)) },
			UpdateFunc: func(e event.UpdateEvent) bool {
				return c.eventPredicateV1(e.ObjectOld.(*apixv1.InferenceObjective)) || c.eventPredicateV1(e.ObjectNew.(*apixv1.InferenceObjective))
			},
			DeleteFunc:  func(e event.DeleteEvent) bool { return c.eventPredicateV1(e.Object.(*apixv1.InferenceObjective)) },
			GenericFunc: func(e event.GenericEvent) bool { return c.eventPredicateV1(e.Object.(*apixv1.InferenceObjective)) },
		}))
		// Selectors only exist on v1 objects. Without a served v1 there
		// is nothing label-driven to requeue for, so the pool watch is
		// set up together with the v1 source.
		b = b.Watches(
			&v1.InferencePool{},
			handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []ctrl.Request {
				if obj.GetName() != c.PoolGKNN.Name || obj.GetNamespace() != c.PoolGKNN.Namespace {
					return nil
				}
				return c.objectivesForPool(ctx)
			}),
			builder.WithPredicates(poolEventPredicate),
		)
		if c.WatchV1Alpha2 {
			b = b.Watches(
				&v1alpha2.InferenceObjective{},
				&handler.EnqueueRequestForObject{},
				builder.WithPredicates(predicate.Funcs{
					CreateFunc: func(e event.CreateEvent) bool {
						o, ok := e.Object.(*v1alpha2.InferenceObjective)
						return ok && c.eventPredicate(o)
					},
					UpdateFunc: func(e event.UpdateEvent) bool {
						old, okOld := e.ObjectOld.(*v1alpha2.InferenceObjective)
						new, okNew := e.ObjectNew.(*v1alpha2.InferenceObjective)
						return (okOld && c.eventPredicate(old)) || (okNew && c.eventPredicate(new))
					},
					DeleteFunc: func(e event.DeleteEvent) bool {
						o, ok := e.Object.(*v1alpha2.InferenceObjective)
						return ok && c.eventPredicate(o)
					},
					GenericFunc: func(e event.GenericEvent) bool {
						o, ok := e.Object.(*v1alpha2.InferenceObjective)
						return ok && c.eventPredicate(o)
					},
				}),
			)
		}
	} else {
		b = b.For(&v1alpha2.InferenceObjective{}, builder.WithPredicates(predicate.Funcs{
			CreateFunc: func(e event.CreateEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
			UpdateFunc: func(e event.UpdateEvent) bool {
				return c.eventPredicate(e.ObjectOld.(*v1alpha2.InferenceObjective)) || c.eventPredicate(e.ObjectNew.(*v1alpha2.InferenceObjective))
			},
			DeleteFunc:  func(e event.DeleteEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
			GenericFunc: func(e event.GenericEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
		}))
	}
	return b.
		WithOptions(controller.Options{NeedLeaderElection: &needLeaderElection}).
		Complete(c)
}

// poolEventPredicate passes pool create events and label changes. Status
// writes carry no label signal, and pool deletion does not requeue: the pool
// reconciler clears the datastore, and recreation is covered by the create
// event.
var poolEventPredicate = predicate.Funcs{
	CreateFunc:  func(event.CreateEvent) bool { return true },
	UpdateFunc:  func(e event.UpdateEvent) bool { return !maps.Equal(e.ObjectOld.GetLabels(), e.ObjectNew.GetLabels()) },
	DeleteFunc:  func(event.DeleteEvent) bool { return false },
	GenericFunc: func(event.GenericEvent) bool { return false },
}

// eventPredicateV1 is a coarse pre-filter on v1 objective events. Selector
// bearing objectives always pass; Reconcile re-evaluates against the pool
// labels authoritatively.
func (c *InferenceObjectiveReconciler) eventPredicateV1(infObjective *apixv1.InferenceObjective) bool {
	if infObjective.Spec.PoolSelector != nil {
		return true
	}
	return matchesPoolRefs(infObjective.Spec, c.PoolGKNN)
}

func (c *InferenceObjectiveReconciler) eventPredicate(infObjective *v1alpha2.InferenceObjective) bool {
	return string(infObjective.Spec.PoolRef.Name) == c.PoolGKNN.Name && string(infObjective.Spec.PoolRef.Group) == c.PoolGKNN.Group
}

// matchesPoolRefs reports whether any list entry targets the pool. An empty
// entry kind falls back to the CRD default so undefaulted objects still
// match.
func matchesPoolRefs(spec apixv1.InferenceObjectiveSpec, pool common.GKNN) bool {
	for _, ref := range spec.PoolRefs {
		if string(ref.Name) == pool.Name && string(ref.Group) == pool.Group &&
			string(cmp.Or(ref.Kind, "InferencePool")) == pool.Kind {
			return true
		}
	}
	return false
}

// matchesPool reports whether the spec targets the pool by list entry or
// selector. A nil poolLabels means the pool is missing and selector
// matching fails closed. An empty selector is rejected at admission by the
// poolSelector validation rule and is not re-checked here.
func matchesPool(spec apixv1.InferenceObjectiveSpec, pool common.GKNN, poolLabels map[string]string) bool {
	if matchesPoolRefs(spec, pool) {
		return true
	}
	if spec.PoolSelector == nil || poolLabels == nil {
		return false
	}
	sel, err := metav1.LabelSelectorAsSelector(spec.PoolSelector)
	if err != nil {
		return false
	}
	return sel.Matches(labels.Set(poolLabels))
}

// ownPoolLabels returns the labels of this controller's pool and whether
// the pool exists. A missing pool yields nil labels so selector matching
// fails closed; other errors propagate for requeue.
func (c *InferenceObjectiveReconciler) ownPoolLabels(ctx context.Context) (map[string]string, bool, error) {
	pool := &v1.InferencePool{}
	if err := c.Get(ctx, types.NamespacedName{Name: c.PoolGKNN.Name, Namespace: c.PoolGKNN.Namespace}, pool); err != nil {
		if errors.IsNotFound(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("unable to get InferencePool - %w", err)
	}
	if pool.Labels == nil {
		return map[string]string{}, true, nil
	}
	return pool.Labels, true, nil
}

// objectivesForPool lists namespaced v1 objectives targeting this pool by
// selector or list entry, for re-reconciliation when the pool is created or
// its labels change.
func (c *InferenceObjectiveReconciler) objectivesForPool(ctx context.Context) []ctrl.Request {
	var reqs []ctrl.Request
	list := &apixv1.InferenceObjectiveList{}
	if err := c.List(ctx, list, client.InNamespace(c.PoolGKNN.Namespace)); err != nil {
		log.FromContext(ctx).Error(err, "Unable to list v1 InferenceObjectives for pool requeue")
		return nil
	}
	for _, obj := range list.Items {
		if obj.Spec.PoolSelector != nil || matchesPoolRefs(obj.Spec, c.PoolGKNN) {
			reqs = append(reqs, ctrl.Request{NamespacedName: types.NamespacedName{Name: obj.Name, Namespace: obj.Namespace}})
		}
	}
	return reqs
}
