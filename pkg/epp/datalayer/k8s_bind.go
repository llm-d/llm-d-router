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

package datalayer

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"

	"github.com/go-logr/logr"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache/synctrack"
	"k8s.io/client-go/util/workqueue"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/controller-runtime/pkg/source"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// BindNotificationSource registers a watcher/reconciler for the source's GVK.
// The framework core owns the cache and reconciliation; the source only receives
// deep-copied events via Notify.
func BindNotificationSource(src fwkdl.NotificationSource, extractors []fwkdl.NotificationExtractor, mgr ctrl.Manager) error {
	_, err := bindNotificationSource(src, extractors, mgr)
	return err
}

func bindNotificationSource(src fwkdl.NotificationSource, extractors []fwkdl.NotificationExtractor, mgr ctrl.Manager) (*notificationInitialSync, error) {
	gvk := src.GVK()
	log := mgr.GetLogger().WithName("notification-controller").WithValues("gvk", gvk.Kind)
	initialSync := newNotificationInitialSync(src.TypedName().String())

	reconciler := &notificationReconciler{
		client:      mgr.GetClient(),
		src:         src,
		extractors:  extractors,
		initialSync: initialSync,
		gvk:         gvk,
		log:         log,
	}

	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(gvk)
	// Kind.WaitForSync completes after its handler has enqueued the initial
	// list. Track those keys through successful notification dispatch.
	enqueue := &handler.TypedEnqueueRequestForObject[*unstructured.Unstructured]{}
	eventHandler := handler.TypedFuncs[*unstructured.Unstructured, reconcile.Request]{
		CreateFunc: func(ctx context.Context, event event.TypedCreateEvent[*unstructured.Unstructured], queue workqueue.TypedRateLimitingInterface[reconcile.Request]) {
			if event.IsInInitialList && event.Object != nil {
				initialSync.processed.Start(client.ObjectKeyFromObject(event.Object))
			}
			enqueue.Create(ctx, event, queue)
		},
		UpdateFunc:  enqueue.Update,
		DeleteFunc:  enqueue.Delete,
		GenericFunc: enqueue.Generic,
	}
	kindSource := source.Kind(
		mgr.GetCache(),
		obj,
		eventHandler,
		predicate.TypedResourceVersionChangedPredicate[*unstructured.Unstructured]{},
	)

	// use the source's name to make the controller name unique
	// This allows multiple notification sources for the same GVK
	// (needed in tests, Configure() sources still imposes
	// one source per GVK).
	controllerName := "notify_" + strings.ToLower(gvk.Kind) + "_" + src.TypedName().Name

	err := ctrl.NewControllerManagedBy(mgr).
		// Naming the controller allows you to see specific metrics/logs for this watch
		Named(controllerName).
		WatchesRawSource(&notificationInitialSyncSource{
			SyncingSource: kindSource,
			initialSync:   initialSync,
		}).
		Complete(reconciler)
	if err != nil {
		return nil, err
	}
	return initialSync, nil
}

// Reconciler for notifications. This is a generic reconciler that can be used for any GVK.
type notificationReconciler struct {
	client      client.Client
	src         fwkdl.NotificationSource
	extractors  []fwkdl.NotificationExtractor
	initialSync *notificationInitialSync
	gvk         schema.GroupVersionKind
	log         logr.Logger
}

type notificationInitialSync struct {
	upstreamSynced atomic.Bool
	processed      synctrack.AsyncTracker[types.NamespacedName]
	sourceName     string
}

func newNotificationInitialSync(sourceName string) *notificationInitialSync {
	initialSync := &notificationInitialSync{sourceName: sourceName}
	initialSync.processed.UpstreamHasSynced = initialSync.upstreamSynced.Load
	return initialSync
}

func (s *notificationInitialSync) hasSynced() bool {
	return s.processed.HasSynced()
}

type notificationInitialSyncSource struct {
	source.SyncingSource
	initialSync *notificationInitialSync
}

func (s *notificationInitialSyncSource) WaitForSync(ctx context.Context) error {
	if err := s.SyncingSource.WaitForSync(ctx); err != nil {
		return err
	}
	s.initialSync.upstreamSynced.Store(true)
	return nil
}

// Reconciler carries out the actual notification logic.
func (rn *notificationReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := rn.log.WithValues("resource", req.NamespacedName, "gvk", rn.gvk.String())

	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(rn.gvk)

	event := &fwkdl.NotificationEvent{
		Type:   fwkdl.EventAddOrUpdate,
		Object: u,
	}

	err := rn.client.Get(ctx, req.NamespacedName, u)
	if err != nil {
		if apierrors.IsNotFound(err) {
			u.SetName(req.Name)
			u.SetNamespace(req.Namespace)
			event.Type = fwkdl.EventDelete
		} else {
			log.Error(err, "failed to fetch resource from cache")
			return ctrl.Result{}, err
		}
	}

	err = rn.dispatch(ctx, log, event)
	if err == nil {
		rn.initialSync.processed.Finished(req.NamespacedName)
	}
	return ctrl.Result{}, err
}

func (rn *notificationReconciler) dispatch(ctx context.Context, log logr.Logger, event *fwkdl.NotificationEvent) error {
	log.V(logging.TRACE).Info("processing notification", "eventType", event.Type)

	processed, err := rn.src.Notify(ctx, *event)
	if err != nil {
		log.Error(err, "notifier failed to process event")
		return err
	}
	if processed == nil {
		return nil
	}

	var extractErrors []error
	for _, ext := range rn.extractors {
		if err := ext.Extract(ctx, *processed); err != nil {
			log.Error(err, "extractor failed", "extractor", ext.TypedName())
			extractErrors = append(extractErrors, fmt.Errorf("extractor %s failed: %w", ext.TypedName(), err))
		}
	}

	return errors.Join(extractErrors...)
}
