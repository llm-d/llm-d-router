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

package controller

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	podutil "github.com/llm-d/llm-d-router/pkg/epp/util/pod"
)

// EPPPeerReconciler discovers peer EPP replicas by watching Pods matching this
// EPP deployment's own selector and drives their membership through a
// PeerNotifier. Peer discovery must run on every replica (not just the
// leader), so leader election is disabled for its controller.
type EPPPeerReconciler struct {
	client.Reader
	// notifier receives peer add/update/remove events.
	notifier fwkdl.PeerNotifier
	// Selector matches this EPP deployment's own pods (its peers).
	Selector labels.Selector
	// Namespace is the namespace of this EPP deployment's pods.
	Namespace string
	// Port is the port peer replicas listen on for state sync. Unlike
	// EndpointSlices, Pods carry no self-reported service port, so it comes
	// from config.
	Port string
	// SelfAddress is this replica's pod IP, excluded from the peer set. Empty
	// includes all matching pods.
	SelfAddress string
	// OnFirstReconcile, if set, is called once after the first successful
	// reconciliation. Used by the plugin to signal readiness.
	OnFirstReconcile func()

	// prev is the last reported peer set, used to compute deletes. Access is
	// serialized by the controller (single concurrent reconcile).
	prev           map[types.NamespacedName]fwkdl.PeerMetadata
	firstReconcile bool
}

func (r *EPPPeerReconciler) Reconcile(ctx context.Context, _ ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var pods corev1.PodList
	if err := r.List(ctx, &pods,
		client.InNamespace(r.Namespace),
		client.MatchingLabelsSelector{Selector: r.Selector},
	); err != nil {
		return ctrl.Result{}, fmt.Errorf("listing peer pods in %s - %w", r.Namespace, err)
	}

	desired := r.desiredPeers(pods.Items)

	if r.notifier != nil {
		for id, peer := range desired {
			if existing, ok := r.prev[id]; !ok || existing != peer {
				p := peer
				r.notifier.Upsert(&p)
			}
		}
		for id := range r.prev {
			if _, ok := desired[id]; !ok {
				r.notifier.Delete(id)
			}
		}
	}
	r.prev = desired

	if !r.firstReconcile {
		r.firstReconcile = true
		if r.OnFirstReconcile != nil {
			r.OnFirstReconcile()
		}
	}

	logger.V(logutil.DEBUG).Info("Reconciled EPP peers", "peers", len(desired))
	return ctrl.Result{}, nil
}

// BindNotifier sets the notifier and replays any previously discovered peers.
// It may only be called once; subsequent calls return an error.
func (r *EPPPeerReconciler) BindNotifier(n fwkdl.PeerNotifier) error {
	if r.notifier != nil {
		return fmt.Errorf("notifier already bound")
	}
	r.notifier = n
	for _, peer := range r.prev {
		p := peer
		n.Upsert(&p)
	}
	return nil
}

// desiredPeers folds the ready, non-self pods matching Selector into a peer
// set, keyed by pod identity.
func (r *EPPPeerReconciler) desiredPeers(pods []corev1.Pod) map[types.NamespacedName]fwkdl.PeerMetadata {
	desired := map[types.NamespacedName]fwkdl.PeerMetadata{}
	for i := range pods {
		pod := &pods[i]
		if !podutil.IsPodReady(pod) || pod.Status.PodIP == "" || pod.Status.PodIP == r.SelfAddress {
			continue
		}
		id := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		desired[id] = fwkdl.PeerMetadata{ID: id, Address: pod.Status.PodIP, Port: r.Port}
	}
	return desired
}

func (r *EPPPeerReconciler) SetupWithManager(mgr ctrl.Manager) error {
	matches := func(obj client.Object) bool {
		return obj.GetNamespace() == r.Namespace && r.Selector.Matches(labels.Set(obj.GetLabels()))
	}
	filter := predicate.Funcs{
		CreateFunc:  func(e event.CreateEvent) bool { return matches(e.Object) },
		UpdateFunc:  func(e event.UpdateEvent) bool { return matches(e.ObjectOld) || matches(e.ObjectNew) },
		DeleteFunc:  func(e event.DeleteEvent) bool { return matches(e.Object) },
		GenericFunc: func(e event.GenericEvent) bool { return matches(e.Object) },
	}
	// Peer discovery runs on every replica; the leader alone is not enough.
	needLeaderElection := false
	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.Pod{}).
		WithEventFilter(filter).
		WithOptions(controller.Options{NeedLeaderElection: &needLeaderElection}).
		Complete(r)
}
