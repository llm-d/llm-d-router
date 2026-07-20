/*
Copyright 2025 The Kubernetes Authors.

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

package datastore

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net"
	"slices"
	"strconv"
	"strings"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	podutil "github.com/llm-d/llm-d-router/pkg/epp/util/pod"
)

var (
	errPoolNotSynced = errors.New("InferencePool is not initialized in data store")
	// errRegistrationDropped reports an endpoint that could not be tracked: its collector is
	// still registered from an earlier registration (an upsert overlapping an in-flight delete)
	// or failed to start. Callers match it with errors.Is to decide whether to retry.
	errRegistrationDropped = errors.New("endpoint registration dropped: collector already registered or failed to start")
	AllPodsPredicate       = func(_ fwkdl.Endpoint) bool { return true }
)

const (
	// activePortsAnnotation is used to specify which ports on a pod should be considered
	// as active for inference traffic. The value should be a comma-separated list of port numbers.
	// Example: "8000,8001,8002"
	activePortsAnnotation = "llm-d.ai/active-ports"

	// legacyGAIEActivePortsAnnotation is the legacy GAIE active ports annotation key, kept for backward compatibility.
	//
	// Deprecated: use activePortsAnnotation instead; this may be removed in a future release.
	legacyGAIEActivePortsAnnotation = "inference.networking.k8s.io/active-ports"
)

// The datastore is a local cache of relevant data for the given InferencePool (currently all pulled from k8s-api)
type Datastore interface {
	// InferencePool operations
	// PoolSet sets the given pool in datastore. If the given pool has different label selector than the previous pool
	// that was stored, the function triggers a resync of the pods to keep the datastore updated. If the given pool
	// is nil, this call triggers the datastore.Clear() function.
	PoolSet(ctx context.Context, reader client.Reader, endpointPool *datalayer.EndpointPool) error
	PoolGet() (*datalayer.EndpointPool, error)
	PoolHasSynced() bool
	PoolLabelsMatch(podLabels map[string]string) bool
	WithEndpointPool(pool *datalayer.EndpointPool) Datastore

	// InferenceObjective operations
	ObjectiveSet(infObjective *v1alpha2.InferenceObjective)
	ObjectiveGet(objectiveName string) *v1alpha2.InferenceObjective
	ObjectiveDelete(namespacedName types.NamespacedName)
	ObjectiveGetAll() []*v1alpha2.InferenceObjective

	// InferenceModelRewrite operations
	ModelRewriteSet(infModelRewrite *v1alpha2.InferenceModelRewrite)
	ModelRewriteDelete(namespacedName types.NamespacedName)
	// ModelRewriteGet returns the highest-precedence rewrite rule for a given
	// model name (prioritizing exact matches over generic wildcard rules) and
	// the name of the InferenceModelRewrite object.
	ModelRewriteGet(modelName string) (*v1alpha2.InferenceModelRewriteRule, string)
	ModelRewriteGetAll() []*v1alpha2.InferenceModelRewrite

	// PodList lists pods matching the given predicate.
	PodList(predicate func(fwkdl.Endpoint) bool) []fwkdl.Endpoint
	// PodUpdateOrAddIfNotExist stores or updates the endpoints for the given pod. It returns an
	// error when an endpoint registration was dropped (see upsertEndpoint); the pod is then not
	// tracked by the datastore and the caller must retry (e.g. by requeuing the reconcile).
	PodUpdateOrAddIfNotExist(ctx context.Context, pod *corev1.Pod) error
	PodDelete(podName string)

	// EndpointUpsert adds or updates an endpoint from a non-Kubernetes discovery source.
	// A dropped registration is logged; the endpoint stays untracked until the discovery
	// source re-emits it.
	EndpointUpsert(ctx context.Context, meta *fwkdl.EndpointMetadata)
	// EndpointDelete removes the endpoint with the given namespaced name.
	EndpointDelete(id types.NamespacedName)

	// Clears the store state, happens when the pool gets deleted.
	Clear()
}

// compile-time type assertion
var _ Datastore = &datastore{}

// NewDatastore creates a new data store.
func NewDatastore(parentCtx context.Context, epFactory datalayer.EndpointFactory) Datastore {
	// Initialize with defaults
	return &datastore{
		parentCtx:     parentCtx,
		pool:          nil,
		mu:            sync.RWMutex{},
		objectives:    make(map[string]*v1alpha2.InferenceObjective),
		modelRewrites: newModelRewriteStore(),
		pods:          &sync.Map{},
		epf:           epFactory,
	}
}

type datastore struct {
	// parentCtx controls the lifecycle of the background metrics goroutines that spawn up by the datastore.
	parentCtx context.Context
	// mu is used to synchronize access to pool, objectives, and rewrites.
	mu   sync.RWMutex
	pool *datalayer.EndpointPool
	// key: InferenceObjective name, value: *InferenceObjective
	objectives map[string]*v1alpha2.InferenceObjective
	// modelRewrites store for InferenceModelRewrite objects.
	modelRewrites *modelRewriteStore
	// key: types.NamespacedName, value: fwkdl.Endpoint
	pods *sync.Map
	epf  datalayer.EndpointFactory
	// needsResync forces the next PoolSet to run podResyncAll even when the pool is unchanged.
	// PoolSet stores the pool before resyncing, so without this flag a PoolSet retried after a
	// resync failure would compare the incoming pool against the already-stored identical pool
	// and skip the resync. Guarded by mu.
	needsResync bool
}

func (ds *datastore) WithEndpointPool(pool *datalayer.EndpointPool) Datastore {
	ds.pool = pool
	return ds
}

func (ds *datastore) Clear() {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.pool = nil
	ds.objectives = make(map[string]*v1alpha2.InferenceObjective)
	ds.modelRewrites = newModelRewriteStore()
	// stop all pods go routines before clearing the pods map.
	ds.pods.Range(func(_, v any) bool {
		ds.epf.ReleaseEndpoint(v.(fwkdl.Endpoint))
		return true
	})
	ds.pods.Clear()
}

// /// Pool APIs ///
func (ds *datastore) PoolSet(ctx context.Context, reader client.Reader, endpointPool *datalayer.EndpointPool) error {
	if endpointPool == nil {
		ds.Clear()
		return nil
	}
	logger := log.FromContext(ctx)
	ds.mu.Lock()
	defer ds.mu.Unlock()

	oldEndpointPool := ds.pool
	ds.pool = endpointPool

	selectorChanged := oldEndpointPool == nil || !selectorEqual(oldEndpointPool.Selector, endpointPool.Selector)
	targetPortsChanged := oldEndpointPool != nil && !slices.Equal(oldEndpointPool.TargetPorts, endpointPool.TargetPorts)

	if selectorChanged || targetPortsChanged || ds.needsResync {
		logger.V(logutil.DEFAULT).Info("Updating endpoints", "selector", endpointPool.Selector, "targetPortsChanged", targetPortsChanged)
		// A full resync is required to address the following cases:
		// 1) At startup, the pod events may get processed before the pool is synced with the datastore,
		//    and hence they will not be added to the store since pool selector is not known yet
		// 2) If the selector on the pool was updated, then we will not get any pod events, and so we need
		//    to resync the whole pool: remove pods in the store that don't match the new selector and add
		//    the ones that may have existed already to the store.
		// 3) If the targetPorts changed, we need to resync to remove orphaned rank endpoints that no longer
		//    exist in the new targetPorts configuration.
		if err := ds.podResyncAll(ctx, reader); err != nil {
			ds.needsResync = true
			return fmt.Errorf("failed to update pods according to the pool selector - %w", err)
		}
		ds.needsResync = false
	}

	return nil
}

func (ds *datastore) PoolGet() (*datalayer.EndpointPool, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	if ds.pool == nil {
		return nil, errPoolNotSynced
	}
	return ds.pool, nil
}

func (ds *datastore) PoolHasSynced() bool {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.pool != nil
}

func (ds *datastore) PoolLabelsMatch(podLabels map[string]string) bool {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	if ds.pool == nil || ds.pool.Selector == nil {
		return false
	}
	return ds.pool.Selector.Matches(labels.Set(podLabels))
}

// /// InferenceObjective APIs ///
func (ds *datastore) ObjectiveSet(infObjective *v1alpha2.InferenceObjective) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.objectives[infObjective.Name] = infObjective
}

func (ds *datastore) ObjectiveGet(objectiveName string) *v1alpha2.InferenceObjective {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.objectives[objectiveName]
}

func (ds *datastore) ObjectiveDelete(namespacedName types.NamespacedName) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	delete(ds.objectives, namespacedName.Name)
}

func (ds *datastore) ObjectiveGetAll() []*v1alpha2.InferenceObjective {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	res := make([]*v1alpha2.InferenceObjective, 0, len(ds.objectives))
	for _, v := range ds.objectives {
		res = append(res, v)
	}
	return res
}

func (ds *datastore) ModelRewriteSet(infModelRewrite *v1alpha2.InferenceModelRewrite) {
	// Configured model names always emit their real metric label; only
	// unconfigured request-supplied names are subject to the cardinality cap.
	metrics.PreAdmitModelLabels(configuredModelNames(infModelRewrite)...)
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.modelRewrites.set(infModelRewrite)
}

func (ds *datastore) ModelRewriteDelete(namespacedName types.NamespacedName) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.modelRewrites.delete(namespacedName)
}

func (ds *datastore) ModelRewriteGet(modelName string) (*v1alpha2.InferenceModelRewriteRule, string) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.modelRewrites.getRule(modelName)
}

func (ds *datastore) ModelRewriteGetAll() []*v1alpha2.InferenceModelRewrite {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.modelRewrites.getAll()
}

// /// Pods/endpoints APIs ///
// TODO: add a flag for callers to specify the staleness threshold for metrics.
// ref: https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/1046#discussion_r2246351694
func (ds *datastore) PodList(predicate func(fwkdl.Endpoint) bool) []fwkdl.Endpoint {
	res := []fwkdl.Endpoint{}

	ds.pods.Range(func(k, v any) bool {
		ep := v.(fwkdl.Endpoint)
		if predicate(ep) {
			res = append(res, ep)
		}
		return true
	})

	return res
}

func (ds *datastore) PodUpdateOrAddIfNotExist(ctx context.Context, pod *corev1.Pod) error {
	// Take a reference to pool under read lock to avoid racing with PoolSet().
	// This is safe because PoolSet() replaces the entire pool struct rather than
	// updating it in-place.
	ds.mu.RLock()
	pool := ds.pool
	ds.mu.RUnlock()

	if pool == nil {
		// Without the pool's target ports the pod cannot be mapped to endpoints; the resync
		// triggered when the pool syncs picks the pod up.
		log.FromContext(ctx).V(logutil.DEBUG).Info("Skipping pod upsert, InferencePool not synced", "name", pod.Name)
		return nil
	}
	return ds.podUpdateOrAddIfNotExist(ctx, pod, pool)
}

// podUpdateOrAddIfNotExist is the lock-free inner implementation.
// Callers must ensure pool is a non-nil consistent snapshot (either read under lock
// or already held, as in podResyncAll which runs under ds.mu.Lock via PoolSet).
// It returns a joined error covering every endpoint of the pod whose registration was dropped.
func (ds *datastore) podUpdateOrAddIfNotExist(ctx context.Context, pod *corev1.Pod, pool *datalayer.EndpointPool) error {
	if pool == nil {
		return nil
	}

	labels := make(map[string]string, len(pod.GetLabels()))
	maps.Copy(labels, pod.GetLabels())

	pods := []*fwkdl.EndpointMetadata{}
	activePorts := extractActivePorts(pod, pool.TargetPorts)
	for idx, port := range pool.TargetPorts {
		if !activePorts.Has(port) {
			continue
		}
		pods = append(pods,
			&fwkdl.EndpointMetadata{
				NamespacedName: createEndpointNamespacedName(pod, idx),
				PodName:        pod.Name,
				Address:        pod.Status.PodIP,
				Port:           strconv.Itoa(port),
				MetricsHost:    net.JoinHostPort(pod.Status.PodIP, strconv.Itoa(port)),
				Labels:         labels,
				RankIndex:      idx,
			})
	}

	if len(pods) == 0 {
		logger := log.FromContext(ctx)
		logger.V(logutil.VERBOSE).Info("No container ports match pool targetPorts, pod will not receive traffic",
			"pod", pod.Name, "namespace", pod.Namespace, "targetPorts", pool.TargetPorts)
	}

	added := false
	var errs []error
	existingEpSet := sets.Set[types.NamespacedName]{}
	for _, endpointMetadata := range pods {
		existingEpSet.Insert(endpointMetadata.NamespacedName)
		created, err := ds.upsertEndpoint(ctx, endpointMetadata)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if created {
			added = true
		}
	}
	logger := log.FromContext(ctx)
	if len(errs) == 0 {
		if added {
			logger.V(logutil.DEFAULT).Info("Pod added", "name", pod.Name)
		} else {
			logger.V(logutil.DEFAULT).Info("Pod already exists", "name", pod.Name)
		}
	}

	// remove endpoints that are no longer active in the pool
	for idx, port := range pool.TargetPorts {
		if activePorts.Has(port) {
			continue
		}

		namespacedName := createEndpointNamespacedName(pod, idx)
		if ep, ok := ds.pods.Load(namespacedName); ok {
			ds.pods.Delete(namespacedName)
			ds.epf.ReleaseEndpoint(ep.(fwkdl.Endpoint))
		}
	}

	return errors.Join(errs...)
}

func (ds *datastore) PodDelete(podName string) {
	ds.pods.Range(func(k, v any) bool {
		ep := v.(fwkdl.Endpoint)
		if ep.GetMetadata().PodName == podName {
			ds.pods.Delete(k)
			ds.epf.ReleaseEndpoint(ep)
		}
		return true
	})
}

func (ds *datastore) EndpointUpsert(ctx context.Context, meta *fwkdl.EndpointMetadata) {
	if _, err := ds.upsertEndpoint(ctx, meta); err != nil {
		log.FromContext(ctx).Error(err, "failed to register endpoint", "endpoint", meta.NamespacedName)
	}
}

func (ds *datastore) EndpointDelete(id types.NamespacedName) {
	if v, ok := ds.pods.LoadAndDelete(id); ok {
		ds.epf.ReleaseEndpoint(v.(fwkdl.Endpoint))
	}
}

// upsertEndpoint stores or updates a single endpoint in the pods map.
// Returns true if the endpoint was newly created, false if it already existed.
// Shared by EndpointUpsert and podUpdateOrAddIfNotExist.
//
// It returns an error wrapping errRegistrationDropped when the endpoint cannot be tracked:
// NewEndpoint returns nil when a collector is still registered for this endpoint or the
// collector failed to start. When a concurrent upsert has stored an entry for the key, this
// call's metadata is applied through the update path; when no entry exists (the upsert
// overlapped an in-flight delete that removed the entry before deregistering the collector,
// or the collector failed to start), the endpoint is untracked and the caller must retry.
func (ds *datastore) upsertEndpoint(ctx context.Context, meta *fwkdl.EndpointMetadata) (bool, error) {
	for {
		if existing, ok := ds.pods.Load(meta.NamespacedName); ok {
			ep := existing.(fwkdl.Endpoint)
			if ep.GetMetadata().Equal(meta) {
				return false, nil
			}
			ep.UpdateMetadata(meta)
			ds.epf.UpdateEndpoint(ctx, ep)
			return false, nil
		}
		ep := ds.epf.NewEndpoint(ds.parentCtx, meta)
		if ep == nil {
			if _, ok := ds.pods.Load(meta.NamespacedName); ok {
				// A concurrent upsert won the registration; apply this call's metadata through
				// the update path above.
				continue
			}
			return false, fmt.Errorf("endpoint %s: %w", meta.NamespacedName, errRegistrationDropped)
		}
		ds.pods.Store(meta.NamespacedName, ep)
		return true, nil
	}
}

func (ds *datastore) podResyncAll(ctx context.Context, reader client.Reader) error {
	logger := log.FromContext(ctx)
	podList := &corev1.PodList{}
	if err := reader.List(ctx, podList, &client.ListOptions{
		LabelSelector: ds.pool.Selector,
		Namespace:     ds.pool.Namespace,
	}); err != nil {
		return fmt.Errorf("failed to list pods - %w", err)
	}

	// Track active endpoints by their full name (including rank suffix).
	// This ensures orphaned rank endpoints are removed when targetPorts shrinks.
	activeEndpoints := sets.New[types.NamespacedName]()
	var errs []error
	for _, pod := range podList.Items {
		if !podutil.IsPodReady(&pod) {
			continue
		}
		// Calculate expected endpoint names based on current targetPorts.
		for idx := range ds.pool.TargetPorts {
			activeEndpoints.Insert(createEndpointNamespacedName(&pod, idx))
		}
		if err := ds.podUpdateOrAddIfNotExist(ctx, &pod, ds.pool); err != nil {
			// Propagate so PoolSet fails; needsResync makes the retried PoolSet resync again.
			errs = append(errs, err)
		}
	}

	// Remove endpoints that don't belong to the pool, are not ready, or are orphaned ranks.
	ds.pods.Range(func(k, v any) bool {
		ep := v.(fwkdl.Endpoint)
		endpointName := ep.GetMetadata().NamespacedName
		if !activeEndpoints.Has(endpointName) {
			logger.V(logutil.VERBOSE).Info("Removing endpoint", "endpoint", endpointName)
			ds.pods.Delete(k)
			ds.epf.ReleaseEndpoint(ep)
		}
		return true
	})

	return errors.Join(errs...)
}

// extractActivePorts extracts the active ports from a pod's annotations.
func extractActivePorts(pod *corev1.Pod, targetPorts []int) sets.Set[int] {
	allPorts := sets.New(targetPorts...)
	annotations := pod.GetAnnotations()
	portsAnnotation, ok := annotations[activePortsAnnotation]
	if !ok {
		portsAnnotation, ok = annotations[legacyGAIEActivePortsAnnotation]
		if !ok {
			return allPorts
		}
	}

	activePorts := sets.New[int]()
	portStrs := strings.SplitSeq(portsAnnotation, ",")
	for portStr := range portStrs {
		var portNum int
		_, err := fmt.Sscanf(strings.TrimSpace(portStr), "%d", &portNum)
		if err == nil && portNum > 0 && allPorts.Has(portNum) {
			activePorts.Insert(portNum)
		}
	}
	return activePorts
}

// createEndpointNamespacedName creates a namespaced name for an endpoint based on pod and rank index.
// This ensures consistent naming between PodUpdateOrAddIfNotExist and podResyncAll.
func createEndpointNamespacedName(pod *corev1.Pod, idx int) types.NamespacedName {
	return types.NamespacedName{
		Name:      pod.Name + "-rank-" + strconv.Itoa(idx),
		Namespace: pod.Namespace,
	}
}

func selectorEqual(a, b labels.Selector) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a.String() == b.String()
}
