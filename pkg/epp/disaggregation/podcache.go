package disaggregation

import (
	"context"
	"errors"
	"fmt"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	toolscache "k8s.io/client-go/tools/cache"
	ctrl "sigs.k8s.io/controller-runtime"
)

// PodCache tallies Ready pods matching a label selector, keyed by
// (revision label value, role label value). Safe for concurrent use.
//
// Backing storage is the controller-runtime Manager's shared Pod informer;
// the constructor attaches an event handler and rebuilds counts on every
// change. Cardinality is tiny (a handful of revisions × roles) so full
// recomputation is cheaper than incremental accounting and eliminates a
// class of double-count / stale-count bugs.
type PodCache struct {
	mutex            sync.Mutex
	pods             map[types.NamespacedName]*corev1.Pod
	counts           map[string]map[string]int
	revisionLabelKey string
	roleLabelKey     string
	namespace        string
	selector         labels.Selector
}

// NewPodCache attaches an event handler to the Manager's shared Pod informer
// so the cache reflects Ready pods matching (namespace, labelSelector). Empty
// roleLabelKey means "no role dimension" — the cache stores per-revision
// counts only and HasRoleForRevision always returns false.
func NewPodCache(ctx context.Context, mgr ctrl.Manager, namespace, labelSelector, revisionLabelKey, roleLabelKey string) (*PodCache, error) {
	pc, err := newPodCache(namespace, labelSelector, revisionLabelKey, roleLabelKey)
	if err != nil {
		return nil, err
	}
	informer, err := mgr.GetCache().GetInformer(ctx, &corev1.Pod{})
	if err != nil {
		return nil, fmt.Errorf("get pod informer: %w", err)
	}
	if _, err := informer.AddEventHandler(toolscache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { pc.upsert(obj) },
		UpdateFunc: func(_, obj interface{}) { pc.upsert(obj) },
		DeleteFunc: func(obj interface{}) { pc.remove(obj) },
	}); err != nil {
		return nil, fmt.Errorf("register informer event handler: %w", err)
	}
	return pc, nil
}

// newPodCache constructs an unattached cache. Tests seed via upsert/remove.
func newPodCache(namespace, labelSelector, revisionLabelKey, roleLabelKey string) (*PodCache, error) {
	if revisionLabelKey == "" {
		return nil, errors.New("revisionLabelKey is required")
	}
	sel, err := labels.Parse(labelSelector)
	if err != nil {
		return nil, fmt.Errorf("parse label selector %q: %w", labelSelector, err)
	}
	return &PodCache{
		pods:             make(map[types.NamespacedName]*corev1.Pod),
		counts:           make(map[string]map[string]int),
		revisionLabelKey: revisionLabelKey,
		roleLabelKey:     roleLabelKey,
		namespace:        namespace,
		selector:         sel,
	}, nil
}

// Count returns the number of Ready pods with the given (revision, role).
func (pc *PodCache) Count(revision, role string) int {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()
	return pc.counts[revision][role]
}

// HasRoleForRevision returns true when ≥1 Ready pod exists for (revision, role).
func (pc *PodCache) HasRoleForRevision(revision, role string) bool {
	return pc.Count(revision, role) > 0
}

// Revisions returns the set of revisions observed with ≥1 Ready pod.
func (pc *PodCache) Revisions() []string {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()
	out := make([]string, 0, len(pc.counts))
	for revision := range pc.counts {
		out = append(out, revision)
	}
	return out
}

func (pc *PodCache) upsert(obj interface{}) {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod.Namespace != pc.namespace || !pc.selector.Matches(labels.Set(pod.Labels)) {
		// Manager's shared informer sees every Pod in the namespace — filter
		// at the boundary so out-of-scope pods never enter our count map.
		return
	}
	pc.mutex.Lock()
	pc.pods[types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}] = pod
	previous := pc.counts
	pc.counts = pc.computeCountsLocked()
	pc.mutex.Unlock()
	publishGaugeDelta(previous, pc.counts)
}

func (pc *PodCache) remove(obj interface{}) {
	pod, _ := obj.(*corev1.Pod)
	if pod == nil {
		// Watch-relist can hand us a tombstone; unwrap the real pod.
		if tombstone, ok := obj.(toolscache.DeletedFinalStateUnknown); ok {
			pod, _ = tombstone.Obj.(*corev1.Pod)
		}
	}
	if pod == nil {
		return
	}
	pc.mutex.Lock()
	delete(pc.pods, types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name})
	previous := pc.counts
	pc.counts = pc.computeCountsLocked()
	pc.mutex.Unlock()
	publishGaugeDelta(previous, pc.counts)
}

func (pc *PodCache) computeCountsLocked() map[string]map[string]int {
	counts := make(map[string]map[string]int)
	for _, pod := range pc.pods {
		if !isPodReady(pod) {
			continue
		}
		revision := pod.Labels[pc.revisionLabelKey]
		if revision == "" {
			continue
		}
		role := ""
		if pc.roleLabelKey != "" {
			role = pod.Labels[pc.roleLabelKey]
			if role == "" {
				continue
			}
		}
		if _, ok := counts[revision]; !ok {
			counts[revision] = make(map[string]int)
		}
		counts[revision][role]++
	}
	return counts
}

// publishGaugeDelta emits Set for pairs whose count differs and Delete for
// pairs that vanished. Reset+Set-everything would leave unchanged pairs
// briefly showing zero, tripping scrape-window alerts.
func publishGaugeDelta(previous, current map[string]map[string]int) {
	for revision, perRole := range current {
		previousPerRole := previous[revision]
		for role, count := range perRole {
			if previousPerRole[role] != count {
				recordCacheSet(role, revision, count)
			}
		}
	}
	for revision, previousPerRole := range previous {
		currentPerRole := current[revision]
		for role := range previousPerRole {
			if _, stillPresent := currentPerRole[role]; !stillPresent {
				recordCacheDelete(role, revision)
			}
		}
	}
}

func isPodReady(pod *corev1.Pod) bool {
	if pod == nil || pod.Status.Phase != corev1.PodRunning || pod.DeletionTimestamp != nil {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}
