package disaggregation

import (
	"context"
	"fmt"
	"sync"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

// PodCache tallies Ready pods observed by an informer, keyed by
// (revision label value, role label value). It is safe for concurrent use.
//
// The design deliberately rebuilds the count map on every informer event rather
// than doing incremental accounting. The cardinality is tiny (a handful of
// revisions × a handful of roles) and full recomputation eliminates a whole
// class of double-count / stale-count bugs.
type PodCache struct {
	mutex            sync.RWMutex
	counts           map[string]map[string]int
	revisionLabelKey string
	roleLabelKey     string
	informer         cache.SharedIndexInformer
	factory          informers.SharedInformerFactory
}

// NewPodCache builds an informer scoped to namespace + labelSelector and
// returns a cache that will track Ready pods once Start is called. An empty
// roleLabelKey is allowed and means "no role dimension" — the cache stores
// per-revision counts only and HasRoleForRevision always returns false
// (callers should skip role-based checks in that mode).
func NewPodCache(client kubernetes.Interface, namespace, labelSelector, revisionLabelKey, roleLabelKey string) (*PodCache, error) {
	if revisionLabelKey == "" {
		return nil, fmt.Errorf("revisionLabelKey is required")
	}
	factory := informers.NewSharedInformerFactoryWithOptions(
		client,
		0,
		informers.WithNamespace(namespace),
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.LabelSelector = labelSelector
		}),
	)
	informer := factory.Core().V1().Pods().Informer()
	podCache := &PodCache{
		counts:           make(map[string]map[string]int),
		revisionLabelKey: revisionLabelKey,
		roleLabelKey:     roleLabelKey,
		informer:         informer,
		factory:          factory,
	}
	_, err := informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(interface{}) { podCache.rebuild() },
		UpdateFunc: func(_, _ interface{}) { podCache.rebuild() },
		DeleteFunc: func(interface{}) { podCache.rebuild() },
	})
	if err != nil {
		return nil, fmt.Errorf("register informer event handler: %w", err)
	}
	return podCache, nil
}

// Start begins the informer factory. Runs until ctx is done.
func (podCache *PodCache) Start(ctx context.Context) {
	podCache.factory.Start(ctx.Done())
}

// WaitForCacheSync blocks until the informer's initial list has completed or ctx
// is cancelled. Returns true on successful sync.
func (podCache *PodCache) WaitForCacheSync(ctx context.Context) bool {
	synced := cache.WaitForCacheSync(ctx.Done(), podCache.informer.HasSynced)
	if synced {
		podCache.rebuild()
	}
	return synced
}

// Count returns the number of Ready pods with the given (revision, role).
func (podCache *PodCache) Count(revision, role string) int {
	podCache.mutex.RLock()
	defer podCache.mutex.RUnlock()
	if perRole, ok := podCache.counts[revision]; ok {
		return perRole[role]
	}
	return 0
}

// HasRoleForRevision returns true when at least one Ready pod exists for the
// given (revision, role).
func (podCache *PodCache) HasRoleForRevision(revision, role string) bool {
	return podCache.Count(revision, role) > 0
}

// rolesForRevision returns the set of roles observed with ≥1 Ready pod for
// the given revision. Iteration order is unspecified. Unexported because no
// production caller needs it — the accessor exists for tests only.
func (podCache *PodCache) rolesForRevision(revision string) map[string]int {
	podCache.mutex.RLock()
	defer podCache.mutex.RUnlock()
	source, ok := podCache.counts[revision]
	if !ok {
		return nil
	}
	out := make(map[string]int, len(source))
	for role, count := range source {
		out[role] = count
	}
	return out
}

// Revisions returns the set of revisions observed with ≥1 Ready pod for any role.
func (podCache *PodCache) Revisions() []string {
	podCache.mutex.RLock()
	defer podCache.mutex.RUnlock()
	out := make([]string, 0, len(podCache.counts))
	for revision := range podCache.counts {
		out = append(out, revision)
	}
	return out
}

func (podCache *PodCache) rebuild() {
	all := podCache.informer.GetStore().List()
	counts := make(map[string]map[string]int)
	for _, obj := range all {
		pod, ok := obj.(*corev1.Pod)
		if !ok || !isPodReady(pod) {
			continue
		}
		revision := pod.Labels[podCache.revisionLabelKey]
		if revision == "" {
			continue
		}
		role := ""
		if podCache.roleLabelKey != "" {
			role = pod.Labels[podCache.roleLabelKey]
			if role == "" {
				// Pod is missing the required role label — skip so it
				// doesn't accidentally satisfy a HasRoleForRevision check.
				continue
			}
		}
		if _, ok := counts[revision]; !ok {
			counts[revision] = make(map[string]int)
		}
		counts[revision][role]++
	}

	podCache.mutex.Lock()
	previous := podCache.counts
	podCache.counts = counts
	podCache.mutex.Unlock()

	// Publish only the deltas. Reset+Set-everything would leave the gauge
	// briefly showing zero for pairs whose count didn't change, which trips
	// scrape-window alerts. Diffing keeps unchanged series untouched.
	publishGaugeDelta(previous, counts)
}

// publishGaugeDelta walks two count maps and emits Set for pairs whose count
// differs and Delete for pairs that vanished from the new map. Pairs whose
// count is unchanged are not touched.
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
	if pod == nil {
		return false
	}
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}
	if pod.DeletionTimestamp != nil {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}
