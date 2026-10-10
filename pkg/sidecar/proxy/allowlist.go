/*
Copyright 2025 The llm-d Authors.

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

/*
Copyright 2025 The llm-d Authors

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

package proxy

import (
	"context"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/utils/set"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

const (
	inferencePoolResource = "inferencepools"
	resyncPeriod          = 30 * time.Second
)

// AllowlistValidator manages allowed prefill targets based on InferencePool resources
type AllowlistValidator struct {
	logger        logr.Logger
	dynamicClient dynamic.Interface
	namespace     string
	poolName      string
	enabled       bool

	gvr schema.GroupVersionResource // detected GVR

	// allowedTargets maps hostport -> bool for allowed prefill targets
	allowedTargets   set.Set[string]
	allowedTargetsMu sync.RWMutex

	// watchers for cleanup
	poolInformer   cache.SharedInformer
	podInformers   map[string]cache.SharedInformer
	podStopChans   map[string]chan struct{} // individual stop channels for pod informers
	poolPorts      map[string][]string      // target ports per pool
	podInformersMu sync.RWMutex
	stopCh         chan struct{}
}

// NewAllowlistValidator creates a new SSRF protection validator
func NewAllowlistValidator(enabled bool, poolGroup, namespace, poolName string) (*AllowlistValidator, error) {
	if !enabled {
		return &AllowlistValidator{
			enabled: false,
		}, nil
	}

	if poolGroup != routing.InferencePoolAPIGroup {
		return nil, fmt.Errorf("pool-group must be %q, got %q", routing.InferencePoolAPIGroup, poolGroup)
	}

	gvr := schema.GroupVersionResource{
		Group:    routing.InferencePoolAPIGroup,
		Version:  "v1",
		Resource: inferencePoolResource,
	}

	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	overrides := &clientcmd.ConfigOverrides{}
	config, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		loadingRules,
		overrides,
	).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get Kubernetes config (ensure running in a pod with proper RBAC): %w", err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create Kubernetes dynamic client: %w", err)
	}

	return &AllowlistValidator{
		enabled:        true,
		dynamicClient:  dynamicClient,
		namespace:      namespace,
		poolName:       poolName,
		gvr:            gvr,
		allowedTargets: set.New[string](),
		podInformers:   make(map[string]cache.SharedInformer),
		podStopChans:   make(map[string]chan struct{}),
		poolPorts:      make(map[string][]string),
		stopCh:         make(chan struct{}),
	}, nil
}

// Start begins watching InferencePool resources and managing the allowlist
func (av *AllowlistValidator) Start(ctx context.Context) error {
	if !av.enabled {
		return nil
	}

	av.logger = log.FromContext(ctx).WithName("allowlist-validator")
	av.logger.Info("starting SSRF protection allowlist validator",
		"namespace", av.namespace, "poolName", av.poolName, "gvr", av.gvr.String())

	// Create informer for the specific InferencePool resource
	lw := &cache.ListWatch{
		ListWithContextFunc: func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
			// List with field selector to get only the specific InferencePool
			options.FieldSelector = "metadata.name=" + av.poolName
			return av.dynamicClient.Resource(av.gvr).Namespace(av.namespace).List(ctx, options)
		},
		WatchFuncWithContext: func(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
			// Watch the specific InferencePool by name using field selector
			options.FieldSelector = "metadata.name=" + av.poolName
			return av.dynamicClient.Resource(av.gvr).Namespace(av.namespace).Watch(ctx, options)
		},
	}

	av.poolInformer = cache.NewSharedInformer(lw, &unstructured.Unstructured{}, resyncPeriod)

	// Add event handlers
	_, _ = av.poolInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    av.onInferencePoolAdd,
		UpdateFunc: av.onInferencePoolUpdate,
		DeleteFunc: av.onInferencePoolDelete,
	})

	// Start the informer
	go av.poolInformer.Run(av.stopCh)

	// Wait for cache sync
	if !cache.WaitForCacheSync(av.stopCh, av.poolInformer.HasSynced) {
		return fmt.Errorf("failed to sync InferencePool cache within timeout (check RBAC permissions for inferencepools.%s and that pool '%s' exists)", av.gvr.String(), av.poolName)
	}

	av.logger.Info("allowlist validator started successfully")
	return nil
}

// Stop stops all watchers and cleans up resources
func (av *AllowlistValidator) Stop() {
	if !av.enabled {
		return
	}

	av.logger.Info("stopping allowlist validator")

	// Stop all pod informers first
	av.podInformersMu.Lock()
	for poolName, stopCh := range av.podStopChans {
		av.logger.V(logging.DEBUG).Info("stopping pod informer", "pool", poolName)
		close(stopCh)
	}
	// Clear the maps
	av.podStopChans = make(map[string]chan struct{})
	av.podInformers = make(map[string]cache.SharedInformer)
	av.podInformersMu.Unlock()

	// Stop the main pool informer
	close(av.stopCh)
}

// IsAllowed checks if a given host:port combination is in the allowlist
func (av *AllowlistValidator) IsAllowed(hostPort string) bool {
	if !av.enabled {
		// If SSRF protection is disabled, allow all requests (backward compatibility)
		return true
	}

	hostPort, _ = strings.CutPrefix(hostPort, "http://")

	av.allowedTargetsMu.RLock()
	defer av.allowedTargetsMu.RUnlock()

	allowed := av.allowedTargets.Has(hostPort)
	av.logger.V(logging.DEBUG).Info("allowlist check", "hostPort", hostPort, "allowed", allowed)
	return allowed
}

// onInferencePoolAdd handles new InferencePool resources
func (av *AllowlistValidator) onInferencePoolAdd(obj interface{}) {
	pool := obj.(*unstructured.Unstructured)
	av.logger.Info("InferencePool added", "name", pool.GetName())
	av.updatePodsForPool(pool)
}

// onInferencePoolUpdate handles updated InferencePool resources
func (av *AllowlistValidator) onInferencePoolUpdate(_, newObj interface{}) {
	pool := newObj.(*unstructured.Unstructured)
	av.logger.Info("InferencePool updated", "name", pool.GetName())
	av.updatePodsForPool(pool)
}

// onInferencePoolDelete handles deleted InferencePool resources
func (av *AllowlistValidator) onInferencePoolDelete(obj interface{}) {
	name, err := cache.DeletionHandlingObjectToName(obj)
	if err != nil {
		av.logger.Error(err, "failed to process InferencePool deletion", "objectType", fmt.Sprintf("%T", obj))
		return
	}
	poolName := name.Name
	av.logger.Info("InferencePool deleted", "name", poolName)

	// Stop watching pods for this pool
	av.podInformersMu.Lock()
	if stopCh, exists := av.podStopChans[poolName]; exists {
		close(stopCh) // properly stop the informer
		delete(av.podStopChans, poolName)
	}
	delete(av.podInformers, poolName)
	delete(av.poolPorts, poolName)
	av.podInformersMu.Unlock()

	// Remove targets associated with this pool (simplified - removes all and rebuilds)
	av.rebuildAllowlist()
}

// updatePodsForPool starts or updates pod watching for a specific InferencePool
func (av *AllowlistValidator) updatePodsForPool(poolObj *unstructured.Unstructured) {
	poolName := poolObj.GetName()

	selector, err := av.poolSelector(poolObj)
	if err != nil {
		av.logger.Error(err, "failed to extract selector from InferencePool", "name", poolName)
		return
	}

	ports, err := av.poolTargetPorts(poolObj)
	if err != nil {
		av.logger.Error(err, "failed to extract target ports from InferencePool", "name", poolName)
		return
	}

	av.createPodInformer(poolName, selector, ports)
}

func (av *AllowlistValidator) poolSelector(poolObj *unstructured.Unstructured) (labels.Selector, error) {
	spec, found, err := unstructured.NestedMap(poolObj.Object, "spec")
	if err != nil || !found {
		return nil, fmt.Errorf("missing or invalid spec field (found=%t): %w", found, err)
	}

	selectorData, found, err := unstructured.NestedStringMap(spec, "selector", "matchLabels")
	if err != nil || !found {
		return nil, fmt.Errorf("missing or invalid spec.selector.matchLabels field (found=%t): %w", found, err)
	}

	return labels.Set(selectorData).AsSelector(), nil
}

func (av *AllowlistValidator) poolTargetPorts(poolObj *unstructured.Unstructured) ([]string, error) {
	// GA API uses spec.targetPorts[].number; deprecated alpha API uses spec.targetPortNumber.
	if av.gvr.Group != routing.InferencePoolAPIGroup {
		port, found, err := unstructured.NestedInt64(poolObj.Object, "spec", "targetPortNumber")
		if err != nil || !found {
			return nil, fmt.Errorf("missing or invalid spec.targetPortNumber (found=%t): %w", found, err)
		}
		return []string{strconv.FormatInt(port, 10)}, nil
	}

	targetPorts, found, err := unstructured.NestedSlice(poolObj.Object, "spec", "targetPorts")
	if err != nil || !found || len(targetPorts) == 0 {
		return nil, fmt.Errorf("missing or invalid spec.targetPorts (found=%t): %w", found, err)
	}
	// One malformed entry skips the whole pool, as with an unreadable selector. The CRD schema
	// requires every number in 1-65535, so this only fires against a CRD without that validation.
	ports := make([]string, 0, len(targetPorts))
	for i, tp := range targetPorts {
		tpMap, ok := tp.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid spec.targetPorts[%d]", i)
		}
		port, found, err := unstructured.NestedInt64(tpMap, "number")
		if err != nil || !found {
			return nil, fmt.Errorf("missing or invalid spec.targetPorts[%d].number (found=%t): %w", i, found, err)
		}
		ports = append(ports, strconv.FormatInt(port, 10))
	}
	return ports, nil
}

// createPodInformer creates a new pod informer for the given selector
func (av *AllowlistValidator) createPodInformer(poolName string, selector labels.Selector, ports []string) {
	av.podInformersMu.Lock()
	defer av.podInformersMu.Unlock()

	// Stop existing informer if it exists
	if _, exists := av.podInformers[poolName]; exists {
		if stopCh, stopExists := av.podStopChans[poolName]; stopExists {
			close(stopCh) // stop the existing informer
			delete(av.podStopChans, poolName)
		}
		delete(av.podInformers, poolName)
	}

	// Create new pod informer
	podLW := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) { //nolint:staticcheck // SA1019
			options.LabelSelector = selector.String()
			return av.dynamicClient.Resource(schema.GroupVersionResource{
				Group:    "",
				Version:  "v1",
				Resource: "pods",
			}).Namespace(av.namespace).List(context.TODO(), options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) { //nolint:staticcheck // SA1019
			options.LabelSelector = selector.String()
			return av.dynamicClient.Resource(schema.GroupVersionResource{
				Group:    "",
				Version:  "v1",
				Resource: "pods",
			}).Namespace(av.namespace).Watch(context.TODO(), options)
		},
	}

	podInformer := cache.NewSharedInformer(podLW, &unstructured.Unstructured{}, resyncPeriod)

	// Add event handlers
	_, _ = podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    av.onPodAdd,
		UpdateFunc: av.onPodUpdate,
		DeleteFunc: av.onPodDelete,
	})

	// Create individual stop channel for this informer
	podStopCh := make(chan struct{})

	av.podInformers[poolName] = podInformer
	av.podStopChans[poolName] = podStopCh
	av.poolPorts[poolName] = ports

	// Start the informer with its own stop channel
	go podInformer.Run(podStopCh)
}

// onPodAdd handles new pods matching our selectors
func (av *AllowlistValidator) onPodAdd(obj interface{}) {
	pod := obj.(*unstructured.Unstructured)
	podIP, _, _ := unstructured.NestedString(pod.Object, "status", "podIP")
	av.logger.V(logging.DEBUG).Info("Pod added", "name", pod.GetName(), "ip", podIP)
	av.rebuildAllowlist()
}

// onPodUpdate handles updated pods
func (av *AllowlistValidator) onPodUpdate(_, newObj interface{}) {
	pod := newObj.(*unstructured.Unstructured)
	podIP, _, _ := unstructured.NestedString(pod.Object, "status", "podIP")
	av.logger.V(logging.DEBUG).Info("Pod updated", "name", pod.GetName(), "ip", podIP)
	av.rebuildAllowlist()
}

// onPodDelete handles deleted pods
func (av *AllowlistValidator) onPodDelete(obj interface{}) {
	name, err := cache.DeletionHandlingObjectToName(obj)
	if err != nil {
		av.logger.Error(err, "failed to process Pod deletion", "objectType", fmt.Sprintf("%T", obj))
		return
	}
	av.logger.V(logging.DEBUG).Info("Pod deleted", "name", name.Name)
	av.rebuildAllowlist()
}

// rebuildAllowlist rebuilds the entire allowlist from current pod state
func (av *AllowlistValidator) rebuildAllowlist() {
	av.allowedTargetsMu.Lock()
	defer av.allowedTargetsMu.Unlock()

	// Clear existing allowlist
	av.allowedTargets = set.New[string]()

	av.podInformersMu.RLock()
	defer av.podInformersMu.RUnlock()
	// Rebuild from all pod informers
	for poolName, informer := range av.podInformers {
		store := informer.GetStore()
		for _, obj := range store.List() {
			pod := obj.(*unstructured.Unstructured)

			// Get pod phase and IP
			podIP, _, _ := unstructured.NestedString(pod.Object, "status", "podIP")

			// Only include pods with valid IPs
			if podIP != "" {
				// Add both IP and hostname variants
				av.addPodToAllowlist(pod, poolName, av.poolPorts[poolName])
			}
		}
	}

	av.logger.Info("rebuilt allowlist", "targetCount", len(av.allowedTargets), "targets", av.allowedTargets)
}

// addPodToAllowlist adds a host:port entry for each of the pod's addresses on each pool target port
func (av *AllowlistValidator) addPodToAllowlist(pod *unstructured.Unstructured, poolName string, ports []string) {
	podIP, _, _ := unstructured.NestedString(pod.Object, "status", "podIP")
	podName := pod.GetName()
	for _, port := range ports {
		if podIP != "" {
			av.allowedTargets.Insert(net.JoinHostPort(podIP, port))
		}
		if podName != "" {
			av.allowedTargets.Insert(net.JoinHostPort(podName, port))
		}
	}

	av.logger.V(logging.TRACE).Info("added pod to allowlist", "pod", podName, "ip", podIP, "ports", ports, "pool", poolName)
}
