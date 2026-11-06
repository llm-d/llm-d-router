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

// Package endpointslice provides an EndpointDiscovery implementation that
// builds the endpoint datastore from Kubernetes EndpointSlices instead of Pods.
package endpointslice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net"
	"os"
	"sort"
	"strconv"
	"sync"

	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	discoverylisters "k8s.io/client-go/listers/discovery/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const PluginType = "endpointslice-discovery"

// params is the user-facing configuration, unmarshalled from the plugin's
// "parameters" block in the EPP config.
type params struct {
	// Namespace of the EndpointSlices. Required.
	Namespace string `json:"namespace"`
	// Selector is a label selector for the slices, for example
	// "kubernetes.io/service-name=my-service". Empty selects every slice in
	// the namespace.
	Selector string `json:"selector"`
	// PortName is the slice port to route to. Empty requires the slice to
	// declare exactly one port.
	PortName string `json:"portName"`
	// Kubeconfig is an optional kubeconfig path. When empty the in-cluster
	// config is used, then $KUBECONFIG.
	Kubeconfig string `json:"kubeconfig"`
}

// EndpointSliceDiscovery implements EndpointDiscovery over EndpointSlices.
type EndpointSliceDiscovery struct {
	typedName fwkplugin.TypedName
	params    params
	selector  labels.Selector
	client    kubernetes.Interface

	// mu guards current, which DumpState reads concurrently with reconcile.
	mu      sync.RWMutex
	current map[types.NamespacedName]*fwkdl.EndpointMetadata

	ready     chan struct{}
	readyOnce sync.Once
}

var (
	_ fwkdl.EndpointDiscovery = (*EndpointSliceDiscovery)(nil)
	_ fwkplugin.StateDumper   = (*EndpointSliceDiscovery)(nil)
)

// Factory is the plugin factory for endpointslice-discovery.
func Factory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	p := params{}
	if parameters != nil {
		if err := parameters.Decode(&p); err != nil {
			return nil, fmt.Errorf("%s: failed to parse parameters: %w", PluginType, err)
		}
	}
	if p.Namespace == "" {
		return nil, errors.New(PluginType + ": 'namespace' parameter is required")
	}
	cfg, err := restConfig(p.Kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("%s: building client config: %w", PluginType, err)
	}
	client, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("%s: building client: %w", PluginType, err)
	}
	return newEndpointSliceDiscovery(name, p, client)
}

func newEndpointSliceDiscovery(name string, p params, client kubernetes.Interface) (*EndpointSliceDiscovery, error) {
	if p.Namespace == "" {
		return nil, errors.New(PluginType + ": 'namespace' parameter is required")
	}
	sel, err := labels.Parse(p.Selector)
	if err != nil {
		return nil, fmt.Errorf("%s: invalid 'selector' %q: %w", PluginType, p.Selector, err)
	}
	if name == "" {
		name = PluginType
	}
	return &EndpointSliceDiscovery{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		params:    p,
		selector:  sel,
		client:    client,
		current:   make(map[types.NamespacedName]*fwkdl.EndpointMetadata),
		ready:     make(chan struct{}),
	}, nil
}

// restConfig resolves the client config. The discovery path runs without a
// controller manager, so the plugin cannot borrow the manager's client.
func restConfig(kubeconfig string) (*rest.Config, error) {
	if kubeconfig != "" {
		return clientcmd.BuildConfigFromFlags("", kubeconfig)
	}
	if cfg, err := rest.InClusterConfig(); err == nil {
		return cfg, nil
	}
	if env := os.Getenv("KUBECONFIG"); env != "" {
		return clientcmd.BuildConfigFromFlags("", env)
	}
	return nil, errors.New("not running in a cluster and neither 'kubeconfig' nor $KUBECONFIG is set")
}

func (d *EndpointSliceDiscovery) TypedName() fwkplugin.TypedName { return d.typedName }

// Ready returns a channel closed after the first reconcile of the synced
// informer cache. See EndpointDiscovery.Ready for the contract.
func (d *EndpointSliceDiscovery) Ready() <-chan struct{} { return d.ready }

// Start watches the selected EndpointSlices and reconciles the datastore.
// Blocks until ctx is cancelled or the informer cache fails to sync.
func (d *EndpointSliceDiscovery) Start(ctx context.Context, notifier fwkdl.DiscoveryNotifier) error {
	logger := log.FromContext(ctx).WithValues("plugin", PluginType, "namespace", d.params.Namespace,
		"selector", d.params.Selector)

	factory := informers.NewSharedInformerFactoryWithOptions(d.client, 0,
		informers.WithNamespace(d.params.Namespace),
		informers.WithTweakListOptions(func(o *metav1.ListOptions) { o.LabelSelector = d.selector.String() }))
	sliceInformer := factory.Discovery().V1().EndpointSlices()
	lister := sliceInformer.Lister()

	// Informer handlers only signal a change. All notifier calls happen on
	// this goroutine because DiscoveryNotifier is not goroutine-safe.
	trigger := make(chan struct{}, 1)
	kick := func() {
		select {
		case trigger <- struct{}{}:
		default:
		}
	}
	if _, err := sliceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(any) { kick() },
		UpdateFunc: func(any, any) { kick() },
		DeleteFunc: func(any) { kick() },
	}); err != nil {
		return fmt.Errorf("%s: adding event handler: %w", PluginType, err)
	}

	factory.Start(ctx.Done())
	defer factory.Shutdown()
	if !cache.WaitForCacheSync(ctx.Done(), sliceInformer.Informer().HasSynced) {
		if ctx.Err() != nil {
			return nil
		}
		return fmt.Errorf("%s: EndpointSlice cache did not sync", PluginType)
	}

	d.reconcile(ctx, lister, notifier)
	d.readyOnce.Do(func() { close(d.ready) })
	logger.Info("initial EndpointSlice sync complete", "endpoints", d.count())

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-trigger:
			d.reconcile(ctx, lister, notifier)
		}
	}
}

// reconcile recomputes the endpoint set from the informer cache and applies
// the difference from the previous reconcile.
func (d *EndpointSliceDiscovery) reconcile(ctx context.Context, lister discoverylisters.EndpointSliceLister, notifier fwkdl.DiscoveryNotifier) {
	logger := log.FromContext(ctx).WithValues("plugin", PluginType)
	slices, err := lister.EndpointSlices(d.params.Namespace).List(d.selector)
	if err != nil {
		logger.Error(err, "failed to list EndpointSlices from cache")
		return
	}
	desired := d.endpointsFrom(ctx, slices)

	d.mu.Lock()
	previous := d.current
	d.current = desired
	d.mu.Unlock()

	ids := make([]types.NamespacedName, 0, len(desired))
	for id := range desired {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i].String() < ids[j].String() })
	for _, id := range ids {
		meta := desired[id]
		if old, ok := previous[id]; ok && sameEndpoint(old, meta) {
			continue
		}
		logger.V(logutil.DEFAULT).Info("upserting endpoint", "id", id.String(), "address", meta.Address, "port", meta.Port)
		notifier.Upsert(meta)
	}
	for id := range previous {
		if _, ok := desired[id]; !ok {
			logger.V(logutil.DEFAULT).Info("deleting endpoint", "id", id.String())
			notifier.Delete(id)
		}
	}
}

func (d *EndpointSliceDiscovery) endpointsFrom(ctx context.Context, slices []*discoveryv1.EndpointSlice) map[types.NamespacedName]*fwkdl.EndpointMetadata {
	logger := log.FromContext(ctx).WithValues("plugin", PluginType)
	sort.Slice(slices, func(i, j int) bool { return slices[i].Name < slices[j].Name })

	out := make(map[types.NamespacedName]*fwkdl.EndpointMetadata)
	for _, s := range slices {
		if s.AddressType != discoveryv1.AddressTypeIPv4 && s.AddressType != discoveryv1.AddressTypeIPv6 {
			logger.V(logutil.DEBUG).Info("skipping slice with unsupported addressType", "slice", s.Name, "addressType", s.AddressType)
			continue
		}
		port, ok := d.pickPort(s)
		if !ok {
			logger.V(logutil.DEBUG).Info("skipping slice without a matching port", "slice", s.Name, "portName", d.params.PortName)
			continue
		}
		svc := s.Labels[discoveryv1.LabelServiceName]
		if svc == "" {
			svc = s.Name
		}
		for _, ep := range s.Endpoints {
			if len(ep.Addresses) == 0 || !usable(ep) {
				continue
			}
			// The EndpointSlice API defines only the first address as meaningful.
			addr := ep.Addresses[0]
			key := addr
			if ep.Hostname != nil && *ep.Hostname != "" {
				key = *ep.Hostname
			}
			id := types.NamespacedName{Namespace: s.Namespace, Name: svc + "." + key}
			// An endpoint listed by two slices of one Service, as during slice
			// replacement, is reported once.
			if _, dup := out[id]; dup {
				continue
			}
			out[id] = &fwkdl.EndpointMetadata{
				ID:          id,
				Name:        id.Name,
				Address:     addr,
				Port:        port,
				MetricsHost: net.JoinHostPort(addr, port),
				Labels:      maps.Clone(s.Labels),
			}
		}
	}
	return out
}

func (d *EndpointSliceDiscovery) pickPort(s *discoveryv1.EndpointSlice) (string, bool) {
	if d.params.PortName == "" {
		if len(s.Ports) != 1 || s.Ports[0].Port == nil {
			return "", false
		}
		return strconv.Itoa(int(*s.Ports[0].Port)), true
	}
	for _, p := range s.Ports {
		if p.Port != nil && p.Name != nil && *p.Name == d.params.PortName {
			return strconv.Itoa(int(*p.Port)), true
		}
	}
	return "", false
}

func usable(ep discoveryv1.Endpoint) bool {
	if ep.Conditions.Terminating != nil && *ep.Conditions.Terminating {
		return false
	}
	// The EndpointSlice API asks consumers to treat an unset ready condition as ready.
	return ep.Conditions.Ready == nil || *ep.Conditions.Ready
}

func sameEndpoint(a, b *fwkdl.EndpointMetadata) bool {
	return a.Address == b.Address && a.Port == b.Port && a.MetricsHost == b.MetricsHost &&
		maps.Equal(a.Labels, b.Labels)
}

func (d *EndpointSliceDiscovery) count() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.current)
}

// maxDebugDumpEndpoints caps DumpState so the debug payload stays bounded.
const maxDebugDumpEndpoints = 100

// discoveryState is the snapshot returned by DumpState: endpoint identities
// only, never addresses or labels.
type discoveryState struct {
	TotalEndpoints int      `json:"totalEndpoints"`
	MaxEndpoints   int      `json:"maxEndpoints"`
	Endpoints      []string `json:"endpoints"`
}

// DumpState reports the endpoint identities from the last reconcile.
func (d *EndpointSliceDiscovery) DumpState() (json.RawMessage, error) {
	d.mu.RLock()
	names := make([]string, 0, len(d.current))
	for id := range d.current {
		names = append(names, id.String())
	}
	d.mu.RUnlock()

	sort.Strings(names)
	state := discoveryState{TotalEndpoints: len(names), MaxEndpoints: maxDebugDumpEndpoints}
	if len(names) > maxDebugDumpEndpoints {
		names = names[:maxDebugDumpEndpoints]
	}
	state.Endpoints = names
	return json.Marshal(state)
}
