/*
Copyright 2026 Google LLC.

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

// Package endpointslice provides an EndpointDiscovery plugin that builds the
// EPP endpoint datastore from Kubernetes EndpointSlices instead of Pods.
//
// It targets environments where the pods backing an inference pool cannot be
// used for discovery, e.g. Project Exos: model servers run on Borneo linked
// runners with hostNetwork, so every pod reports the GKE control node's IP,
// while the Exos Control Plane (ECP) publishes the runners' real (external,
// IPv6) addresses as EndpointSlices of a selector-less headless Service.
//
// Behaviour:
//   - Watches EndpointSlices in one namespace that match a label selector
//     (typically kubernetes.io/service-name=<service>).
//   - One EPP endpoint per slice endpoint (first address only, per the
//     EndpointSlice API contract). The same endpoint appearing in several
//     slices (e.g. while a slice is being replaced) is reported once.
//   - Endpoints that are not ready, or are terminating, are skipped unless
//     includeNotReady is set. A nil "ready" condition counts as ready.
//   - Slice labels are copied onto the endpoint, optionally aliased through
//     labelMappings (e.g. exos/role -> llm-d.ai/role), plus staticLabels.
//   - All DiscoveryNotifier calls happen on the Start goroutine, in order.
//     Reconciliation is level-triggered: every informer event re-computes the
//     desired endpoint set from the informer cache and diffs it against what
//     was last reported.
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

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// PluginType is the type name used in the EPP config.
const PluginType = "exos-endpointslice-discovery"

// Params is the plugin's "parameters" block.
type Params struct {
	// Namespace of the EndpointSlices. Required.
	Namespace string `json:"namespace"`
	// LabelSelector selects the slices, e.g.
	// "kubernetes.io/service-name=glm52-server". Empty selects every slice in
	// the namespace.
	LabelSelector string `json:"labelSelector"`
	// PortName picks the slice port to route to. Empty means the slice must
	// have exactly one port.
	PortName string `json:"portName"`
	// Kubeconfig is an optional kubeconfig path. When empty the plugin uses
	// the in-cluster config, falling back to $KUBECONFIG.
	Kubeconfig string `json:"kubeconfig"`
	// IncludeNotReady also reports endpoints whose ready condition is false
	// or that are terminating.
	IncludeNotReady bool `json:"includeNotReady"`
	// LabelMappings copies a slice label to another endpoint label key, e.g.
	// {"exos/role": "llm-d.ai/role"} so the built-in prefill/decode filters
	// can read an Exos label.
	LabelMappings map[string]string `json:"labelMappings"`
	// StaticLabels are added to every endpoint, e.g.
	// {"llm-d.ai/engine-type": "sglang"} so the metrics extractor uses the
	// SGLang metric names.
	StaticLabels map[string]string `json:"staticLabels"`
	// AttributeAnnotations lists slice annotations to expose as endpoint
	// attributes, e.g. ["exos/k8s-fqdn"].
	AttributeAnnotations []string `json:"attributeAnnotations"`
}

// Discovery implements fwkdl.EndpointDiscovery over EndpointSlices.
type Discovery struct {
	typedName fwkplugin.TypedName
	params    Params
	selector  labels.Selector
	client    kubernetes.Interface

	mu      sync.RWMutex
	current map[types.NamespacedName]*fwkdl.EndpointMetadata

	ready     chan struct{}
	readyOnce sync.Once
}

var (
	_ fwkdl.EndpointDiscovery = (*Discovery)(nil)
	_ fwkplugin.StateDumper   = (*Discovery)(nil)
)

// Factory is the plugin factory registered with fwkplugin.Register.
func Factory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	p, err := decodeParams(parameters)
	if err != nil {
		return nil, err
	}
	cfg, err := restConfig(p.Kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("%s: building kube client config: %w", PluginType, err)
	}
	cs, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("%s: building kube client: %w", PluginType, err)
	}
	return New(name, p, cs)
}

// New builds the plugin with an explicit client (used by Factory and tests).
func New(name string, p Params, client kubernetes.Interface) (*Discovery, error) {
	if p.Namespace == "" {
		return nil, fmt.Errorf("%s: parameter namespace is required", PluginType)
	}
	sel, err := labels.Parse(p.LabelSelector)
	if err != nil {
		return nil, fmt.Errorf("%s: invalid labelSelector %q: %w", PluginType, p.LabelSelector, err)
	}
	if name == "" {
		name = PluginType
	}
	return &Discovery{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		params:    p,
		selector:  sel,
		client:    client,
		current:   map[types.NamespacedName]*fwkdl.EndpointMetadata{},
		ready:     make(chan struct{}),
	}, nil
}

func decodeParams(d *json.Decoder) (Params, error) {
	var p Params
	if d == nil {
		return p, fmt.Errorf("%s: parameters are required", PluginType)
	}
	if err := d.Decode(&p); err != nil {
		return p, fmt.Errorf("%s: decoding parameters: %w", PluginType, err)
	}
	return p, nil
}

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
	return nil, errors.New("not running in a cluster and neither parameters.kubeconfig nor $KUBECONFIG is set")
}

// TypedName implements fwkplugin.Plugin.
func (d *Discovery) TypedName() fwkplugin.TypedName { return d.typedName }

// Ready implements fwkdl.EndpointDiscovery.
func (d *Discovery) Ready() <-chan struct{} { return d.ready }

// Start implements fwkdl.EndpointDiscovery. It blocks until ctx is done.
func (d *Discovery) Start(ctx context.Context, notifier fwkdl.DiscoveryNotifier) error {
	logger := log.FromContext(ctx).WithValues("plugin", PluginType, "namespace", d.params.Namespace,
		"labelSelector", d.params.LabelSelector)

	factory := informers.NewSharedInformerFactoryWithOptions(d.client, 0,
		informers.WithNamespace(d.params.Namespace),
		informers.WithTweakListOptions(func(o *metav1.ListOptions) { o.LabelSelector = d.selector.String() }))
	sliceInformer := factory.Discovery().V1().EndpointSlices()
	lister := sliceInformer.Lister()

	// Coalescing trigger: informer events only signal "something changed";
	// the Start goroutine does all the work, which keeps notifier calls on a
	// single goroutine as the DiscoveryNotifier contract requires.
	trigger := make(chan struct{}, 1)
	kick := func(any) {
		select {
		case trigger <- struct{}{}:
		default:
		}
	}
	if _, err := sliceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    kick,
		UpdateFunc: func(_, n any) { kick(n) },
		DeleteFunc: kick,
	}); err != nil {
		return fmt.Errorf("%s: adding event handler: %w", PluginType, err)
	}

	factory.Start(ctx.Done())
	defer factory.Shutdown()
	if !cache.WaitForCacheSync(ctx.Done(), sliceInformer.Informer().HasSynced) {
		if ctx.Err() != nil {
			return nil
		}
		return fmt.Errorf("%s: EndpointSlice cache never synced", PluginType)
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

type sliceLister interface {
	List(selector labels.Selector) ([]*discoveryv1.EndpointSlice, error)
}

// reconcile recomputes the desired endpoints and reports the delta.
func (d *Discovery) reconcile(ctx context.Context, lister discoverylisters.EndpointSliceLister, notifier fwkdl.DiscoveryNotifier) {
	d.reconcileFrom(ctx, lister.EndpointSlices(d.params.Namespace), notifier)
}

func (d *Discovery) reconcileFrom(ctx context.Context, l sliceLister, notifier fwkdl.DiscoveryNotifier) {
	logger := log.FromContext(ctx).WithValues("plugin", PluginType)
	slices, err := l.List(d.selector)
	if err != nil {
		logger.Error(err, "listing EndpointSlices from cache")
		return
	}
	desired := d.desired(ctx, slices)

	d.mu.Lock()
	previous := d.current
	d.current = desired
	d.mu.Unlock()

	// Deterministic order makes logs and tests stable.
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
		logger.Info("upserting endpoint", "id", id.String(), "address", meta.Address, "port", meta.Port,
			"labels", meta.Labels)
		notifier.Upsert(meta)
	}
	for id := range previous {
		if _, ok := desired[id]; !ok {
			logger.Info("deleting endpoint", "id", id.String())
			notifier.Delete(id)
		}
	}
}

func (d *Discovery) desired(ctx context.Context, slices []*discoveryv1.EndpointSlice) map[types.NamespacedName]*fwkdl.EndpointMetadata {
	logger := log.FromContext(ctx).WithValues("plugin", PluginType)
	sort.Slice(slices, func(i, j int) bool { return slices[i].Name < slices[j].Name })

	out := map[types.NamespacedName]*fwkdl.EndpointMetadata{}
	for _, s := range slices {
		if s.AddressType != discoveryv1.AddressTypeIPv4 && s.AddressType != discoveryv1.AddressTypeIPv6 {
			logger.V(1).Info("skipping slice with unsupported addressType", "slice", s.Name, "addressType", s.AddressType)
			continue
		}
		port, ok := d.pickPort(s)
		if !ok {
			logger.V(1).Info("skipping slice without a matching port", "slice", s.Name, "portName", d.params.PortName)
			continue
		}
		svc := s.Labels[discoveryv1.LabelServiceName]
		if svc == "" {
			svc = s.Name
		}
		for _, ep := range s.Endpoints {
			if len(ep.Addresses) == 0 || !d.usable(ep) {
				continue
			}
			addr := ep.Addresses[0]
			key := addr
			if ep.Hostname != nil && *ep.Hostname != "" {
				key = *ep.Hostname
			}
			id := types.NamespacedName{Namespace: s.Namespace, Name: svc + "." + key}
			if _, dup := out[id]; dup {
				continue // same endpoint published by two slices
			}
			out[id] = &fwkdl.EndpointMetadata{
				ID:          id,
				Name:        id.Name,
				Address:     addr,
				Port:        port,
				MetricsHost: net.JoinHostPort(addr, port),
				Labels:      d.endpointLabels(s),
				Attributes:  d.endpointAttributes(s),
			}
		}
	}
	return out
}

func (d *Discovery) pickPort(s *discoveryv1.EndpointSlice) (string, bool) {
	for _, p := range s.Ports {
		if p.Port == nil {
			continue
		}
		name := ""
		if p.Name != nil {
			name = *p.Name
		}
		if d.params.PortName == "" && len(s.Ports) == 1 || d.params.PortName != "" && name == d.params.PortName {
			return strconv.Itoa(int(*p.Port)), true
		}
	}
	return "", false
}

func (d *Discovery) usable(ep discoveryv1.Endpoint) bool {
	if d.params.IncludeNotReady {
		return true
	}
	if ep.Conditions.Terminating != nil && *ep.Conditions.Terminating {
		return false
	}
	// Per the EndpointSlice API, a nil ready condition is "unknown" and
	// consumers should treat it as ready.
	return ep.Conditions.Ready == nil || *ep.Conditions.Ready
}

func (d *Discovery) endpointLabels(s *discoveryv1.EndpointSlice) map[string]string {
	l := make(map[string]string, len(s.Labels)+len(d.params.StaticLabels)+len(d.params.LabelMappings))
	maps.Copy(l, s.Labels)
	for from, to := range d.params.LabelMappings {
		if v, ok := s.Labels[from]; ok {
			l[to] = v
		}
	}
	maps.Copy(l, d.params.StaticLabels)
	return l
}

func (d *Discovery) endpointAttributes(s *discoveryv1.EndpointSlice) map[string]string {
	if len(d.params.AttributeAnnotations) == 0 {
		return nil
	}
	a := map[string]string{}
	for _, k := range d.params.AttributeAnnotations {
		if v, ok := s.Annotations[k]; ok {
			a[k] = v
		}
	}
	return a
}

func sameEndpoint(a, b *fwkdl.EndpointMetadata) bool {
	return a.Address == b.Address && a.Port == b.Port && a.MetricsHost == b.MetricsHost &&
		maps.Equal(a.Labels, b.Labels) && maps.Equal(a.Attributes, b.Attributes)
}

func (d *Discovery) count() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.current)
}

type dumpedEndpoint struct {
	ID          string            `json:"id"`
	Address     string            `json:"address"`
	Port        string            `json:"port"`
	MetricsHost string            `json:"metricsHost"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// DumpState implements fwkplugin.StateDumper (served on the metrics port at
// /debug/plugins/state). This is a PoC: it includes addresses and labels.
func (d *Discovery) DumpState() (json.RawMessage, error) {
	d.mu.RLock()
	out := make([]dumpedEndpoint, 0, len(d.current))
	for id, m := range d.current {
		out = append(out, dumpedEndpoint{ID: id.String(), Address: m.Address, Port: m.Port,
			MetricsHost: m.MetricsHost, Labels: m.Labels})
	}
	d.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return json.Marshal(map[string]any{"totalEndpoints": len(out), "endpoints": out})
}
