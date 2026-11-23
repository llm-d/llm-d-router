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
//
// The plugin owns no Kubernetes client or watch. It registers an EndpointSlice
// notification source scoped to its namespace and selector; the datalayer
// runtime binds that source to a controller manager and delivers deep-copied
// slice events to the plugin's extractor. The plugin only maps slices to
// endpoints.
package endpointslice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net"
	"sort"
	"strconv"
	"sync"

	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
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
}

// EndpointSliceDiscovery implements EndpointDiscovery over EndpointSlices.
type EndpointSliceDiscovery struct {
	typedName fwkplugin.TypedName
	namespace string
	selector  labels.Selector
	portName  string

	// events carries slice changes from the extractor to Start, which applies
	// them on its own goroutine because DiscoveryNotifier is not
	// goroutine-safe.
	events chan sliceEvent
	// stopped is closed when Start returns, so the extractor never blocks on
	// a plugin that is no longer running.
	stopped     chan struct{}
	stoppedOnce sync.Once

	// slices is the current set of selected slices, keyed by name. Only the
	// Start goroutine touches it.
	slices map[types.NamespacedName]*discoveryv1.EndpointSlice

	// endpoints holds the endpoints last applied to the datastore, keyed by
	// endpoint ID. reconcile diffs the newly computed set against it. mu
	// guards it because DumpState reads it concurrently with reconcile.
	mu        sync.RWMutex
	endpoints map[types.NamespacedName]*fwkdl.EndpointMetadata

	ready     chan struct{}
	readyOnce sync.Once
}

// sliceEvent is one slice change, produced by the extractor and applied by
// Start. done is closed once the change is applied to the datastore.
type sliceEvent struct {
	kind  fwkdl.EventType
	key   types.NamespacedName
	slice *discoveryv1.EndpointSlice // set for EventAddOrUpdate
	done  chan struct{}
}

var (
	_ fwkdl.EndpointDiscovery = (*EndpointSliceDiscovery)(nil)
	_ fwkdl.Registrant        = (*EndpointSliceDiscovery)(nil)
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
	return newEndpointSliceDiscovery(name, p)
}

func newEndpointSliceDiscovery(name string, p params) (*EndpointSliceDiscovery, error) {
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
		namespace: p.Namespace,
		selector:  sel,
		portName:  p.PortName,
		events:    make(chan sliceEvent),
		stopped:   make(chan struct{}),
		slices:    make(map[types.NamespacedName]*discoveryv1.EndpointSlice),
		endpoints: make(map[types.NamespacedName]*fwkdl.EndpointMetadata),
		ready:     make(chan struct{}),
	}, nil
}

func (d *EndpointSliceDiscovery) TypedName() fwkplugin.TypedName { return d.typedName }

// RegisterDependencies registers the EndpointSlice notification source,
// scoped to the configured namespace and selector, and the extractor that
// hands slice events to Start. The datalayer runtime owns the watch.
func (d *EndpointSliceDiscovery) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      d.typedName,
		SourceType: notifications.NotificationSourceType,
		Extractor:  &sliceExtractor{plugin: d},
		DefaultSource: notifications.NewK8sNotificationSource(
			notifications.NotificationSourceType,
			d.typedName.Name+"/endpointslice",
			endpointSliceGVK,
			notifications.WithNamespace(d.namespace),
			notifications.WithLabelSelector(d.selector),
		),
	})
}

// Ready returns a channel closed once Start is applying slice events. The
// runner additionally waits for the notification source's initial list to be
// processed, and the extractor returns only after its event is applied, so
// the datastore holds the initial endpoints before serving starts.
func (d *EndpointSliceDiscovery) Ready() <-chan struct{} { return d.ready }

// Start applies slice events from the extractor to notifier until ctx is
// cancelled.
func (d *EndpointSliceDiscovery) Start(ctx context.Context, notifier fwkdl.DiscoveryNotifier) error {
	defer d.stoppedOnce.Do(func() { close(d.stopped) })
	d.readyOnce.Do(func() { close(d.ready) })
	for {
		select {
		case <-ctx.Done():
			return nil
		case ev := <-d.events:
			switch ev.kind {
			case fwkdl.EventAddOrUpdate:
				d.slices[ev.key] = ev.slice
			case fwkdl.EventDelete:
				delete(d.slices, ev.key)
			}
			d.reconcile(ctx, notifier)
			close(ev.done)
		}
	}
}

// emit hands an event to Start and waits until Start has applied it, so a
// successful Extract means the datastore reflects the slice.
func (d *EndpointSliceDiscovery) emit(ctx context.Context, ev sliceEvent) error {
	ev.done = make(chan struct{})
	select {
	case d.events <- ev:
	case <-d.stopped:
		return fmt.Errorf("%s: discovery is not running", PluginType)
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case <-ev.done:
		return nil
	case <-d.stopped:
		return fmt.Errorf("%s: discovery stopped before applying slice %s", PluginType, ev.key)
	case <-ctx.Done():
		return ctx.Err()
	}
}

// reconcile recomputes the endpoint set from the current slices and applies
// the difference from the previous reconcile.
func (d *EndpointSliceDiscovery) reconcile(ctx context.Context, notifier fwkdl.DiscoveryNotifier) {
	logger := log.FromContext(ctx).WithValues("plugin", PluginType)
	slices := make([]*discoveryv1.EndpointSlice, 0, len(d.slices))
	for _, s := range d.slices {
		slices = append(slices, s)
	}
	desired := d.endpointsFrom(ctx, slices)

	d.mu.Lock()
	previous := d.endpoints
	d.endpoints = desired
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
			logger.V(logutil.DEBUG).Info("skipping slice without a matching port", "slice", s.Name, "portName", d.portName)
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
	if d.portName == "" {
		if len(s.Ports) != 1 || s.Ports[0].Port == nil {
			return "", false
		}
		return strconv.Itoa(int(*s.Ports[0].Port)), true
	}
	for _, p := range s.Ports {
		if p.Port != nil && p.Name != nil && *p.Name == d.portName {
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
	names := make([]string, 0, len(d.endpoints))
	for id := range d.endpoints {
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
