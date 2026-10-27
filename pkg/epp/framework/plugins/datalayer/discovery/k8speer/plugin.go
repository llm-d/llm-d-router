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

// Package k8speer provides a PeerDiscovery plugin that watches this EPP
// deployment's own Pods through the datalayer notification runtime.
package k8speer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	podutil "github.com/llm-d/llm-d-router/pkg/epp/util/pod"
)

const PluginType = "k8s-peer-discovery"

type params struct {
	// Selector matches this EPP deployment's own pods (its peers), in
	// Kubernetes label selector syntax (e.g. "app=my-epp"). Required.
	Selector string `json:"selector"`
	// Port is the port peer replicas listen on for state sync. Required.
	Port string `json:"port"`
	// Namespace is the namespace of this EPP deployment's pods. Required.
	Namespace string `json:"namespace"`
}

// peerEvent is one change to the peer set, produced by the Pod extractor and
// applied to the PeerNotifier by Start.
type peerEvent struct {
	kind fwkdl.EventType
	// id identifies the peer to delete. Set for EventDelete.
	id types.NamespacedName
	// peer is the peer to upsert. Set for EventAddOrUpdate.
	peer *fwkdl.PeerMetadata
}

// Plugin implements PeerDiscovery. Pod events arrive through a datalayer
// notification extractor and are applied to the PeerNotifier from Start's
// goroutine, which keeps the notifier's single-goroutine contract.
type Plugin struct {
	typedName   fwkplugin.TypedName
	selector    labels.Selector
	port        string
	namespace   string
	selfAddress string
	selfName    string

	// events carries peer set changes from the extractor to Start.
	events chan peerEvent

	ready     chan struct{}
	readyOnce sync.Once
}

var _ fwkdl.PeerDiscovery = (*Plugin)(nil)
var _ fwkdl.Registrant = (*Plugin)(nil)

func Factory(name string, parameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	p := &params{}
	if parameters != nil {
		if err := parameters.Decode(p); err != nil {
			return nil, fmt.Errorf("%s: failed to parse parameters: %w", PluginType, err)
		}
	}
	if p.Selector == "" {
		return nil, errors.New(PluginType + ": 'selector' parameter is required")
	}
	if p.Port == "" {
		return nil, errors.New(PluginType + ": 'port' parameter is required")
	}
	if port, err := strconv.Atoi(p.Port); err != nil || port < 1 || port > 65535 {
		return nil, fmt.Errorf("%s: invalid port %q", PluginType, p.Port)
	}
	if p.Namespace == "" {
		return nil, errors.New(PluginType + ": 'namespace' parameter is required")
	}
	if namespace, ok := os.LookupEnv("NAMESPACE"); ok {
		if namespace != p.Namespace {
			return nil, fmt.Errorf("%s: configured 'namespace' %q does not match NAMESPACE %q", PluginType, p.Namespace, namespace)
		}
	} else if handle != nil {
		log.FromContext(handle.Context()).Info("NAMESPACE is unset; cannot validate configured peer namespace", "namespace", p.Namespace)
	}
	selfAddress := os.Getenv("POD_IP")
	selfName := ""
	if selfAddress == "" {
		var err error
		selfName, err = os.Hostname()
		if err != nil {
			return nil, fmt.Errorf("%s: get hostname: %w", PluginType, err)
		}
	}
	selector, err := labels.Parse(p.Selector)
	if err != nil {
		return nil, fmt.Errorf("%s: invalid 'selector' %q: %w", PluginType, p.Selector, err)
	}
	if name == "" {
		name = PluginType
	}
	return &Plugin{
		typedName:   fwkplugin.TypedName{Type: PluginType, Name: name},
		selector:    selector,
		port:        p.Port,
		namespace:   p.Namespace,
		selfAddress: selfAddress,
		selfName:    selfName,
		events:      make(chan peerEvent),
		ready:       make(chan struct{}),
	}, nil
}

func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

// RegisterDependencies registers the Pod notification source used for peer
// discovery. The datalayer runtime owns the Kubernetes watch and dispatches
// deep-copied Pod events to the extractor.
func (p *Plugin) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      p.typedName,
		SourceType: notifications.NotificationSourceType,
		Extractor:  &podExtractor{plugin: p},
		DefaultSource: notifications.NewK8sNotificationSource(
			notifications.NotificationSourceType,
			p.typedName.Name+"/pod",
			podGVK,
		),
	})
}

func (p *Plugin) Ready() <-chan struct{} { return p.ready }

// Start signals readiness, then applies extractor events to notifier until
// ctx is cancelled. Peers from the initial Pod list may arrive after Ready.
func (p *Plugin) Start(ctx context.Context, notifier fwkdl.PeerNotifier) error {
	p.readyOnce.Do(func() { close(p.ready) })

	for {
		select {
		case ev := <-p.events:
			if err := apply(notifier, ev); err != nil {
				return err
			}
		case <-ctx.Done():
			return nil
		}
	}
}

func apply(notifier fwkdl.PeerNotifier, ev peerEvent) error {
	switch ev.kind {
	case fwkdl.EventAddOrUpdate:
		notifier.Upsert(ev.peer)
	case fwkdl.EventDelete:
		notifier.Delete(ev.id)
	default:
		return fmt.Errorf("%s: unhandled peer event kind %v", PluginType, ev.kind)
	}
	return nil
}

// emit hands an event to Start. The channel is unbuffered, so an event raised
// before Start is draining waits for it instead of being dropped.
func (p *Plugin) emit(ctx context.Context, ev peerEvent) error {
	select {
	case p.events <- ev:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (p *Plugin) acceptsPod(pod *corev1.Pod) bool {
	return p.selector.Matches(labels.Set(pod.Labels)) &&
		pod.Status.PodIP != "" &&
		pod.Status.PodIP != p.selfAddress &&
		pod.Name != p.selfName &&
		podutil.IsPodReady(pod)
}
