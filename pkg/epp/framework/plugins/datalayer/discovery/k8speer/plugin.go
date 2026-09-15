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
// deployment's own Pods via a controller-runtime reconciler.
package k8speer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"

	"k8s.io/apimachinery/pkg/labels"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	"github.com/llm-d/llm-d-router/pkg/epp/controller"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/statesync"
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

// Plugin implements PeerDiscovery by watching Pods via a controller-runtime
// reconciler registered with the caller's manager.
//
// SetupWithManager wires the reconciler to emit directly into the plugin's
// MemoryPeerStore and registers a Start runnable with the manager. Peers are
// available in Store() from the first reconcile; no buffering or late binding
// is needed because the store is goroutine-safe and owned by the plugin.
type Plugin struct {
	typedName   fwkplugin.TypedName
	selector    labels.Selector
	port        string
	namespace   string
	selfAddress string
	store       *statesync.MemoryPeerStore
	notifier    fwkdl.PeerNotifier

	ready     chan struct{}
	readyOnce sync.Once
}

var _ fwkdl.PeerDiscovery = (*Plugin)(nil)

func Factory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
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
	if p.Namespace == "" {
		return nil, errors.New(PluginType + ": 'namespace' parameter is required")
	}
	selector, err := labels.Parse(p.Selector)
	if err != nil {
		return nil, fmt.Errorf("%s: invalid 'selector' %q: %w", PluginType, p.Selector, err)
	}
	if name == "" {
		name = PluginType
	}
	store := statesync.NewMemoryPeerStore()
	return &Plugin{
		typedName:   fwkplugin.TypedName{Type: PluginType, Name: name},
		selector:    selector,
		port:        p.Port,
		namespace:   p.Namespace,
		selfAddress: os.Getenv("POD_IP"),
		store:       store,
		notifier:    fwkdl.NewPeerNotifier(store),
		ready:       make(chan struct{}),
	}, nil
}

func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

// Store returns the peer store. The CrossReplicaSyncer reads from it to know
// which replicas to sync with.
func (p *Plugin) Store() *statesync.MemoryPeerStore { return p.store }

// SetupWithManager registers the EPPPeerReconciler and a Start runnable with
// the given manager. The reconciler emits directly into the plugin's store via
// its notifier.
func (p *Plugin) SetupWithManager(mgr ctrl.Manager) error {
	reconciler := &controller.EPPPeerReconciler{
		Reader:      mgr.GetClient(),
		Notifier:    p.notifier,
		Selector:    p.selector,
		Namespace:   p.namespace,
		Port:        p.port,
		SelfAddress: p.selfAddress,
		OnFirstReconcile: func() {
			p.readyOnce.Do(func() { close(p.ready) })
		},
	}
	if err := reconciler.SetupWithManager(mgr); err != nil {
		return err
	}
	return mgr.Add(manager.RunnableFunc(func(ctx context.Context) error {
		return p.Start(ctx, nil)
	}))
}

func (p *Plugin) Ready() <-chan struct{} { return p.ready }

// Start blocks until ctx is cancelled. The notifier parameter is unused; the
// reconciler emits directly into the plugin's store.
func (p *Plugin) Start(ctx context.Context, _ fwkdl.PeerNotifier) error {
	<-ctx.Done()
	return nil
}
