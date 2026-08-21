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
type Plugin struct {
	typedName   fwkplugin.TypedName
	selector    labels.Selector
	port        string
	namespace   string
	selfAddress string
	store       *statesync.MemoryPeerStore
	reconciler  *controller.EPPPeerReconciler

	ready     chan struct{}
	readyOnce sync.Once
}

var (
	_ fwkdl.PeerDiscovery    = (*Plugin)(nil)
	_ fwkplugin.ManagerSetup = (*Plugin)(nil)
)

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
	return &Plugin{
		typedName:   fwkplugin.TypedName{Type: PluginType, Name: name},
		selector:    selector,
		port:        p.Port,
		namespace:   p.Namespace,
		selfAddress: os.Getenv("POD_IP"),
		store:       statesync.NewMemoryPeerStore(),
		ready:       make(chan struct{}),
	}, nil
}

func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

// Store returns the peer store. Currently unused; will be consumed by the
// CrossReplicaSyncer once it is wired.
func (p *Plugin) Store() *statesync.MemoryPeerStore { return p.store }

// SetupWithManager registers the EPPPeerReconciler with the given manager.
// Must be called before the manager starts. The reconciler's notifier is
// bound later by Start via BindNotifier.
func (p *Plugin) SetupWithManager(mgr ctrl.Manager) error {
	p.reconciler = &controller.EPPPeerReconciler{
		Reader:      mgr.GetClient(),
		Selector:    p.selector,
		Namespace:   p.namespace,
		Port:        p.port,
		SelfAddress: p.selfAddress,
		OnFirstReconcile: func() {
			p.readyOnce.Do(func() { close(p.ready) })
		},
	}
	return p.reconciler.SetupWithManager(mgr)
}

func (p *Plugin) Ready() <-chan struct{} { return p.ready }

// Start binds the caller's notifier to the reconciler and blocks until ctx
// is cancelled. The reconciler is driven by the controller-runtime manager;
// this method wires the notifier so discovered peers are forwarded.
func (p *Plugin) Start(ctx context.Context, notifier fwkdl.PeerNotifier) error {
	if err := p.reconciler.BindNotifier(notifier); err != nil {
		return err
	}
	<-ctx.Done()
	return nil
}
