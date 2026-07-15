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

package plugin

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/types"

	statepb "github.com/llm-d/llm-d-router/pkg/epp/statestore/stateapi/proto/gen"
)

// StateAccessMode mirrors pkg/epp/statestore.AccessMode as a plain string:
// pkg/epp/statestore transitively imports this package (via
// pkg/epp/framework/interface/flowcontrol), so it cannot be imported here
// without a cycle. The empty string is treated as "Local" by convention,
// matching statestore.AccessMode's own zero-value semantics.
type StateAccessMode string

const (
	// StateAccessModeLocal disables remote access for a capability; it
	// behaves like classic even when a State API client is configured.
	StateAccessModeLocal StateAccessMode = "Local"
	// StateAccessModeFailOpen prefers a remote read/write and falls back to
	// the local shadow on failure or timeout. Valid for inflight and prefix.
	StateAccessModeFailOpen StateAccessMode = "FailOpen"
	// StateAccessModeLocalFallback is flow-control only.
	StateAccessModeLocalFallback StateAccessMode = "LocalFallback"
)

// StateAccessModeConfig mirrors pkg/epp/statestore.AccessModeConfig.
type StateAccessModeConfig struct {
	Inflight    StateAccessMode
	Prefix      StateAccessMode
	FlowControl StateAccessMode
}

// Handle provides plugins a set of standard data and tools to work with
type Handle interface {
	// Context returns a context the plugins can use, if they need one
	Context() context.Context

	HandlePlugins

	// PodList lists pods. Returns nil if no pod source was configured on the handle.
	PodList() []types.NamespacedName

	// Metrics returns a recorder plugins can use to register metrics. It may return
	// nil when no recorder is configured.
	Metrics() MetricsRecorder

	// RemoteStateClient returns the shared gRPC client for the internal State
	// API (RFC #1593 feasibility spike) and its configured per-call timeout.
	// ok is false when not running in stateless mode with a configured
	// stateful-epp-address, in which case client is nil and must not be used.
	// Producers use this to build their own Remote/FailOpen-wrapped
	// pkg/epp/statestore state, using themselves as the Local backend.
	RemoteStateClient() (client statepb.StateAPIClient, timeout time.Duration, ok bool)

	// StateAccessMode returns the configured per-capability access mode
	// (RFC #1593 feasibility spike). The zero value of each field is treated
	// as statestore.AccessModeLocal. Producers and the admission controller
	// consult this alongside RemoteStateClient to decide whether to build a
	// Remote/FailOpen-wrapped state or stay purely local even when a State
	// API client is configured -- this is what isolates "more replicas
	// sharing CPU work" from "cost of the remote read/write path" in the
	// ha-benchmark.sh perf-test harness (bin/, gitignored).
	StateAccessMode() StateAccessModeConfig
}

// HandlePlugins defines a set of APIs to work with instantiated plugins
type HandlePlugins interface {
	// Plugin returns the named plugin instance
	Plugin(name string) Plugin

	// AddPlugin adds a plugin to the set of known plugin instances
	AddPlugin(name string, plugin Plugin)

	// GetAllPlugins returns all of the known plugins
	GetAllPlugins() []Plugin

	// GetAllPluginsWithNames returns all of the known plugins with their names
	GetAllPluginsWithNames() map[string]Plugin
}

// PodListFunc is a function type that filters and returns a list of pod metrics
type PodListFunc func() []types.NamespacedName

// eppHandle is an implementation of the interface plugins.Handle
type eppHandle struct {
	ctx context.Context
	HandlePlugins
	podList            PodListFunc
	metricsRecorder    MetricsRecorder
	remoteStateClient  statepb.StateAPIClient
	remoteStateTimeout time.Duration
	stateAccessMode    StateAccessModeConfig
}

// Context returns a context the plugins can use, if they need one
func (h *eppHandle) Context() context.Context {
	return h.ctx
}

// eppHandlePlugins implements the set of APIs to work with instantiated plugins
type eppHandlePlugins struct {
	plugins map[string]Plugin
}

// Plugin returns the named plugin instance
func (h *eppHandlePlugins) Plugin(name string) Plugin {
	return h.plugins[name]
}

// AddPlugin adds a plugin to the set of known plugin instances
func (h *eppHandlePlugins) AddPlugin(name string, plugin Plugin) {
	h.plugins[name] = plugin
}

// GetAllPlugins returns all of the known plugins
func (h *eppHandlePlugins) GetAllPlugins() []Plugin {
	result := make([]Plugin, 0, len(h.plugins))
	for _, plugin := range h.plugins {
		result = append(result, plugin)
	}
	return result
}

// GetAllPluginsWithNames returns al of the known plugins with their names
func (h *eppHandlePlugins) GetAllPluginsWithNames() map[string]Plugin {
	return h.plugins
}

// PodList lists pods.
func (h *eppHandle) PodList() []types.NamespacedName {
	if h.podList == nil {
		return nil
	}
	return h.podList()
}

// Metrics returns the MetricsRecorder.
func (h *eppHandle) Metrics() MetricsRecorder {
	return h.metricsRecorder
}

// RemoteStateClient returns the shared State API client, if configured.
func (h *eppHandle) RemoteStateClient() (statepb.StateAPIClient, time.Duration, bool) {
	if h.remoteStateClient == nil {
		return nil, 0, false
	}
	return h.remoteStateClient, h.remoteStateTimeout, true
}

// StateAccessMode returns the configured per-capability access mode.
func (h *eppHandle) StateAccessMode() StateAccessModeConfig {
	return h.stateAccessMode
}

// HandleOption configures an eppHandle constructed via NewEppHandle.
type HandleOption func(*eppHandle)

// WithMetricsRecorder sets the MetricsRecorder used by the handle. A nil recorder
// is ignored.
func WithMetricsRecorder(recorder MetricsRecorder) HandleOption {
	return func(h *eppHandle) {
		if recorder != nil {
			h.metricsRecorder = recorder
		}
	}
}

// WithRemoteStateClient sets the shared State API client and its per-call
// timeout used by the handle. A nil client is ignored.
func WithRemoteStateClient(client statepb.StateAPIClient, timeout time.Duration) HandleOption {
	return func(h *eppHandle) {
		if client != nil {
			h.remoteStateClient = client
			h.remoteStateTimeout = timeout
		}
	}
}

// WithStateAccessMode sets the per-capability access mode used by the handle.
func WithStateAccessMode(cfg StateAccessModeConfig) HandleOption {
	return func(h *eppHandle) {
		h.stateAccessMode = cfg
	}
}

func NewEppHandle(ctx context.Context, podList PodListFunc, opts ...HandleOption) Handle {
	h := &eppHandle{
		ctx: ctx,
		HandlePlugins: &eppHandlePlugins{
			plugins: map[string]Plugin{},
		},
		podList: podList,
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

// PluginByType retrieves the specified plugin by name and verifies its type
func PluginByType[P Plugin](handlePlugins HandlePlugins, name string) (P, error) {
	var zero P

	rawPlugin := handlePlugins.Plugin(name)
	if rawPlugin == nil {
		return zero, fmt.Errorf("there is no plugin with the name '%s' defined", name)
	}
	plugin, ok := rawPlugin.(P)
	if !ok {
		return zero, fmt.Errorf("the plugin with the name '%s' is not an instance of %T", name, zero)
	}
	return plugin, nil
}
