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

package datalayer

import (
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// sourceHit identifies a matched source by its variant, registered name, and value.
// src is plugin.Plugin because PollingDispatcher is not a DataSource.
type sourceHit struct {
	variant sourceVariant
	name    string
	src     fwkplugin.Plugin
}

// variantSourceMap stores DataSources of one variant, keyed by TypedName.Name.
// sync.Map: registrations happen during Configure; reads run concurrently from
// NewEndpoint, dispatch, and Collector goroutines (godoc case 1).
type variantSourceMap[T fwkdl.DataSource] struct {
	v sourceVariant
	m sync.Map
}

func newVariantSourceMap[T fwkdl.DataSource](v sourceVariant) *variantSourceMap[T] {
	return &variantSourceMap[T]{v: v}
}

// Set stores src under its TypedName.Name, overwriting any prior entry.
func (m *variantSourceMap[T]) Set(src T) {
	m.m.Store(src.TypedName().Name, src)
}

// Get returns the source stored under name, if any.
func (m *variantSourceMap[T]) Get(name string) (T, bool) {
	raw, ok := m.m.Load(name)
	if !ok {
		var zero T
		return zero, false
	}
	return raw.(T), true
}

// Sources returns a snapshot slice of every stored source. Order is unspecified.
func (m *variantSourceMap[T]) Sources() []T {
	var out []T
	m.m.Range(func(_, raw any) bool {
		out = append(out, raw.(T))
		return true
	})
	return out
}

// Count returns the number of stored entries.
func (m *variantSourceMap[T]) Count() int {
	n := 0
	m.m.Range(func(_, _ any) bool {
		n++
		return true
	})
	return n
}

// IsEmpty reports whether no entries are stored.
func (m *variantSourceMap[T]) IsEmpty() bool {
	empty := true
	m.m.Range(func(_, _ any) bool {
		empty = false
		return false
	})
	return empty
}

// Range invokes f for every entry. f returning false stops iteration.
func (m *variantSourceMap[T]) Range(f func(name string, src T) bool) {
	m.m.Range(func(k, raw any) bool {
		return f(k.(string), raw.(T))
	})
}

// ForEach calls f for every entry. The first error from f stops iteration
// and is returned to the caller.
func (m *variantSourceMap[T]) ForEach(f func(name string, src T) error) error {
	var firstErr error
	m.m.Range(func(k, raw any) bool {
		if err := f(k.(string), raw.(T)); err != nil {
			firstErr = err
			return false
		}
		return true
	})
	return firstErr
}

// findFirst returns the first source for which matches returns true.
// matches takes plugin.Plugin so the same predicate works across all variants
// (including PollingDispatcher, which is not a DataSource).
func (m *variantSourceMap[T]) findFirst(matches func(fwkplugin.Plugin) bool) sourceHit {
	var found sourceHit
	m.m.Range(func(k, raw any) bool {
		src := raw.(T)
		if !matches(src) {
			return true
		}
		found = sourceHit{variant: m.v, name: k.(string), src: src}
		return false
	})
	return found
}

// pollingDispatchers stores PollingDispatchers keyed by source name. Each
// dispatcher owns its own extractors internally; the framework treats them
// as opaque dispatch units.
type pollingDispatchers struct {
	mu sync.RWMutex
	m  map[string]fwkdl.PollingDispatcher
}

func newPollingDispatchers() *pollingDispatchers {
	return &pollingDispatchers{m: make(map[string]fwkdl.PollingDispatcher)}
}

// Register installs disp under its TypedName.Name. Duplicate names fail loudly
// so a config error surfaces at startup instead of silently shadowing telemetry.
func (p *pollingDispatchers) Register(disp fwkdl.PollingDispatcher) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	name := disp.TypedName().Name
	if _, exists := p.m[name]; exists {
		return fmt.Errorf("duplicate %s source name %q", variantPolling, name)
	}
	p.m[name] = disp
	return nil
}

// Get returns the dispatcher registered under name, if any.
func (p *pollingDispatchers) Get(name string) (fwkdl.PollingDispatcher, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	d, ok := p.m[name]
	return d, ok
}

// Dispatchers returns a snapshot of all dispatchers.
func (p *pollingDispatchers) Dispatchers() map[string]fwkdl.PollingDispatcher {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make(map[string]fwkdl.PollingDispatcher, len(p.m))
	for k, v := range p.m {
		out[k] = v
	}
	return out
}

func (p *pollingDispatchers) Count() int    { p.mu.RLock(); defer p.mu.RUnlock(); return len(p.m) }
func (p *pollingDispatchers) IsEmpty() bool { return p.Count() == 0 }

// notificationManager owns the registered NotificationSources.
// GVK uniqueness is enforced per-Configure-call by a caller-owned gvk tracker
// (see runtime.go); the manager itself is pure typed storage.
type notificationManager struct {
	*variantSourceMap[fwkdl.NotificationSource]
}

func newNotificationManager() *notificationManager {
	return &notificationManager{variantSourceMap: newVariantSourceMap[fwkdl.NotificationSource](variantNotification)}
}

// endpointManager owns the registered EndpointSources.
type endpointManager struct {
	*variantSourceMap[fwkdl.EndpointSource]
}

func newEndpointManager() *endpointManager {
	return &endpointManager{variantSourceMap: newVariantSourceMap[fwkdl.EndpointSource](variantEndpoint)}
}

// collectorManager tracks per-endpoint Collectors keyed by namespaced name.
type collectorManager struct {
	// sync.Map: per-pod reconcilers concurrently add and remove
	// collectors on disjoint keys.
	m sync.Map
}

func newCollectorManager() *collectorManager {
	return &collectorManager{}
}

// Register stores c under key if absent. Returns true if c was stored, false
// if a collector was already registered for key.
func (cm *collectorManager) Register(key types.NamespacedName, c *Collector) bool {
	_, loaded := cm.m.LoadOrStore(key, c)
	return !loaded
}

// Remove deletes and returns the collector for key.
func (cm *collectorManager) Remove(key types.NamespacedName) (*Collector, bool) {
	v, ok := cm.m.LoadAndDelete(key)
	if !ok {
		return nil, false
	}
	c, _ := v.(*Collector)
	return c, c != nil
}

// StopAll calls Stop on every registered collector.
func (cm *collectorManager) StopAll() {
	cm.m.Range(func(_, v any) bool {
		if c, ok := v.(*Collector); ok {
			c.Stop()
		}
		return true
	})
}

// extractorMap is a name-keyed map of extractor slices. Populated by Runtime.Configure
// (single-threaded) and read-only thereafter; mutations after Configure returns are
// not supported. Duplicate-Type detection is the caller's responsibility (see
// runtime.Configure).
type extractorMap struct {
	m map[string][]fwkplugin.Plugin
}

func newExtractorMap() *extractorMap {
	return &extractorMap{m: make(map[string][]fwkplugin.Plugin)}
}

// Get returns the extractors stored under srcName, if any.
func (e *extractorMap) Get(srcName string) ([]fwkplugin.Plugin, bool) {
	exts, ok := e.m[srcName]
	return exts, ok
}

// Count returns the number of stored entries.
func (e *extractorMap) Count() int {
	return len(e.m)
}

// Range invokes f for every entry; f returning false stops iteration.
func (e *extractorMap) Range(f func(name string, exts []fwkplugin.Plugin) bool) {
	for k, v := range e.m {
		if !f(k, v) {
			return
		}
	}
}

// Append adds ext to srcName's slice. Pure append; callers must dedup upstream.
func (e *extractorMap) Append(srcName string, ext fwkplugin.Plugin) {
	e.m[srcName] = append(e.m[srcName], ext)
}
