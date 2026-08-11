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

// Runtime confinement of a plugin's endpoint attribute access to the DataKeys
// it declares. Typing AttributeMap by DataKey stops a plugin naming a key it
// never declared in source; the scope here stops it reaching a key declared by
// some other plugin, which is what makes Produces() and Consumes() a contract
// rather than documentation. ValidateAndOrderDataDependencies enforces the same
// contract statically, over the declarations alone.
//
// The scope covers endpoint attributes reached through the extension points
// that receive a []scheduling.Endpoint: filters, scorers, and DataProducers.
// Two paths are deliberately outside it:
//
//   - The per-request store (InferenceRequest.PutAttribute/GetAttribute). It is
//     typed by DataKey but not confined, because in-tree plugins currently
//     exchange request attributes they do not declare.
//   - Datalayer extractors, which write through Endpoint.GetAttributes()
//     directly rather than through a scoped endpoint.

package datalayer

import (
	"fmt"
	"sync"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Violations records what a plugin did outside its declarations during one
// extension-point invocation. Every endpoint wrapper produced by a single Scope
// call shares one instance, so a plugin that reaches for the same undeclared key
// on each of a hundred candidates is reported once rather than a hundred times.
//
// The lock is taken only on the rejection path. A conforming plugin never
// touches it, so confinement costs an allowed access nothing beyond the map
// lookup.
type Violations struct {
	mu           sync.Mutex
	write        error
	readReported bool
}

// Write returns the first write rejected during the invocation, or nil when the
// plugin stayed inside its declarations. Extension points that can fail a
// request turn this into an error; those without an error channel ignore it.
func (v *Violations) Write() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.write
}

// recordWrite stores the first rejected write and reports whether this was it.
func (v *Violations) recordWrite(err error) bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.write != nil {
		return false
	}
	v.write = err
	return true
}

// recordRead reports whether this is the first rejected read of the invocation.
func (v *Violations) recordRead() bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.readReported {
		return false
	}
	v.readReported = true
	return true
}

// ScopedEndpoint wraps a scheduling.Endpoint and confines attribute access to a
// plugin's declared keys. Writes outside Produces() are dropped and recorded;
// reads outside Consumes() (plus the plugin's own Produces(), since a producer
// may read back what it wrote) resolve as absent.
//
// Reads resolve as absent rather than failing because the read paths -- Filter,
// Score -- have no error channel, and every consumer already handles a missing
// optional attribute. Writes have one: a producer's violation is surfaced
// through Violations.Write and turned into an error by the caller that ran it.
//
// Either way the rejection increments a counter, so an extension point that
// cannot fail the request still makes a misdeclared plugin visible in
// production rather than leaving it to log verbosity.
type ScopedEndpoint struct {
	inner          fwksched.Endpoint
	allowedPut     map[fwkplugin.DataKey]struct{}
	allowedGet     map[fwkplugin.DataKey]struct{}
	typedName      fwkplugin.TypedName
	extensionPoint string
	logger         logr.Logger
	violations     *Violations
}

var _ fwksched.Endpoint = &ScopedEndpoint{}

func (e *ScopedEndpoint) GetMetadata() *fwkdl.EndpointMetadata { return e.inner.GetMetadata() }
func (e *ScopedEndpoint) GetMetrics() *fwkdl.Metrics           { return e.inner.GetMetrics() }
func (e *ScopedEndpoint) String() string                       { return e.inner.String() }

// Unwrap returns the underlying endpoint. Callers hand plugin results back to
// the framework unwrapped so endpoint identity -- which the scheduler relies on
// to key score maps -- survives the round trip.
func (e *ScopedEndpoint) Unwrap() fwksched.Endpoint { return e.inner }

func (e *ScopedEndpoint) Put(key fwkplugin.DataKey, value fwkdl.Cloneable) {
	if _, ok := e.allowedPut[key]; !ok {
		e.reject(metrics.DataScopeAccessWrite, fmt.Errorf(
			"plugin %q wrote undeclared DataKey %q; add it to Produces()", e.typedName.String(), key))
		return
	}
	e.inner.Put(key, value)
}

func (e *ScopedEndpoint) Get(key fwkplugin.DataKey) (fwkdl.Cloneable, bool) {
	if _, ok := e.allowedGet[key]; !ok {
		e.reject(metrics.DataScopeAccessRead, fmt.Errorf(
			"plugin %q read undeclared DataKey %q; add it to Consumes()", e.typedName.String(), key))
		return nil, false
	}
	return e.inner.Get(key)
}

// reject counts the violation and logs it. The first offence of each kind in an
// invocation is logged at Error, the rest at DEBUG: a plugin that misreaches on
// every candidate endpoint would otherwise emit one error line per endpoint per
// request.
func (e *ScopedEndpoint) reject(access string, err error) {
	metrics.RecordPluginDataScopeViolation(e.extensionPoint, e.typedName.Type, e.typedName.Name, access)

	first := false
	if access == metrics.DataScopeAccessWrite {
		first = e.violations.recordWrite(err)
	} else {
		first = e.violations.recordRead()
	}
	if first {
		e.logger.Error(err, "Rejected access outside the plugin's declared keys",
			"extensionPoint", e.extensionPoint, "access", access)
		return
	}
	e.logger.V(logging.DEBUG).Info("Rejected access outside the plugin's declared keys",
		"extensionPoint", e.extensionPoint, "access", access, "error", err.Error())
}

// Keys returns only the declared keys that are present, so enumeration cannot
// be used to discover an attribute the plugin may not read.
func (e *ScopedEndpoint) Keys() []fwkplugin.DataKey {
	var keys []fwkplugin.DataKey
	for _, key := range e.inner.Keys() {
		if _, ok := e.allowedGet[key]; ok {
			keys = append(keys, key)
		}
	}
	return keys
}

// Clone returns a copy holding only the declared keys, so cloning cannot be
// used to read around the scope.
func (e *ScopedEndpoint) Clone() fwkdl.AttributeMap {
	clone := fwkdl.NewAttributes()
	for _, key := range e.Keys() {
		if value, ok := e.inner.Get(key); ok {
			clone.Put(key, value)
		}
	}
	return clone
}

// Scope confines endpoints to what plugin declares. extensionPoint labels the
// violation counter, so a misdeclared plugin is attributable to the point that
// ran it. The returned Violations accumulates what the plugin did outside its
// declarations, for callers whose extension point can report it.
//
// A plugin that declares nothing reaches nothing: an absent declaration is a
// statement that the plugin exchanges no data, not a request to be exempt.
func Scope(logger logr.Logger, extensionPoint string, plugin fwkplugin.Plugin, endpoints []fwksched.Endpoint) ([]fwksched.Endpoint, *Violations) {
	produces := map[fwkplugin.DataKey]any{}
	if producer, ok := plugin.(fwkplugin.ProducerPlugin); ok {
		produces = producer.Produces()
	}

	allowedPut := make(map[fwkplugin.DataKey]struct{}, len(produces))
	allowedGet := make(map[fwkplugin.DataKey]struct{}, len(produces))
	for key := range produces {
		allowedPut[key] = struct{}{}
		// A producer may read back its own output.
		allowedGet[key] = struct{}{}
	}
	if consumer, ok := plugin.(fwkplugin.ConsumerPlugin); ok {
		deps := consumer.Consumes()
		for key := range deps.Required {
			allowedGet[key] = struct{}{}
		}
		for key := range deps.Optional {
			allowedGet[key] = struct{}{}
		}
	}

	violations := &Violations{}
	typedName := plugin.TypedName()
	// One backing array rather than an allocation per endpoint: this runs for
	// every filter and scorer on every request, over the whole candidate set.
	wrappers := make([]ScopedEndpoint, len(endpoints))
	scoped := make([]fwksched.Endpoint, len(endpoints))
	for i, endpoint := range endpoints {
		wrappers[i] = ScopedEndpoint{
			inner:          endpoint,
			allowedPut:     allowedPut,
			allowedGet:     allowedGet,
			typedName:      typedName,
			extensionPoint: extensionPoint,
			logger:         logger,
			violations:     violations,
		}
		scoped[i] = &wrappers[i]
	}
	return scoped, violations
}

// Unscope restores the underlying endpoints of a plugin's result.
func Unscope(endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	unscoped := make([]fwksched.Endpoint, len(endpoints))
	for i, endpoint := range endpoints {
		unscoped[i] = unwrap(endpoint)
	}
	return unscoped
}

// UnscopeScores rekeys a scorer's result by the underlying endpoints. The
// scheduler sums scores across scorers in a map keyed by endpoint, so a wrapper
// left in a key would split one endpoint's score into several entries.
func UnscopeScores(scores map[fwksched.Endpoint]float64) map[fwksched.Endpoint]float64 {
	unscoped := make(map[fwksched.Endpoint]float64, len(scores))
	for endpoint, score := range scores {
		unscoped[unwrap(endpoint)] = score
	}
	return unscoped
}

func unwrap(endpoint fwksched.Endpoint) fwksched.Endpoint {
	if scoped, ok := endpoint.(*ScopedEndpoint); ok {
		return scoped.inner
	}
	return endpoint
}
