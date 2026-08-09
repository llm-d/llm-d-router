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

// Package datascope confines a plugin's attribute access to the DataKeys it
// declares. Typing AttributeMap by DataKey stops a plugin naming a key it never
// declared in source; the scope here stops it reaching a key declared by some
// other plugin, which is what makes Produces() and Consumes() a contract rather
// than documentation.
package datascope

import (
	"fmt"

	"github.com/go-logr/logr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Endpoint wraps a scheduling.Endpoint and confines attribute access to a
// plugin's declared keys. Writes outside Produces() are dropped and recorded;
// reads outside Consumes() (plus the plugin's own Produces(), since a producer
// may read back what it wrote) resolve as absent.
//
// Reads resolve as absent rather than failing because the read paths -- Filter,
// Score -- have no error channel, and every consumer already handles a missing
// optional attribute. Writes have one: a producer's violation is surfaced
// through Violation and turned into an error by the caller that ran it.
type Endpoint struct {
	inner      fwksched.Endpoint
	allowedPut map[fwkplugin.DataKey]struct{}
	allowedGet map[fwkplugin.DataKey]struct{}
	pluginName string
	logger     logr.Logger
	violation  *error
}

var _ fwksched.Endpoint = &Endpoint{}

func (e *Endpoint) GetMetadata() *fwkdl.EndpointMetadata { return e.inner.GetMetadata() }
func (e *Endpoint) GetMetrics() *fwkdl.Metrics           { return e.inner.GetMetrics() }
func (e *Endpoint) String() string                       { return e.inner.String() }

// Unwrap returns the underlying endpoint. Callers hand plugin results back to
// the framework unwrapped so endpoint identity -- which the scheduler relies on
// to key score maps -- survives the round trip.
func (e *Endpoint) Unwrap() fwksched.Endpoint { return e.inner }

func (e *Endpoint) Put(key fwkplugin.DataKey, value fwkdl.Cloneable) {
	if _, ok := e.allowedPut[key]; !ok {
		err := fmt.Errorf("plugin %q wrote undeclared DataKey %q; add it to Produces()", e.pluginName, key)
		if *e.violation == nil {
			*e.violation = err
		}
		e.logger.Error(err, "Rejected write outside the plugin's declared keys")
		return
	}
	e.inner.Put(key, value)
}

func (e *Endpoint) Get(key fwkplugin.DataKey) (fwkdl.Cloneable, bool) {
	if _, ok := e.allowedGet[key]; !ok {
		e.logger.Error(
			fmt.Errorf("plugin %q read undeclared DataKey %q; add it to Consumes()", e.pluginName, key),
			"Rejected read outside the plugin's declared keys")
		return nil, false
	}
	return e.inner.Get(key)
}

// Keys returns only the declared keys that are present, so enumeration cannot
// be used to discover an attribute the plugin may not read.
func (e *Endpoint) Keys() []fwkplugin.DataKey {
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
func (e *Endpoint) Clone() fwkdl.AttributeMap {
	clone := fwkdl.NewAttributes()
	for _, key := range e.Keys() {
		if value, ok := e.inner.Get(key); ok {
			clone.Put(key, value)
		}
	}
	return clone
}

// Scope confines endpoints to what plugin declares. The returned violation
// pointer holds the first rejected write, for callers whose extension point can
// report one; it stays nil when the plugin stays inside its declarations.
//
// A plugin that declares nothing reaches nothing: an absent declaration is a
// statement that the plugin exchanges no data, not a request to be exempt.
func Scope(logger logr.Logger, plugin fwkplugin.Plugin, endpoints []fwksched.Endpoint) ([]fwksched.Endpoint, *error) {
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

	violation := new(error)
	pluginName := plugin.TypedName().String()
	// One backing array rather than an allocation per endpoint: this runs for
	// every filter and scorer on every request, over the whole candidate set.
	wrappers := make([]Endpoint, len(endpoints))
	scoped := make([]fwksched.Endpoint, len(endpoints))
	for i, endpoint := range endpoints {
		wrappers[i] = Endpoint{
			inner:      endpoint,
			allowedPut: allowedPut,
			allowedGet: allowedGet,
			pluginName: pluginName,
			logger:     logger,
			violation:  violation,
		}
		scoped[i] = &wrappers[i]
	}
	return scoped, violation
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
	if scoped, ok := endpoint.(*Endpoint); ok {
		return scoped.inner
	}
	return endpoint
}
