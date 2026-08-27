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

package scheduling

import (
	"sync"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// AttributeScope decides whether the plugin holding a request may reach a
// request-attribute key. It is the request-store half of the confinement the
// datalayer package applies to endpoint attributes, and datalayer implements
// it; the interface lives here so InferenceRequest can consult it without
// importing that package.
//
// Both methods record the rejection they report, so a caller that denies an
// access does not also have to count it.
type AttributeScope interface {
	// AllowPut reports whether the plugin declared key in Produces().
	AllowPut(key fwkplugin.DataKey) bool
	// AllowGet reports whether the plugin declared key in Consumes(), or in
	// Produces() since a producer may read back its own output.
	AllowGet(key fwkplugin.DataKey) bool
	// Declares answers the same question as AllowGet without recording a
	// rejection, for enumerating the store. Listing a key the plugin may not
	// read is not an access, so it counts as no violation.
	Declares(key fwkplugin.DataKey) bool
}

// WithAttributeScope returns a shallow copy of the request whose attribute
// access is confined to what scope allows. The copy shares the backing store,
// so an allowed write is visible to the framework and to every later plugin.
// A nil scope returns the request unchanged, leaving it unconfined.
//
// The store and the header map are materialized on the receiver before the
// copy is taken. A shallow copy cannot propagate an assignment to a field, and
// both are lazily allocated by their writers -- PreRequest plugins initialize
// Headers before setting one -- so an allocation that landed on the copy would
// be dropped when the plugin returned.
func (r *InferenceRequest) WithAttributeScope(scope AttributeScope) *InferenceRequest {
	if r == nil || scope == nil {
		return r
	}
	if r.attributes == nil {
		r.attributes = &sync.Map{}
	}
	if r.Headers == nil {
		r.Headers = map[string]string{}
	}
	scoped := *r
	scoped.scope = scope
	return &scoped
}

// PutAttribute stores value at key in the request's attribute store.
// The backing store is lazily allocated on first write.
// Callers must not write concurrently to the same request from multiple goroutines.
//
// Keys are DataKey values for the same reason the endpoint AttributeMap uses
// them: the per-request store is the other half of the producer/consumer
// exchange, so a plugin reaches an entry only through a key it names in
// Produces() or Consumes().
//
// A write the request's scope rejects is dropped. The extension points with an
// error return surface it through Violations; the rest leave the counter as the
// only signal, matching endpoint attributes.
func (r *InferenceRequest) PutAttribute(key fwkplugin.DataKey, value any) {
	if r.scope != nil && !r.scope.AllowPut(key) {
		return
	}
	if r.attributes == nil {
		r.attributes = &sync.Map{}
	}
	r.attributes.Store(key, value)
}

// GetAttribute returns the value stored at key, or nil and false if absent.
// A read the request's scope rejects resolves as absent.
// Prefer ReadRequestAttribute for type-safe access.
func (r *InferenceRequest) GetAttribute(key fwkplugin.DataKey) (any, bool) {
	if r.scope != nil && !r.scope.AllowGet(key) {
		return nil, false
	}
	if r.attributes == nil {
		return nil, false
	}
	return r.attributes.Load(key)
}

// AttributeKeys returns the keys currently present in the request's attribute store.
// The order is unspecified. Under a scope only the declared keys are returned,
// so enumeration cannot be used to discover an attribute the plugin may not read.
func (r *InferenceRequest) AttributeKeys() []fwkplugin.DataKey {
	keys := make([]fwkplugin.DataKey, 0)
	if r.attributes == nil {
		return keys
	}
	// PutAttribute is the only writer, so every key is a DataKey.
	r.attributes.Range(func(k, _ any) bool {
		key := k.(fwkplugin.DataKey)
		if r.scope == nil || r.scope.Declares(key) {
			keys = append(keys, key)
		}
		return true
	})
	return keys
}

// ReadRequestAttribute returns the value stored at key, type-asserted to T.
// It returns the zero value of T and false if the key is missing or the value
// is not of type T.
func ReadRequestAttribute[T any](r *InferenceRequest, key fwkplugin.DataKey) (T, bool) {
	var zero T
	v, ok := r.GetAttribute(key)
	if !ok {
		return zero, false
	}
	t, ok := v.(T)
	if !ok {
		return zero, false
	}
	return t, true
}
