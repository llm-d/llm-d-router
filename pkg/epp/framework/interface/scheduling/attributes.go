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

// PutAttribute stores value at key in the request's attribute store.
// The backing store is lazily allocated on first write.
// Callers must not write concurrently to the same request from multiple goroutines.
//
// Keys are DataKey values for the same reason the endpoint AttributeMap uses
// them: the per-request store is the other half of the producer/consumer
// exchange, so a plugin reaches an entry only through a key it names in
// Produces() or Consumes().
func (r *InferenceRequest) PutAttribute(key fwkplugin.DataKey, value any) {
	if r.attributes == nil {
		r.attributes = &sync.Map{}
	}
	r.attributes.Store(key, value)
}

// GetAttribute returns the value stored at key, or nil and false if absent.
// Prefer ReadRequestAttribute for type-safe access.
func (r *InferenceRequest) GetAttribute(key fwkplugin.DataKey) (any, bool) {
	if r.attributes == nil {
		return nil, false
	}
	return r.attributes.Load(key)
}

// AttributeKeys returns the keys currently present in the request's attribute store.
// The order is unspecified.
func (r *InferenceRequest) AttributeKeys() []fwkplugin.DataKey {
	keys := make([]fwkplugin.DataKey, 0)
	if r.attributes == nil {
		return keys
	}
	// PutAttribute is the only writer, so every key is a DataKey.
	r.attributes.Range(func(k, _ any) bool {
		keys = append(keys, k.(fwkplugin.DataKey))
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
