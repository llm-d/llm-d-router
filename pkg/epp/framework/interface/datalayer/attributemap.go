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
	"sync"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// Cloneable types support cloning of the value.
type Cloneable interface {
	Clone() Cloneable
}

// DynamicAttribute wraps a getter function to allow on-demand resolution of attributes.
type DynamicAttribute struct {
	Get func() Cloneable
}

// Clone implements Cloneable. It copies the function pointer.
func (d *DynamicAttribute) Clone() Cloneable {
	if d == nil {
		return nil
	}
	return &DynamicAttribute{Get: d.Get}
}

// AttributeMap is used to store flexible metadata or traits
// across different aspects of an inference server.
// Stored values must be Cloneable.
//
// Keys are DataKey values rather than strings so that a plugin can only reach
// an attribute through a key it holds -- the same value it names in Produces()
// or Consumes(). This removes the raw-string escape hatch by which a plugin
// could read or write an attribute unrelated to its declaration.
type AttributeMap interface {
	Put(fwkplugin.DataKey, Cloneable)
	Get(fwkplugin.DataKey) (Cloneable, bool)
	Keys() []fwkplugin.DataKey
	Clone() AttributeMap
}

// Attributes provides a goroutine-safe implementation of AttributeMap.
// Put is the only writer, so every key in the map is a DataKey.
type Attributes struct {
	data sync.Map // key: DataKey, value: attribute value (opaque, Cloneable)
}

// NewAttributes returns a new instance of Attributes.
func NewAttributes() AttributeMap {
	return &Attributes{}
}

// Put adds or updates an attribute in the map.
func (a *Attributes) Put(key fwkplugin.DataKey, value Cloneable) {
	if value != nil {
		a.data.Store(key, value) // TODO: Clone into map to ensure isolation
	}
}

// Get retrieves an attribute by key, returning a cloned copy (or resolving it dynamically).
func (a *Attributes) Get(key fwkplugin.DataKey) (Cloneable, bool) {
	val, ok := a.data.Load(key)
	if !ok {
		return nil, false
	}

	if dynamic, ok := val.(*DynamicAttribute); ok {
		realVal := dynamic.Get()
		if realVal == nil {
			return nil, false
		}
		return realVal.Clone(), true
	}

	if cloneable, ok := val.(Cloneable); ok {
		return cloneable.Clone(), true
	}
	return nil, false
}

// Keys returns all keys in the attribute map.
func (a *Attributes) Keys() []fwkplugin.DataKey {
	var keys []fwkplugin.DataKey
	a.data.Range(func(key, _ any) bool {
		keys = append(keys, key.(fwkplugin.DataKey))
		return true
	})
	return keys
}

// Clone creates a deep copy of the entire attribute map.
func (a *Attributes) Clone() AttributeMap {
	clone := NewAttributes()
	a.data.Range(func(key, value any) bool {
		if v, ok := value.(Cloneable); ok {
			clone.Put(key.(fwkplugin.DataKey), v)
		}
		return true
	})
	return clone
}

// ReadAttribute retrieves attribute with the given key from AttributeMap and asserts it to type T.
// Second return value is 'false' if the key is not found or the type assertion fails.
func ReadAttribute[T Cloneable](attributeMap AttributeMap, key fwkplugin.DataKey) (T, bool) {
	var zero T

	raw, ok := attributeMap.Get(key)
	if !ok {
		return zero, false
	}

	val, ok := raw.(T)
	if !ok {
		return zero, false
	}

	return val, true
}
