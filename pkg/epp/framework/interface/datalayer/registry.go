/*
Copyright 2026 The Kubernetes Authors.

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
	"reflect"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// Registry holds the set of data keys the framework knows how to write and
// read, paired with the reflect.Type of their declared values. It is built
// once at framework startup by walking every registered ProducerPlugin's
// Produces() declaration, then used to validate:
//
//   - that every producer declares a single, consistent type for each key
//     it produces (catches two plugins claiming the same DataKey with
//     different value types),
//   - that every consumer's Consumes() declaration references a key the
//     registry knows and names the same type the producer declared (catches
//     a typo in a consumer's key and a type the consumer expected to read
//     but no producer writes).
//
// This is the value-safety complement to PR 2190's key-safety check, which
// constrains WHO may produce or consume a key. The registry constrains
// WHAT (the DataKey name + declared value type) — which plugin claims the
// key is handled separately by key-safety validation.
type Registry struct {
	types map[string]reflect.Type
}

// NewRegistry returns an empty registry. Tests use this; production code
// builds one via BuildRegistry from the registered producers.
func NewRegistry() *Registry {
	return &Registry{types: make(map[string]reflect.Type)}
}

// TypeOf returns the declared reflect.Type for key, or nil if the registry
// does not know about it.
func (r *Registry) TypeOf(key plugin.DataKey) reflect.Type {
	if r == nil {
		return nil
	}
	return r.types[key.String()]
}

// Has reports whether the registry knows about key.
func (r *Registry) Has(key plugin.DataKey) bool {
	if r == nil {
		return false
	}
	_, ok := r.types[key.String()]
	return ok
}

// Keys returns the data keys the registry knows about. Order is unspecified.
func (r *Registry) Keys() []string {
	if r == nil {
		return nil
	}
	out := make([]string, 0, len(r.types))
	for k := range r.types {
		out = append(out, k)
	}
	return out
}

// BuildRegistry walks the given plugins and returns a registry populated from
// every ProducerPlugin's Produces() map. Two producers that declare the same
// DataKey with different reflect.Types is a configuration error; the same
// type (e.g. two producers both declaring *Topology for a topology key)
// passes because DataKey.String() already namespaces by producer.
//
// A Produces() entry with a nil placeholder (some tests pass nil for the
// value rather than a typed zero value) is recorded with a nil type and
// does not error — but ValidateConsumer will reject consumers that name
// such a key with a concrete type because the registry has no type to
// compare against.
func BuildRegistry(plugins []plugin.Plugin) (*Registry, error) {
	r := NewRegistry()
	for _, p := range plugins {
		prod, ok := p.(plugin.ProducerPlugin)
		if !ok {
			continue
		}
		for dk, val := range prod.Produces() {
			declared := typeOfProduced(val)
			existing, found := r.types[dk.String()]
			if found {
				if !typesCompatible(existing, declared) {
					return nil, fmt.Errorf(
						"data key %q is produced by %q with type %v, but already registered with type %v",
						dk.String(), p.TypedName().String(), declared, existing,
					)
				}
				continue
			}
			r.types[dk.String()] = declared
		}
	}
	return r, nil
}

// ValidateConsumer checks that every DataKey consumer c declares in
// Consumes() is known to the registry and carries the same value type the
// registry recorded for it. This is run at startup, alongside the DAG
// build, so a typo or stale key is caught before traffic flows.
//
// Required keys with no producer are an error. Optional keys with no
// producer are skipped: CreateMissingDataProducers already tolerates an
// Optional key going unproduced (the consumer falls back at read time), so
// the registry must not turn that into a startup error. An Optional key
// that IS produced is still type-checked like a Required one.
func (r *Registry) ValidateConsumer(c plugin.ConsumerPlugin) error {
	if r == nil {
		return nil
	}
	name := c.TypedName().String()
	deps := c.Consumes()
	for dk, val := range deps.Required {
		if err := r.checkKey(name, dk, val); err != nil {
			return err
		}
	}
	for dk, val := range deps.Optional {
		if !r.Has(dk) {
			continue
		}
		if err := r.checkKey(name, dk, val); err != nil {
			return err
		}
	}
	return nil
}

func (r *Registry) checkKey(consumerName string, dk plugin.DataKey, val any) error {
	declared := typeOfProduced(val)
	registered, ok := r.types[dk.String()]
	if !ok {
		return fmt.Errorf(
			"plugin %q consumes data key %q which is not produced by any registered plugin",
			consumerName, dk.String(),
		)
	}
	if !typesCompatible(registered, declared) {
		return fmt.Errorf(
			"plugin %q consumes data key %q with type %v, but the producer declared type %v",
			consumerName, dk.String(), declared, registered,
		)
	}
	return nil
}

// typeOfProduced returns the reflect.Type encoded in a Produces()/Consumes()
// value. Plugins pass a typed zero value (e.g. &Topology{} or SessionID(""))
// so reflect.TypeOf captures the declared type. A nil val returns nil —
// some tests and very old plugins pass nil as a placeholder; the registry
// accepts it as "no type declared" rather than failing the build.
func typeOfProduced(v any) reflect.Type {
	if v == nil {
		return nil
	}
	return reflect.TypeOf(v)
}

// typesCompatible reports whether two declared types agree. A nil on either
// side (the placeholder case) is treated as compatible so producers and
// consumers that did not bother to declare a type still link up; the
// existing read-time type assertion is the final guard for those.
func typesCompatible(a, b reflect.Type) bool {
	if a == nil || b == nil {
		return true
	}
	return a == b
}
