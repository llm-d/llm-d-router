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

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// attributeWriter is the minimal contract Slot.Put needs: anything that can
// store a Cloneable value under a DataKey. Both datalayer.AttributeMap
// and scheduling.Endpoint satisfy it, so Slot works on both stores without
// a GetAttributes indirection.
type attributeWriter interface {
	Put(plugin.DataKey, Cloneable)
}

// Slot is a typed write handle for a single DataKey. The declared value type
// is captured once at construction, so Put writes through a single interface
// assertion. No reflect on the hot path.
//
// One producer owns one Slot per DataKey it writes. NewSlot[T] pins the type
// at compile time; a producer that builds Slot[*Foo] but holds a *Bar value
// fails at the assignment boundary, not at some downstream reader.
//
// Slot is the value-safety complement to PR 2190's key-safety check: that PR
// keeps a producer from claiming a key it does not own, this keeps it from
// writing a value of the wrong type to the key it does own.
type Slot[T Cloneable] struct {
	dk        plugin.DataKey
	valueType reflect.Type
}

// NewSlot returns a typed Slot bound to dk. The reflect.Type is captured here
// (once, at construction) for the framework startup validator; Put uses a
// direct type assertion and never reflects the value.
func NewSlot[T Cloneable](dk plugin.DataKey) *Slot[T] {
	return &Slot[T]{dk: dk, valueType: reflect.TypeOf((*T)(nil)).Elem()}
}

// DataKey returns the key this slot writes to. Useful for Produces()
// declarations and log messages.
func (s *Slot[T]) DataKey() plugin.DataKey {
	return s.dk
}

// Type returns the declared reflect.Type for this slot. Used by the framework
// startup validator when building the attribute registry from a producer's
// Produces() map.
func (s *Slot[T]) Type() reflect.Type {
	return s.valueType
}

// Put writes val to w under this slot's key. A typed-nil pointer, interface,
// map, slice, channel, or func val is dropped (the underlying store's nil
// check does not catch typed nils because the interface still carries the
// type); for value-type Ts there is no nil state and the value is stored
// unconditionally.
//
// The kind switch keeps reflect off the hot path for the common value-type
// case; the reflect call only fires for Ts where nil is possible, and the
// cost is a single IsNil probe.
func (s *Slot[T]) Put(w attributeWriter, val T) bool {
	rv := reflect.ValueOf(val)
	switch rv.Kind() {
	case reflect.Ptr, reflect.Interface, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func:
		if rv.IsNil() {
			return false
		}
	}
	w.Put(s.dk, val)
	return true
}

// PutAny is the same as Put but accepts an interface value, for the small
// number of call sites that hold values as `any`. It type-asserts to T using
// the same rules as Put. Use Put when the caller already has a T.
func (s *Slot[T]) PutAny(w attributeWriter, val any) bool {
	if val == nil {
		return false
	}
	typed, ok := val.(T)
	if !ok {
		log.Log.V(4).Info("slot put: type mismatch, dropping value",
			"key", s.dk.String(),
			"declaredType", s.valueType,
			"gotType", reflect.TypeOf(val),
		)
		return false
	}
	w.Put(s.dk, typed)
	return true
}

// String renders the slot for logs and error messages.
func (s *Slot[T]) String() string {
	return fmt.Sprintf("Slot[%s @ %s]", s.valueType, s.dk.String())
}
