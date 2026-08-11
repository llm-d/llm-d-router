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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

type slotDummy struct {
	Value string
}

func (d *slotDummy) Clone() Cloneable {
	return &slotDummy{Value: d.Value}
}

// slotOther is structurally identical to slotDummy but a distinct type so
// reflect.TypeOf distinguishes them; this models the named-type case the
// session-affinity consumer used to need reflect for.
type slotOther struct {
	Value string
}

func (d *slotOther) Clone() Cloneable {
	return &slotOther{Value: d.Value}
}

func TestSlotPut_Match(t *testing.T) {
	dk := plugin.NewDataKey("slot-match", "test")
	s := NewSlot[*slotDummy](dk)
	m := NewAttributes()

	ok := s.Put(m, &slotDummy{Value: "hello"})
	assert.True(t, ok, "matched type should be accepted")

	got, present := m.Get(dk)
	assert.True(t, present)
	d, isDummy := got.(*slotDummy)
	assert.True(t, isDummy, "stored value should be the declared type")
	assert.Equal(t, "hello", d.Value)
}

func TestSlotPut_NilDrops(t *testing.T) {
	dk := plugin.NewDataKey("slot-nil", "test")
	s := NewSlot[*slotDummy](dk)
	m := NewAttributes()

	ok := s.Put(m, nil)
	assert.False(t, ok, "nil value should not be stored")

	_, present := m.Get(dk)
	assert.False(t, present, "nil Put must not write the key")
}

func TestSlotPutAny_TypeMismatchDrops(t *testing.T) {
	dk := plugin.NewDataKey("slot-mismatch", "test")
	s := NewSlot[*slotDummy](dk)
	m := NewAttributes()

	// slotOther has the same shape as slotDummy but a different
	// reflect.Type; PutAny must reject it. This is the value-safety
	// complement to PR 2190's key-safety check.
	ok := s.PutAny(m, &slotOther{Value: "wrong"})
	assert.False(t, ok, "value of a different struct type must be rejected")

	_, present := m.Get(dk)
	assert.False(t, present, "rejected value must not be written")
}

func TestSlotPutAny_Match(t *testing.T) {
	dk := plugin.NewDataKey("slot-any-match", "test")
	s := NewSlot[*slotDummy](dk)
	m := NewAttributes()

	ok := s.PutAny(m, &slotDummy{Value: "v"})
	assert.True(t, ok)

	got, present := m.Get(dk)
	assert.True(t, present)
	assert.Equal(t, "v", got.(*slotDummy).Value)
}

func TestNewSlot_TypeCaptured(t *testing.T) {
	dk := plugin.NewDataKey("type-capture", "test")
	s := NewSlot[*slotDummy](dk)
	assert.Equal(t, "*datalayer.slotDummy", s.Type().String())
}

func TestSlotDataKey(t *testing.T) {
	dk := plugin.NewDataKey("dk-roundtrip", "test")
	s := NewSlot[*slotDummy](dk)
	assert.Equal(t, dk, s.DataKey())
}
