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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

var (
	keyA = fwkplugin.NewDataKey("a", "test-producer")
	keyB = fwkplugin.NewDataKey("b", "test-producer")
)

type dummy struct {
	Text string
}

func (d *dummy) Clone() Cloneable {
	return &dummy{Text: d.Text}
}

type anotherDummy struct {
	Number int
}

func (d *anotherDummy) Clone() Cloneable {
	return &anotherDummy{Number: d.Number}
}

func TestExpectPutThenGetToMatch(t *testing.T) {
	attrs := NewAttributes()
	original := &dummy{"foo"}
	attrs.Put(keyA, original)

	got, ok := attrs.Get(keyA)
	assert.True(t, ok, "expected key to exist")
	assert.NotSame(t, original, got, "expected Get to return a clone, not original")

	dv, ok := got.(*dummy)
	assert.True(t, ok, "expected value to be of type *dummy")
	assert.Equal(t, "foo", dv.Text)

	_, ok = attrs.Get(keyB)
	assert.False(t, ok, "expected key not to exist")
}

func TestExpectKeysToMatchAdded(t *testing.T) {
	attrs := NewAttributes()
	keyX := fwkplugin.NewDataKey("x", "test-producer")
	keyY := fwkplugin.NewDataKey("y", "test-producer")
	attrs.Put(keyX, &dummy{"1"})
	attrs.Put(keyY, &dummy{"2"})

	keys := attrs.Keys()
	assert.Len(t, keys, 2)
	assert.ElementsMatch(t, keys, []fwkplugin.DataKey{keyX, keyY})
}

func TestCloneReturnsCopy(t *testing.T) {
	original := NewAttributes()
	keyK := fwkplugin.NewDataKey("k", "test-producer")
	original.Put(keyK, &dummy{"value"})

	cloned := original.Clone()

	kOrig, _ := original.Get(keyK)
	kClone, _ := cloned.Get(keyK)

	assert.NotSame(t, kOrig, kClone, "expected cloned value to be a different instance")
	if diff := cmp.Diff(kOrig, kClone); diff != "" {
		t.Errorf("Unexpected output (-want +got): %v", diff)
	}
}

func TestReadAttribute(t *testing.T) {
	// successful retrieval
	attrs := NewAttributes()
	original := &dummy{"foo"}
	attrs.Put(keyA, original)

	got, ok := ReadAttribute[*dummy](attrs, keyA)
	assert.True(t, ok, "expected key to exist and value to be of type *dummy")
	assert.NotSame(t, original, got, "expected Get to return a clone, not original")
	assert.Equal(t, "foo", got.Text)

	// missing key
	_, ok = ReadAttribute[*dummy](attrs, keyB)
	assert.False(t, ok, "expected key not to exist")

	// type mismatch
	other, ok := ReadAttribute[*anotherDummy](attrs, keyA)
	assert.False(t, ok, "expected type mismatch")
	assert.Nil(t, other) // zero value of pointer is nil

}

func TestKeysWithDifferentProducersAreDistinct(t *testing.T) {
	attrs := NewAttributes()
	attrs.Put(keyA, &dummy{"foo"})

	other := keyA.WithNonEmptyProducerName("other-producer")
	_, ok := attrs.Get(other)
	assert.False(t, ok, "expected key with a different producer name to be a distinct entry")

	attrs.Put(other, &dummy{"bar"})
	got, ok := attrs.Get(keyA)
	assert.True(t, ok)
	assert.Equal(t, "foo", got.(*dummy).Text, "expected the original entry to be untouched")
}

func TestDynamicAttribute(t *testing.T) {
	attrs := NewAttributes()

	val := &dummy{"live"}
	dynamic := &DynamicAttribute{
		Get: func() Cloneable {
			return val
		},
	}

	dynamicKey := fwkplugin.NewDataKey("dynamic", "test-producer")
	attrs.Put(dynamicKey, dynamic)

	// First read
	got, ok := attrs.Get(dynamicKey)
	assert.True(t, ok)
	assert.Equal(t, "live", got.(*dummy).Text)

	// Mutate the source value
	val.Text = "changed"

	// Second read should reflect change
	got2, ok := attrs.Get(dynamicKey)
	assert.True(t, ok)
	assert.Equal(t, "changed", got2.(*dummy).Text)
}
