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

package plugin

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestErrNotFound(t *testing.T) {
	assert.NotNil(t, ErrNotFound)
	assert.Equal(t, "not found", ErrNotFound.Error())
	assert.True(t, errors.Is(ErrNotFound, ErrNotFound))
}

func TestStateKey(t *testing.T) {
	k1 := StateKey("foo")
	k2 := StateKey("foo")
	k3 := StateKey("bar")

	assert.Equal(t, k1, k2)
	assert.NotEqual(t, k1, k3)
	assert.Equal(t, "foo", string(k1))
}

type stubStateData struct {
	v int
}

func (d *stubStateData) Clone() StateData {
	if d == nil {
		return nil
	}
	return &stubStateData{v: d.v}
}

func TestStateData_CloneContract(t *testing.T) {
	var d StateData = &stubStateData{v: 42}
	clone := d.Clone()

	assert.NotSame(t, d, clone)
	cloned, ok := clone.(*stubStateData)
	assert.True(t, ok)
	assert.Equal(t, 42, cloned.v)

	cloned.v = 100
	assert.Equal(t, 42, d.(*stubStateData).v, "mutating clone must not affect original")
}

func TestStateData_NilClone(t *testing.T) {
	var d *stubStateData
	assert.Nil(t, d.Clone())
}
