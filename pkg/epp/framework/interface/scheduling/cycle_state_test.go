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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

type cycleStateTestData struct {
	value string
}

func (d *cycleStateTestData) Clone() plugin.StateData {
	if d == nil {
		return nil
	}
	return &cycleStateTestData{value: d.value}
}

type otherStateData struct{}

func (o *otherStateData) Clone() plugin.StateData { return &otherStateData{} }

func TestNewCycleState(t *testing.T) {
	cs := NewCycleState()
	assert.NotNil(t, cs)

	_, err := cs.Read(plugin.StateKey("missing"))
	assert.Equal(t, plugin.ErrNotFound, err)
}

func TestCycleState_WriteRead(t *testing.T) {
	cs := NewCycleState()
	key := plugin.StateKey("k1")
	data := &cycleStateTestData{value: "v1"}

	cs.Write(key, data)

	got, err := cs.Read(key)
	assert.NoError(t, err)
	td, ok := got.(*cycleStateTestData)
	assert.True(t, ok)
	assert.Equal(t, "v1", td.value)
}

func TestCycleState_OverwriteKey(t *testing.T) {
	cs := NewCycleState()
	key := plugin.StateKey("k")

	cs.Write(key, &cycleStateTestData{value: "first"})
	cs.Write(key, &cycleStateTestData{value: "second"})

	got, err := cs.Read(key)
	assert.NoError(t, err)
	assert.Equal(t, "second", got.(*cycleStateTestData).value)
}

func TestCycleState_Delete(t *testing.T) {
	cs := NewCycleState()
	key := plugin.StateKey("k")
	cs.Write(key, &cycleStateTestData{value: "v"})

	cs.Delete(key)

	_, err := cs.Read(key)
	assert.Equal(t, plugin.ErrNotFound, err)
}

func TestCycleState_DeleteUnknownIsNoop(t *testing.T) {
	cs := NewCycleState()
	assert.NotPanics(t, func() {
		cs.Delete(plugin.StateKey("never-written"))
	})
}

func TestReadCycleStateKey_Success(t *testing.T) {
	cs := NewCycleState()
	key := plugin.StateKey("typed")
	cs.Write(key, &cycleStateTestData{value: "hello"})

	got, err := ReadCycleStateKey[*cycleStateTestData](cs, key)
	assert.NoError(t, err)
	assert.Equal(t, "hello", got.value)
}

func TestReadCycleStateKey_NotFound(t *testing.T) {
	cs := NewCycleState()
	got, err := ReadCycleStateKey[*cycleStateTestData](cs, plugin.StateKey("missing"))
	assert.Equal(t, plugin.ErrNotFound, err)
	assert.Nil(t, got)
}

func TestReadCycleStateKey_WrongType(t *testing.T) {
	cs := NewCycleState()
	key := plugin.StateKey("k")
	cs.Write(key, &otherStateData{})

	got, err := ReadCycleStateKey[*cycleStateTestData](cs, key)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unexpected type")
	assert.Nil(t, got)
}

func TestCycleState_ConcurrentReadWrite(t *testing.T) {
	cs := NewCycleState()
	const n = 64

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(2)
		key := plugin.StateKey(rune(i))
		go func(k plugin.StateKey) {
			defer wg.Done()
			cs.Write(k, &cycleStateTestData{value: "v"})
		}(key)
		go func(k plugin.StateKey) {
			defer wg.Done()
			_, _ = cs.Read(k)
		}(key)
	}
	wg.Wait()

	// Verify writes are observable after the storm.
	for i := 0; i < n; i++ {
		key := plugin.StateKey(rune(i))
		got, err := cs.Read(key)
		assert.NoError(t, err)
		assert.Equal(t, "v", got.(*cycleStateTestData).value)
	}
}
