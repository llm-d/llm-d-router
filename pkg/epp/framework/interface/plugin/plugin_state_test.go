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
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// pluginTestData implements the StateData interface for testing purposes.
// It provides a simple string value that can be stored and retrieved.
type pluginTestData struct {
	value string
}

// Clone implements the StateData interface, creating a deep copy of the data.
func (d *pluginTestData) Clone() StateData {
	if d == nil {
		return nil
	}
	return &pluginTestData{value: d.value}
}

type evictableTestData struct {
	pluginTestData
	evictedID  string
	evictedKey StateKey
}

func (d *evictableTestData) OnEvicted(requestID string, key StateKey) {
	d.evictedID = requestID
	d.evictedKey = key
}

// Clone implements the StateData interface, ensuring that the cloned data
// remains evictable (OnEvicted is not lost).
func (d *evictableTestData) Clone() StateData {
	if d == nil {
		return nil
	}
	return &evictableTestData{
		pluginTestData: pluginTestData{value: d.value},
		evictedID:      d.evictedID,
		evictedKey:     d.evictedKey,
	}
}

func TestEvictableTestData_Clone(t *testing.T) {
	data := &evictableTestData{
		pluginTestData: pluginTestData{value: "test"},
	}
	cloned := data.Clone()

	evictable, ok := cloned.(EvictableStateData)
	assert.True(t, ok, "cloned data should satisfy EvictableStateData")

	evictable.OnEvicted("req-1", "key-1")
	assert.Equal(t, "req-1", cloned.(*evictableTestData).evictedID)
	assert.Equal(t, StateKey("key-1"), cloned.(*evictableTestData).evictedKey)
}

// TestPluginState_EvictionCallback verifies that OnEvicted is called when data is removed.
func TestPluginState_EvictionCallback(t *testing.T) {
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	t.Cleanup(cancel)
	state := NewPluginState(ctx)

	requestID := "req-evict"
	key := StateKey("foo")
	data := &evictableTestData{pluginTestData: pluginTestData{value: "bar"}}

	state.Write(requestID, key, data)

	// Case 1: DeleteKey
	state.DeleteKey(requestID, key)
	assert.Equal(t, requestID, data.evictedID)
	assert.Equal(t, key, data.evictedKey)

	// Reset
	data.evictedID = ""
	data.evictedKey = ""
	state.Write(requestID, key, data)

	// Case 2: Delete (request wide)
	state.Delete(requestID)
	assert.Equal(t, requestID, data.evictedID)
	assert.Equal(t, key, data.evictedKey)

	// Case 3: Cleanup (stale request)
	data.evictedID = ""
	data.evictedKey = ""
	state.Write(requestID, key, data)
	state.requestToLastAccessTime.Store(requestID, time.Now().Add(-2*defaultStalenessThreshold))
	state.cleanStaleRequests()
	assert.Equal(t, requestID, data.evictedID)
	assert.Equal(t, key, data.evictedKey)
}

// TestPluginState_Touch verifies that Touch extends request lifetime.
func TestPluginState_Touch(t *testing.T) {
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	t.Cleanup(cancel)
	state := NewPluginState(ctx)

	requestID := "req-touch"
	key := StateKey("foo")
	data := &pluginTestData{value: "bar"}

	state.Write(requestID, key, data)

	// Set last access time to near-stale
	nearStale := time.Now().Add(-defaultStalenessThreshold + time.Second*10)
	state.requestToLastAccessTime.Store(requestID, nearStale)

	// Touch it
	state.Touch(requestID)

	// Verify access time was updated to now
	val, ok := state.requestToLastAccessTime.Load(requestID)
	assert.True(t, ok)
	lastAccess := val.(time.Time)
	assert.True(t, lastAccess.After(nearStale))
	assert.True(t, time.Since(lastAccess) < time.Second)

	// Manually cleanup, should NOT be removed
	state.cleanStaleRequests()
	_, err := state.Read(requestID, key)
	assert.NoError(t, err)
}

// TestPluginState_ReadWrite verifies the basic operations of PluginState:
// - Writing data for a request
// - Reading the data back
// - Deleting the data and confirming it's removed
func TestPluginState_ReadWrite(t *testing.T) {
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	t.Cleanup(cancel)

	state := NewPluginState(ctx)

	req1 := "req1"
	key := StateKey("key")
	data1 := "bar1"
	req2 := "req2"
	data2 := "bar2"

	// Write data to the state storage
	state.Write(req1, key, &pluginTestData{value: data1})
	state.Write(req2, key, &pluginTestData{value: data2})

	// Read back the req1 data and verify its content
	readData, err := state.Read(req1, key)
	assert.NoError(t, err)
	td, ok := readData.(*pluginTestData)
	assert.True(t, ok, "should be able to cast to pluginTestData")
	assert.Equal(t, data1, td.value)

	// Delete the req2 data and verify content that was read before is still valid
	readData, err = state.Read(req2, key)
	assert.NoError(t, err)
	state.Delete(req2)
	td, ok = readData.(*pluginTestData)
	assert.True(t, ok, "should be able to cast to pluginTestData")
	assert.Equal(t, data2, td.value)
	// try to read again aftet deletion, verify error
	readData, err = state.Read(req2, key)
	assert.Equal(t, ErrNotFound, err)
	assert.Nil(t, readData, "expected no data after delete")

	// Read back the req1 data and verify its content after the req2 deleted
	readData, err = state.Read(req1, key)
	assert.NoError(t, err)
	td, ok = readData.(*pluginTestData)
	assert.True(t, ok, "should be able to cast to pluginTestData")
	assert.Equal(t, data1, td.value)
}

// TestReadPluginStateKey tests the generic helper function ReadPluginStateKey which provides
// type-safe access to stored data. It verifies:
// - Successful type assertion and data retrieval
// - Error handling for non-existent keys
func TestReadPluginStateKey(t *testing.T) {
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	t.Cleanup(cancel)
	state := NewPluginState(ctx)

	requestID := "req-1"
	key := StateKey("foo")
	data := &pluginTestData{value: "bar"}

	state.Write(requestID, key, data)

	// Read
	val, err := ReadPluginStateKey[*pluginTestData](state, requestID, key)
	assert.NoError(t, err)
	assert.Equal(t, "bar", val.value)

	// Not Found
	_, err = ReadPluginStateKey[*pluginTestData](state, "not-exist", key)
	assert.Equal(t, ErrNotFound, err)
}

// TestPluginState_Cleanup verifies the automatic cleanup of stale data.
// It tests that data which hasn't been accessed for longer than the staleness threshold
// is properly removed from the storage.
func TestPluginState_Cleanup(t *testing.T) {
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	t.Cleanup(cancel)

	state := NewPluginState(ctx)

	requestID := "req-stale"
	key := StateKey("foo")
	data := &pluginTestData{value: "bar"}

	state.Write(requestID, key, data)

	// Manually set last access time to far in the past
	state.requestToLastAccessTime.Store(requestID, time.Now().Add(-2*defaultStalenessThreshold))
	// Manually CleanUp
	state.cleanStaleRequests()

	_, err := state.Read(requestID, key)
	assert.Equal(t, ErrNotFound, err)
}

// TestSetDefaultStalenessThreshold verifies that the process-wide default is applied to new
// PluginState instances, that a configured value drives reaping, and that non-positive values are
// ignored.
func TestSetDefaultStalenessThreshold(t *testing.T) {
	orig := defaultStalenessThreshold
	t.Cleanup(func() { defaultStalenessThreshold = orig })

	SetDefaultStalenessThreshold(30 * time.Second)
	assert.Equal(t, 30*time.Second, defaultStalenessThreshold)

	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	t.Cleanup(cancel)
	state := NewPluginState(ctx)
	assert.Equal(t, 30*time.Second, state.stalenessThreshold, "new state should inherit the configured default")

	// A one-minute-old request is stale under the 30s configured threshold (but would survive the 5m built-in default).
	requestID := "req-configured-threshold"
	key := StateKey("foo")
	state.Write(requestID, key, &pluginTestData{value: "bar"})
	state.requestToLastAccessTime.Store(requestID, time.Now().Add(-time.Minute))
	state.cleanStaleRequests()
	_, err := state.Read(requestID, key)
	assert.Equal(t, ErrNotFound, err, "request older than the configured threshold should be reaped")

	SetDefaultStalenessThreshold(0)
	assert.Equal(t, 30*time.Second, defaultStalenessThreshold, "non-positive value must be ignored")
}

// TestPluginState_DeleteKey verifies that DeleteKey correctly removes only the specified key for a request.
func TestPluginState_DeleteKey(t *testing.T) {
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	t.Cleanup(cancel)
	state := NewPluginState(ctx)

	requestID := "req-1"
	key1 := StateKey("key1")
	key2 := StateKey("key2")
	data1 := &pluginTestData{value: "val1"}
	data2 := &pluginTestData{value: "val2"}

	state.Write(requestID, key1, data1)
	state.Write(requestID, key2, data2)

	// Delete key1
	state.DeleteKey(requestID, key1)

	// Verify key1 is gone
	_, err := state.Read(requestID, key1)
	assert.Equal(t, ErrNotFound, err)

	// Verify key2 is still there
	val2, err := state.Read(requestID, key2)
	assert.NoError(t, err)
	assert.Equal(t, "val2", val2.(*pluginTestData).value)
}
