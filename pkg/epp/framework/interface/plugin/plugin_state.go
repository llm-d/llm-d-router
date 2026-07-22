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
	"fmt"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

const (
	// DefaultStalenessThreshold is the built-in threshold for considering data as stale: if a
	// request's data has not been read/written within this duration it is reaped in the next cleanup
	// cycle. It applies when no process-wide (SetDefaultStalenessThreshold) override is set.
	DefaultStalenessThreshold = time.Minute * 5
	// cleanupInterval defines the periodic interval that the cleanup go routine uses to check for stale data.
	cleanupInterval = time.Minute
)

// defaultStalenessThreshold is the process-wide default applied to PluginState instances.
// It may be overridden once during startup via SetDefaultStalenessThreshold.
var defaultStalenessThreshold = DefaultStalenessThreshold

// SetDefaultStalenessThreshold overrides the process-wide default staleness threshold used by
// PluginState instances that do not set an explicit WithStalenessThreshold. It is intended to be
// called once at startup, before any plugin is instantiated. Non-positive values are ignored.
func SetDefaultStalenessThreshold(d time.Duration) {
	if d > 0 {
		defaultStalenessThreshold = d
	}
}

// NewPluginState initializes a new PluginState and returns its pointer. Each instance captures the
// process-wide staleness threshold at construction; configure it via SetDefaultStalenessThreshold
// before instantiating plugins.
func NewPluginState(ctx context.Context) *PluginState {
	pluginState := &PluginState{stalenessThreshold: defaultStalenessThreshold}
	go pluginState.cleanup(ctx)
	return pluginState
}

// PluginState is per-plugin scratch storage scoped to a single request. A plugin's
// extension points (e.g. PreRequest, ResponseBody) can write, read, and alter entries
// here to coordinate within that plugin. Entries are keyed by RequestID and reaped
// after "stalenessThreshold" of inactivity.
//
// PluginState is not a cross-plugin handoff channel. Data shared between plugins must
// flow through the Producer/Consumer DAG: write to Endpoint AttributeMap for
// per-endpoint data, or to the InferenceRequest attribute store for per-request data.
// The DAG validates type compatibility and execution ordering; PluginState does not.
//
// Note: PluginState uses a sync.Map to back the storage, because it is thread safe.
// It's aimed to optimize for the "write once and read many times" scenarios.
type PluginState struct {
	// key: RequestID, value: sync.Map[StateKey]StateData
	storage sync.Map
	// key: RequestID, value: time.Time
	requestToLastAccessTime sync.Map
	// stalenessThreshold is the inactivity duration after which a request's data is reaped.
	stalenessThreshold time.Duration
}

// Read retrieves data with the given "key" in the context of "requestID" from PluginState.
// If the key is not present, ErrNotFound is returned.
func (s *PluginState) Read(requestID string, key StateKey) (StateData, error) {
	stateMap, ok := s.storage.Load(requestID)
	if !ok {
		return nil, ErrNotFound
	}
	s.requestToLastAccessTime.Store(requestID, time.Now())

	stateData := stateMap.(*sync.Map)
	if value, ok := stateData.Load(key); ok {
		return value.(StateData), nil
	}

	return nil, ErrNotFound
}

// Write stores the given "val" in PluginState with the given "key" in the context of the given "requestID".
// Note: overwriting an existing key does NOT trigger OnEvicted on the displaced value.
func (s *PluginState) Write(requestID string, key StateKey, val StateData) {
	s.requestToLastAccessTime.Store(requestID, time.Now())
	var stateData *sync.Map
	stateMap, ok := s.storage.Load(requestID)
	if ok {
		stateData = stateMap.(*sync.Map)
	} else {
		stateData = &sync.Map{}
	}

	stateData.Store(key, val)

	s.storage.Store(requestID, stateData)
}

// Delete deletes data associated with the given requestID from PluginState.
//
// Triggers OnEvicted for every EvictableStateData entry being removed.
// OnEvicted is invoked at most once per entry: Delete uses LoadAndDelete
// per key, so it does not fire OnEvicted on entries that were concurrently
// removed by a racing DeleteKey (or another Delete) on the same requestID.
func (s *PluginState) Delete(requestID string) {
	s.requestToLastAccessTime.Delete(requestID)
	val, ok := s.storage.LoadAndDelete(requestID)
	if !ok {
		return
	}
	stateData := val.(*sync.Map)
	stateData.Range(func(k, _ any) bool {
		if claimed, ok := stateData.LoadAndDelete(k); ok {
			if evictable, ok := claimed.(EvictableStateData); ok {
				evictable.OnEvicted(requestID, k.(StateKey))
			}
		}
		return true
	})
}

// DeleteKey deletes the data associated with the given "key" in the context of "requestID" from PluginState.
//
// Note: DeleteKey triggers the OnEvicted callback for the EvictableStateData entry being removed.
func (s *PluginState) DeleteKey(requestID string, key StateKey) {
	stateMap, ok := s.storage.Load(requestID)
	if !ok {
		return
	}

	stateData := stateMap.(*sync.Map)
	if val, ok := stateData.LoadAndDelete(key); ok {
		if evictable, ok := val.(EvictableStateData); ok {
			evictable.OnEvicted(requestID, key)
		}
	}
}

// Touch updates the last access time for the given requestID, extending its
// lifetime before being reaped by the janitor.
func (s *PluginState) Touch(requestID string) {
	s.requestToLastAccessTime.Store(requestID, time.Now())
}

// LastAccessTime returns the last access time for the given requestID and a
// boolean indicating if the requestID was found.
func (s *PluginState) LastAccessTime(requestID string) (time.Time, bool) {
	if val, ok := s.requestToLastAccessTime.Load(requestID); ok {
		return val.(time.Time), true
	}
	return time.Time{}, false
}

// cleanup periodically deletes data associated with the given requestID.
func (s *PluginState) cleanup(ctx context.Context) {
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			log.FromContext(ctx).V(logutil.DEFAULT).Info("Shutting down plugin state cleanup")
			return
		case <-ticker.C:
			s.cleanStaleRequests()
		}
	}
}

// cleanStaleRequests iterates through all requests and removes those that haven't been
// accessed for longer than the staleness threshold. This operation is safe to run concurrently
// with other operations on the PluginState.
func (s *PluginState) cleanStaleRequests() {
	s.requestToLastAccessTime.Range(func(k, v any) bool {
		requestID := k.(string)
		lastAccessTime := v.(time.Time)
		if time.Since(lastAccessTime) > s.stalenessThreshold {
			log.Log.V(logutil.DEBUG).Info("Cleaning up stale request from PluginState", "requestID", requestID, "lastAccessTime", lastAccessTime, "stalenessThreshold", s.stalenessThreshold)
			s.Delete(requestID) // cleanup stale requests (this is safe in sync.Map)
		}
		return true
	})
}

// ReadPluginStateKey retrieves data with the given key from PluginState and asserts it to type T.
// Returns an error if the key is not found or the type assertion fails.
func ReadPluginStateKey[T StateData](state *PluginState, requestID string, key StateKey) (T, error) {
	var zero T

	raw, err := state.Read(requestID, key)
	if err != nil {
		return zero, err
	}

	val, ok := raw.(T)
	if !ok {
		return zero, fmt.Errorf("unexpected type for key %q: got %T", key, raw)
	}

	return val, nil
}
