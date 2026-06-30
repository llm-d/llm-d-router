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

package nohitlru

import (
	"encoding/json"
	"fmt"
	"testing"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func lruWith(t *testing.T, size int, keys ...string) *lru.Cache[string, struct{}] {
	t.Helper()
	c, err := lru.New[string, struct{}](size)
	require.NoError(t, err)
	for _, k := range keys {
		c.Add(k, struct{}{})
	}
	return c
}

func decodeNoHitDump(t *testing.T, s *NoHitLRU) noHitLRUState {
	t.Helper()
	raw, err := s.DumpState()
	require.NoError(t, err)
	var state noHitLRUState
	require.NoError(t, json.Unmarshal(raw, &state))
	return state
}

func TestDumpState_NilCacheIsEmpty(t *testing.T) {
	state := decodeNoHitDump(t, &NoHitLRU{})

	assert.Equal(t, 0, state.TotalEndpoints)
	assert.Empty(t, state.Endpoints)
	assert.False(t, state.Truncated)
	assert.Equal(t, maxDebugDumpEndpoints, state.MaxEndpoints)
}

func TestDumpState_ReportsEndpointsInLRUOrder(t *testing.T) {
	// Added oldest-first; Keys() returns least-recently-used first.
	s := &NoHitLRU{lruCache: lruWith(t, 10, "ns/ep-a", "ns/ep-b", "ns/ep-c")}

	state := decodeNoHitDump(t, s)

	assert.Equal(t, 3, state.TotalEndpoints)
	assert.False(t, state.Truncated)
	assert.Equal(t, []string{"ns/ep-a", "ns/ep-b", "ns/ep-c"}, state.Endpoints)
}

func TestDumpState_CapsAndFlagsTruncation(t *testing.T) {
	const total = maxDebugDumpEndpoints + 50
	keys := make([]string, 0, total)
	for i := 0; i < total; i++ {
		keys = append(keys, fmt.Sprintf("ns/ep-%04d", i))
	}
	s := &NoHitLRU{lruCache: lruWith(t, total+10, keys...)}

	state := decodeNoHitDump(t, s)

	assert.Equal(t, total, state.TotalEndpoints, "total reflects the true count, not the capped sample")
	assert.Len(t, state.Endpoints, maxDebugDumpEndpoints)
	assert.True(t, state.Truncated)
}

func TestNoHitLRU_ImplementsStateDumper(t *testing.T) {
	require.Implements(t, (*interface {
		DumpState() (json.RawMessage, error)
	})(nil), &NoHitLRU{})
}
