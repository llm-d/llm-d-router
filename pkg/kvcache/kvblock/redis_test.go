/*
Copyright 2025 The llm-d Authors.

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

package kvblock_test

import (
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/require"

	. "github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvblock"
)

// createRedisIndexForTesting creates a new RedisIndex with a mock Redis server for testing.
func createRedisIndexForTesting(t *testing.T) Index {
	t.Helper()
	server, err := miniredis.Run()
	require.NoError(t, err)

	// Store server reference for cleanup
	t.Cleanup(func() {
		server.Close()
	})

	redisConfig := &RedisIndexConfig{
		Address: server.Addr(),
	}
	index, err := NewRedisIndex(redisConfig)
	require.NoError(t, err)
	return index
}

// TestRedisIndexBehavior tests the Redis index implementation using common test behaviors.
func TestRedisIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createRedisIndexForTesting)
}
