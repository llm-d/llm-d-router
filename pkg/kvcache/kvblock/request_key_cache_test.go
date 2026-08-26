/*
Copyright 2026 The llm-d Authors.

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

package kvblock

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRequestKeyCacheGetPrefixBatchRefreshesKeys(t *testing.T) {
	cache, err := newRequestKeyCache(2)
	require.NoError(t, err)

	first := &PodCache{}
	second := &PodCache{}
	cache.Add(10, first)
	cache.Add(20, second)

	values := cache.GetPrefixBatch([]BlockHash{10}, nil)
	require.Equal(t, []*PodCache{first}, values)
	cache.Add(30, &PodCache{})

	_, firstFound := cache.Peek(10)
	_, secondFound := cache.Peek(20)
	assert.True(t, firstFound)
	assert.False(t, secondFound)
}

func TestRequestKeyCacheGetPrefixBatchPreservesRequestOrder(t *testing.T) {
	cache, err := newRequestKeyCache(3)
	require.NoError(t, err)

	cache.Add(10, &PodCache{})
	cache.Add(20, &PodCache{})
	cache.Add(30, &PodCache{})
	cache.GetPrefixBatch([]BlockHash{10, 20}, nil)
	cache.Add(40, &PodCache{})
	cache.Add(50, &PodCache{})

	_, firstFound := cache.Peek(10)
	_, secondFound := cache.Peek(20)
	_, untouchedFound := cache.Peek(30)
	assert.False(t, firstFound)
	assert.True(t, secondFound)
	assert.False(t, untouchedFound)
}

func TestRequestKeyCacheGetPrefixBatchStopsAtGap(t *testing.T) {
	cache, err := newRequestKeyCache(3)
	require.NoError(t, err)

	first := &PodCache{}
	cache.Add(10, first)
	cache.Add(30, &PodCache{})

	values := cache.GetPrefixBatch([]BlockHash{10, 20, 30}, nil)
	assert.Equal(t, []*PodCache{first}, values)
}

func TestRequestKeyCacheGetPrefixBatchLimitsLockScope(t *testing.T) {
	cache, err := newRequestKeyCache(requestKeyBatchSize + 1)
	require.NoError(t, err)

	keys := make([]BlockHash, requestKeyBatchSize+1)
	for i := range keys {
		keys[i] = BlockHash(i + 1)
		cache.Add(keys[i], &PodCache{})
	}

	values := cache.GetPrefixBatch(keys, nil)
	assert.Len(t, values, requestKeyBatchSize)
}

func TestRequestKeyCacheGetPrefixBatchColdMissDoesNotAllocateSnapshot(t *testing.T) {
	cache, err := newRequestKeyCache(1)
	require.NoError(t, err)

	values := cache.GetPrefixBatch(make([]BlockHash, 1<<16), nil)
	assert.Zero(t, cap(values))
}
