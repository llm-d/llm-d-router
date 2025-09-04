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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	. "github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvblock"
)

func createInstrumentedIndexForTesting(t *testing.T) Index {
	t.Helper()
	cfg := DefaultInMemoryIndexConfig()
	cfg.PodCacheSize = 100 // for testConcurrentOperations
	index, err := NewInMemoryIndex(cfg)
	require.NoError(t, err)
	instrumented := NewInstrumentedIndex(index)
	assert.NotNil(t, instrumented)
	return instrumented
}

func TestNewInstrumentedIndex(t *testing.T) {
	// Wrap with instrumentation
	instrumented := createInstrumentedIndexForTesting(t)
	// Verify it implements Index interface
	assert.Implements(t, (*Index)(nil), instrumented)
}

func TestInstrumentedIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createInstrumentedIndexForTesting)
}
