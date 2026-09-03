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

package local

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

func TestLocalSyncerAggregatesOnSet(t *testing.T) {
	syncer := NewLocalSyncer("test", "replica-a")
	aggregateCalls := 0
	aggregate := func(values []any) any {
		aggregateCalls++
		return values[0].(int) * 2
	}

	require.NoError(t, syncer.Set(context.Background(), "load", "default/backend-0", 21, aggregate))
	value, ok, err := syncer.Get(context.Background(), fwkdl.StateKey("load"), "default/backend-0")

	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, 42, value)
	assert.Equal(t, 1, aggregateCalls)
}

func TestLocalSyncerGetOrSet(t *testing.T) {
	syncer := NewLocalSyncer("test", "replica-a")

	value, existed, err := syncer.GetOrSet(context.Background(), "request", "request-id", "first")
	require.NoError(t, err)
	assert.False(t, existed)
	assert.Equal(t, "first", value)

	value, existed, err = syncer.GetOrSet(context.Background(), "request", "request-id", "second")
	require.NoError(t, err)
	assert.True(t, existed)
	assert.Equal(t, "first", value)
}
