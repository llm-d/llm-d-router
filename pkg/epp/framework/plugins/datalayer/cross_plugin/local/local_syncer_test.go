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

type cloneableInt int

func (v cloneableInt) Clone() fwkdl.Cloneable { return v }

func TestLocalSyncerAggregatesLiveLocalValue(t *testing.T) {
	syncer := NewLocalSyncer("test", "replica-a")
	aggregate := func(values []any) any {
		return values[0].(cloneableInt) * 2
	}
	local := 21
	state := syncer.Bind(
		"load",
		func(string) fwkdl.Cloneable { return cloneableInt(local) },
		aggregate,
	)

	require.NoError(t, state.Set(context.Background(), "default/backend-0"))
	value, err := state.Get(context.Background(), "default/backend-0")

	require.NoError(t, err)
	assert.Equal(t, cloneableInt(42), value)

	local = 22
	value, err = state.Get(context.Background(), "default/backend-0")
	require.NoError(t, err)
	assert.Equal(t, cloneableInt(44), value)
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
