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

package groupsched

import (
	"fmt"
	"testing"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	testutils "github.com/llm-d/llm-d-router/test/utils"
	"github.com/stretchr/testify/require"
)

// stubPlugin is a minimal plugin carrying only a TypedName (the startup gate inspects nothing
// else).
type stubPlugin struct{ tn fwkplugin.TypedName }

func (s *stubPlugin) TypedName() fwkplugin.TypedName { return s.tn }

// TestBlockSizeMustMatchThreadIdentity: the startup gate. The scheduler charges the blocks thread
// identity publishes, so the two block lengths are ONE setting that lives in two configs — a
// disagreement is refused at startup rather than discovered as wrong costs under load.
func TestBlockSizeMustMatchThreadIdentity(t *testing.T) {
	build := func(t *testing.T) *GroupScheduler {
		t.Helper()
		p, err := Factory("gs", nil, testutils.NewTestHandle(t.Context()))
		if err != nil {
			t.Fatalf("Factory must succeed (the gate is not at construction): %v", err)
		}
		return p.(*GroupScheduler)
	}

	t.Run("an agreeing block size passes", func(t *testing.T) {
		gs := build(t)
		gs.handle.AddPlugin("thread", newThreadIdentityAt(t, gs.cfg.BlockSizeTokens))
		require.NoError(t, gs.validateThreadProducer())
	})
	t.Run("a disagreeing block size is refused, naming both", func(t *testing.T) {
		gs := build(t)
		gs.handle.AddPlugin("thread", newThreadIdentityAt(t, gs.cfg.BlockSizeTokens/2))
		err := gs.validateThreadProducer()
		require.Error(t, err, "two block lengths must not start together")
		require.Contains(t, err.Error(), "block size")
		require.Contains(t, err.Error(), fmt.Sprintf("%d", gs.cfg.BlockSizeTokens/2))
		require.Contains(t, err.Error(), fmt.Sprintf("%d", gs.cfg.BlockSizeTokens))
	})
	t.Run("a token producer is no longer required", func(t *testing.T) {
		gs := build(t)
		gs.handle.AddPlugin("thread", newThreadIdentityAt(t, gs.cfg.BlockSizeTokens))
		require.NoError(t, gs.validateThreadProducer(),
			"the scheduler reads the blocks thread identity rendered: it needs no token producer of its own")
	})
	t.Run("nil handle skips the gate (unit/test mode)", func(t *testing.T) {
		p, err := Factory("gs", nil, nil)
		if err != nil {
			t.Fatalf("Factory with nil handle: %v", err)
		}
		if err := p.(*GroupScheduler).validateThreadProducer(); err != nil {
			t.Fatalf("nil handle must skip the gate: %v", err)
		}
	})
}
