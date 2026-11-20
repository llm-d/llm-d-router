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

// THE STARTUP REFUSALS. This policy charges the blocks thread identity publishes, so a config
// that gives it nothing to charge — or two block lengths — is refused at startup, not discovered
// as wrong costs under load.

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// stubPlugin is a minimal plugin carrying only a TypedName: the gate inspects nothing else.
type stubPlugin struct{ tn fwkplugin.TypedName }

func (s *stubPlugin) TypedName() fwkplugin.TypedName { return s.tn }

// threadIdentityAt builds a real thread identity plugin at a chosen block length: its block size is
// the number this policy has to agree with.
func threadIdentityAt(t *testing.T, blockSize int) *threadidentity.Plugin {
	t.Helper()
	params := fmt.Sprintf(`{"blockSizeTokens":%d,"tokenProducer":{"modelName":"test-model","vllm":{"url":"http://127.0.0.1:1"}}}`, blockSize)
	built, err := threadidentity.Factory("thread", json.NewDecoder(strings.NewReader(params)), testutils.NewTestHandle(t.Context()))
	require.NoError(t, err, "thread identity must build from its parameters alone")
	return built.(*threadidentity.Plugin)
}

func TestBlockSizeMustMatchThreadIdentity(t *testing.T) {
	build := func(t *testing.T) *GroupScheduler {
		t.Helper()
		p, err := Factory("gs", nil, testutils.NewTestHandle(t.Context()))
		require.NoError(t, err, "the gate is not at construction")
		return p.(*GroupScheduler)
	}

	t.Run("an agreeing block size passes", func(t *testing.T) {
		gs := build(t)
		gs.handle.AddPlugin("thread", threadIdentityAt(t, gs.cfg.BlockSizeTokens))
		require.NoError(t, gs.validateThreadProducer())
	})
	t.Run("a disagreeing block size is refused, naming both", func(t *testing.T) {
		gs := build(t)
		other := gs.cfg.BlockSizeTokens / 2
		gs.handle.AddPlugin("thread", threadIdentityAt(t, other))
		err := gs.validateThreadProducer()
		require.Error(t, err, "two block lengths must not start together")
		require.Contains(t, err.Error(), "block size")
		require.Contains(t, err.Error(), fmt.Sprintf("%d", other))
		require.Contains(t, err.Error(), fmt.Sprintf("%d", gs.cfg.BlockSizeTokens))
	})
	t.Run("no thread identity is refused", func(t *testing.T) {
		gs := build(t)
		err := gs.validateThreadProducer()
		require.Error(t, err, "with no thread identity every request is anonymous and every head passes through")
		require.Contains(t, err.Error(), threadidentity.PluginType)
	})
	t.Run("agent identity alone is refused", func(t *testing.T) {
		gs := build(t)
		gs.handle.AddPlugin("agent", &stubPlugin{tn: fwkplugin.TypedName{Type: agentidentity.PluginType, Name: "agent"}})
		err := gs.validateThreadProducer()
		require.Error(t, err, "an agent id is a multi-chain member: this policy keeps one chain per id")
		require.Contains(t, err.Error(), threadidentity.PluginType)
	})
	t.Run("agent identity beside thread identity is allowed here", func(t *testing.T) {
		gs := build(t)
		gs.handle.AddPlugin("thread", threadIdentityAt(t, gs.cfg.BlockSizeTokens))
		gs.handle.AddPlugin("agent", &stubPlugin{tn: fwkplugin.TypedName{Type: agentidentity.PluginType, Name: "agent"}})
		// the flow-key attribute's single-owner rule is thread identity's own startup check
		require.NoError(t, gs.validateThreadProducer())
	})
	t.Run("no handle: unit mode, the gate is skipped", func(t *testing.T) {
		p, err := Factory("gs", nil, nil)
		require.NoError(t, err)
		require.NoError(t, p.(*GroupScheduler).validateThreadProducer())
	})
}
