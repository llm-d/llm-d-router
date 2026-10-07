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

package thunderagent

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	eppdatalayer "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

// Agent identity is a required input, so configuration loading fails without
// an identity provider.
func TestAgentIdentityIsARequiredDependency(t *testing.T) {
	a := newTestAgent(testConfig())
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	handle.AddPlugin(a.TypedName().Name, a)

	err := eppdatalayer.CreateMissingDataProducers(context.Background(),
		map[string]string{}, map[string]fwkplugin.FactoryFunc{}, handle)
	require.ErrorIs(t, err, eppdatalayer.ErrNoDefaultProducer)
	require.ErrorContains(t, err, agentidentity.AgentIdentityKey.String())
}
