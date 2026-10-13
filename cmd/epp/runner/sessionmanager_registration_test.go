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

package runner

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionmanager"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/sessionidentity"
)

func TestSessionPluginsRegisteredAsExplicitAlphaPlugins(t *testing.T) {
	(&Runner{}).registerInTreePlugins()
	for _, pluginType := range []string{sessionidentity.PluginType, sessionmanager.PluginType} {
		require.Contains(t, fwkplugin.Registry, pluginType)
		assert.Equal(t, fwkplugin.StabilityAlpha, fwkplugin.GetPluginStability(pluginType))
		for _, producerType := range fwkplugin.DefaultProducerRegistry {
			assert.NotEqual(t, pluginType, producerType)
		}
	}
}

func TestSessionPluginChainValidatesInExecutionOrder(t *testing.T) {
	keyPath := filepath.Join(t.TempDir(), "hmac-key")
	key := base64.RawURLEncoding.EncodeToString(make([]byte, 32))
	require.NoError(t, os.WriteFile(keyPath, []byte(key), 0o600))

	agent, err := agentidentity.PluginFactory(agentidentity.PluginType, nil, nil)
	require.NoError(t, err)
	identity, err := sessionidentity.Factory(
		sessionidentity.PluginType,
		fwkplugin.StrictDecoder(json.RawMessage(
			`{"deploymentID":"prod-a","hmacKeyFile":`+string(mustJSON(t, keyPath))+`}`,
		)),
		nil,
	)
	require.NoError(t, err)
	manager, err := sessionmanager.Factory(sessionmanager.PluginType, nil, nil)
	require.NoError(t, err)

	ordered, err := datalayer.ValidateAndOrderDataDependencies([]fwkplugin.Plugin{
		manager,
		identity,
		agent,
	})
	require.NoError(t, err)
	assert.Equal(t, []string{
		agent.TypedName().String(),
		identity.TypedName().String(),
		manager.TypedName().String(),
	}, ordered)
}

func mustJSON(t *testing.T, value string) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	require.NoError(t, err)
	return raw
}
