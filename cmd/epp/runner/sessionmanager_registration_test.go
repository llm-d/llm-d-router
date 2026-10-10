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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionmanager"
)

func TestSessionManagerRegisteredAsExplicitAlphaPlugin(t *testing.T) {
	(&Runner{}).registerInTreePlugins()
	require.Contains(t, fwkplugin.Registry, sessionmanager.PluginType)
	assert.Equal(t, fwkplugin.StabilityAlpha, fwkplugin.GetPluginStability(sessionmanager.PluginType))
	for _, producerType := range fwkplugin.DefaultProducerRegistry {
		assert.NotEqual(t, sessionmanager.PluginType, producerType)
	}
}
