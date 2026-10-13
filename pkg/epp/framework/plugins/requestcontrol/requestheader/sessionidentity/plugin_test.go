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

package sessionidentity

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

func testPlugin(t *testing.T, handle fwkplugin.Handle) *Plugin {
	t.Helper()
	raw := fmt.Sprintf(`{"deploymentID":"prod-a","hmacKeyFile":%q}`, writeTestKey(t))
	created, err := Factory(PluginType, fwkplugin.StrictDecoder(json.RawMessage(raw)), handle)
	require.NoError(t, err)
	plugin, ok := created.(*Plugin)
	require.True(t, ok)
	return plugin
}

func requestWithAlias(alias string) *fwksched.InferenceRequest {
	request := &fwksched.InferenceRequest{}
	if alias != "" {
		request.PutAttribute(agentidentity.AgentIdentityKey, alias)
	}
	return request
}

func TestRequestHeaderPublishesScopedIdentityAndStringTag(t *testing.T) {
	t.Parallel()
	plugin := testPlugin(t, nil)
	request := requestWithAlias("private-alias")

	require.NoError(t, plugin.RequestHeader(t.Context(), request))

	identity, ok := requestcontrol.ReadSessionIdentity(request, PluginType)
	require.True(t, ok)
	assert.NotEmpty(t, identity.SessionTag)
	assert.NotContains(t, identity.SessionTag, "private-alias")

	tag, ok := fwksched.ReadRequestAttribute[string](
		request,
		SessionTagDataKey.WithNonEmptyProducerName(PluginType),
	)
	require.True(t, ok)
	assert.Equal(t, identity.SessionTag, tag)
}

func TestRequestHeaderFailsOpenWithoutIdentity(t *testing.T) {
	t.Parallel()
	plugin := testPlugin(t, nil)
	request := requestWithAlias("")

	require.NoError(t, plugin.RequestHeader(t.Context(), request))
	assert.Empty(t, request.AttributeKeys())
}

func TestProducesDeclaresIdentityAttributes(t *testing.T) {
	t.Parallel()
	plugin := testPlugin(t, nil)
	assert.Equal(t, map[fwkplugin.DataKey]any{
		requestcontrol.SessionIdentityDataKey.WithNonEmptyProducerName(PluginType): requestcontrol.SessionIdentity{},
		SessionTagDataKey.WithNonEmptyProducerName(PluginType):                     "",
	}, plugin.Produces())
}

func TestMetricsContainNoIdentityData(t *testing.T) {
	t.Parallel()
	registry := prometheus.NewRegistry()
	handle := fwkplugin.NewEppHandle(t.Context(), nil, fwkplugin.WithMetricsRecorder(registry))
	plugin := testPlugin(t, handle)
	request := requestWithAlias("private-alias")
	require.NoError(t, plugin.RequestHeader(t.Context(), request))

	identity, _ := requestcontrol.ReadSessionIdentity(request, PluginType)
	families, err := registry.Gather()
	require.NoError(t, err)
	rendered := fmt.Sprint(families)
	assert.Contains(t, rendered, "session_identity_total")
	assert.NotContains(t, rendered, "private-alias")
	assert.NotContains(t, rendered, identity.SessionTag)
}
