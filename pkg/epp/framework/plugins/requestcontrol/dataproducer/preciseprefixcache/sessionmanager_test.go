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

package preciseprefixcache

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionmanager"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

func TestSessionManagerPublishesZeroPrefixEvidenceAndStampsRequest(t *testing.T) {
	t.Parallel()
	keyPath := filepath.Join(t.TempDir(), "key")
	key := base64.RawURLEncoding.EncodeToString(make([]byte, 32))
	require.NoError(t, os.WriteFile(keyPath, []byte(key), 0o600))

	handle := fwkplugin.NewEppHandle(t.Context(), nil)
	managerJSON := fmt.Sprintf(`{
		"deploymentID":"test",
		"hmacKeyFile":%q,
		"tokenProducer":"tokens",
		"eventCorrelationEnabled":true
	}`, keyPath)
	created, err := sessionmanager.Factory("sessions", fwkplugin.StrictDecoder(json.RawMessage(managerJSON)), handle)
	require.NoError(t, err)
	manager := created.(*sessionmanager.Producer)
	assert.Empty(t, manager.CacheNamespace(
		kvevents.EventSource{Endpoint: "10.0.0.1:8000", ModelName: "model"},
		nil,
	))
	handle.AddPlugin("sessions", manager)

	created, err = PluginFactory(
		"cache",
		fwkplugin.StrictDecoder(json.RawMessage(`{"sessionManager":"sessions"}`)),
		handle,
	)
	require.NoError(t, err)
	cache := created.(*Producer)

	request := &fwksched.InferenceRequest{
		TargetModel: "model",
		Body: &fwkrh.InferenceRequestBody{
			Payload:          fwkrh.PayloadMap{"model": "model", "prompt": "hello"},
			TokenizedRequest: fwkrh.NewTokenizedRequest([][]uint32{{1, 2, 3}}),
		},
	}
	request.PutAttribute(agentidentity.AgentIdentityKey, "private-alias")
	require.NoError(t, manager.Produce(t.Context(), request, nil))

	endpoints := freshEndpoints()
	require.NoError(t, cache.Produce(t.Context(), request, endpoints))
	for _, endpoint := range endpoints {
		raw, ok := endpoint.Get(cache.dk)
		require.True(t, ok)
		info := raw.(*attrprefix.PrefixCacheMatchInfo)
		assert.Zero(t, info.MatchBlocks())
		assert.Zero(t, info.CachedBlockCount())
	}

	require.NoError(t, cache.PreRequest(t.Context(), request, nil))
	payload := request.Body.Payload.(fwkrh.PayloadMap)
	assert.NotEqual(t, "private-alias", payload["session_id"])
	assert.NotEmpty(t, payload["session_id"])
	xargs := payload["vllm_xargs"].(map[string]any)
	assert.Equal(t, "incremental", xargs["kv_cache_report_mode"])
}
