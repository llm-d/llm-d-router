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

package sessionmanager

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
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

func testProducer(t *testing.T) *Producer {
	t.Helper()
	raw := fmt.Sprintf(`{"deploymentID":"prod-a","hmacKeyFile":%q}`, writeTestKey(t))
	created, err := Factory("session-manager", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	require.NoError(t, err)
	producer, ok := created.(*Producer)
	require.True(t, ok)
	return producer
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
	producer := testProducer(t)
	request := requestWithAlias("private-alias")

	require.NoError(t, producer.RequestHeader(t.Context(), request))

	identity, ok := requestcontrol.ReadSessionIdentity(request, "session-manager")
	require.True(t, ok)
	assert.NotEmpty(t, identity.SessionTag)
	assert.NotContains(t, identity.SessionTag, "private-alias")

	tag, ok := fwksched.ReadRequestAttribute[string](
		request,
		SessionTagDataKey.WithNonEmptyProducerName("session-manager"),
	)
	require.True(t, ok)
	assert.Equal(t, identity.SessionTag, tag)
}

func TestRequestHeaderFailsOpenWithoutIdentity(t *testing.T) {
	t.Parallel()
	producer := testProducer(t)
	request := requestWithAlias("")

	require.NoError(t, producer.RequestHeader(t.Context(), request))
	assert.Empty(t, request.AttributeKeys())
}

func TestProducePublishesStampedCacheRequestOnlyAfterIdentity(t *testing.T) {
	t.Parallel()
	producer := testProducer(t)
	key := attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName("session-manager")

	missing := requestWithAlias("")
	require.NoError(t, producer.Produce(t.Context(), missing, nil))
	_, ok := fwksched.ReadRequestAttribute[attrsession.SessionCacheRequest](missing, key)
	assert.False(t, ok)

	request := requestWithAlias("alias")
	require.NoError(t, producer.RequestHeader(t.Context(), request))
	require.NoError(t, producer.Produce(t.Context(), request, nil))
	cacheRequest, ok := fwksched.ReadRequestAttribute[attrsession.SessionCacheRequest](request, key)
	require.True(t, ok)
	assert.NotEmpty(t, cacheRequest.SessionID)
	assert.False(t, cacheRequest.FullReport)
	assert.Zero(t, cacheRequest.TotalTokens)
	assert.Empty(t, cacheRequest.Prefixes)
}

func TestProducesDeclaresAllRequestAttributes(t *testing.T) {
	t.Parallel()
	producer := testProducer(t)
	assert.Equal(t, map[fwkplugin.DataKey]any{
		requestcontrol.SessionIdentityDataKey.WithNonEmptyProducerName("session-manager"):  requestcontrol.SessionIdentity{},
		SessionTagDataKey.WithNonEmptyProducerName("session-manager"):                      "",
		attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName("session-manager"): attrsession.SessionCacheRequest{},
	}, producer.Produces())
}

func TestMetricsContainNoIdentityData(t *testing.T) {
	t.Parallel()
	registry := prometheus.NewRegistry()
	raw := fmt.Sprintf(`{"deploymentID":"prod-a","hmacKeyFile":%q}`, writeTestKey(t))
	handle := fwkplugin.NewEppHandle(t.Context(), nil, fwkplugin.WithMetricsRecorder(registry))
	created, err := Factory("session-manager", fwkplugin.StrictDecoder(json.RawMessage(raw)), handle)
	require.NoError(t, err)
	producer := created.(*Producer)
	request := requestWithAlias("private-alias")
	require.NoError(t, producer.RequestHeader(t.Context(), request))
	require.NoError(t, producer.Produce(t.Context(), request, nil))

	identity, _ := requestcontrol.ReadSessionIdentity(request, "session-manager")
	cacheRequest, _ := fwksched.ReadRequestAttribute[attrsession.SessionCacheRequest](
		request,
		attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName("session-manager"),
	)
	families, err := registry.Gather()
	require.NoError(t, err)
	rendered := fmt.Sprint(families)
	assert.Contains(t, rendered, "session_manager_identity_total")
	assert.Contains(t, rendered, "session_manager_cache_request_total")
	assert.NotContains(t, rendered, "session_manager_event_total")
	assert.NotContains(t, rendered, "private-alias")
	assert.NotContains(t, rendered, identity.SessionTag)
	assert.NotContains(t, rendered, cacheRequest.SessionID)
}
