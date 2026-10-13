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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/sessionidentity"
)

func testProducer(t *testing.T, identityProducer string, handle fwkplugin.Handle) *Producer {
	t.Helper()
	var decoder *json.Decoder
	if identityProducer != "" {
		decoder = fwkplugin.StrictDecoder(json.RawMessage(
			fmt.Sprintf(`{"sessionIdentityProducer":%q}`, identityProducer),
		))
	}
	created, err := Factory(PluginType, decoder, handle)
	require.NoError(t, err)
	producer, ok := created.(*Producer)
	require.True(t, ok)
	return producer
}

func putIdentity(request *fwksched.InferenceRequest, producer string) {
	identity := requestcontrol.SessionIdentity{
		SessionTag:     "opaque-tag",
		IdentitySource: requestcontrol.IdentitySourceAgentIdentityAttribute,
		ScopeVersion:   "opaque-scope",
	}
	request.PutAttribute(
		requestcontrol.SessionIdentityDataKey.WithNonEmptyProducerName(producer),
		identity,
	)
}

func TestProducePublishesStampedCacheRequestAfterIdentity(t *testing.T) {
	t.Parallel()
	producer := testProducer(t, "", nil)
	key := attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName(PluginType)

	missing := &fwksched.InferenceRequest{}
	require.NoError(t, producer.Produce(t.Context(), missing, nil))
	_, ok := fwksched.ReadRequestAttribute[attrsession.SessionCacheRequest](missing, key)
	assert.False(t, ok)

	request := &fwksched.InferenceRequest{}
	putIdentity(request, sessionidentity.PluginType)
	require.NoError(t, producer.Produce(t.Context(), request, nil))
	cacheRequest, ok := fwksched.ReadRequestAttribute[attrsession.SessionCacheRequest](request, key)
	require.True(t, ok)
	assert.NotEmpty(t, cacheRequest.SessionID)
	assert.False(t, cacheRequest.FullReport)
	assert.Zero(t, cacheRequest.TotalTokens)
	assert.Empty(t, cacheRequest.Prefixes)
}

func TestConsumesNamedSessionIdentityProducer(t *testing.T) {
	t.Parallel()
	producer := testProducer(t, "tenant-identity", nil)
	assert.Equal(t, fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			requestcontrol.SessionIdentityDataKey.WithNonEmptyProducerName("tenant-identity"): requestcontrol.SessionIdentity{},
		},
	}, producer.Consumes())
}

func TestProducesDeclaresOnlyCacheRequest(t *testing.T) {
	t.Parallel()
	producer := testProducer(t, "", nil)
	assert.Equal(t, map[fwkplugin.DataKey]any{
		attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName(PluginType): attrsession.SessionCacheRequest{},
	}, producer.Produces())
}

func TestMetricsContainNoIdentityData(t *testing.T) {
	t.Parallel()
	registry := prometheus.NewRegistry()
	handle := fwkplugin.NewEppHandle(t.Context(), nil, fwkplugin.WithMetricsRecorder(registry))
	producer := testProducer(t, "", handle)
	request := &fwksched.InferenceRequest{}
	putIdentity(request, sessionidentity.PluginType)
	require.NoError(t, producer.Produce(t.Context(), request, nil))

	cacheRequest, _ := fwksched.ReadRequestAttribute[attrsession.SessionCacheRequest](
		request,
		attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName(PluginType),
	)
	families, err := registry.Gather()
	require.NoError(t, err)
	rendered := fmt.Sprint(families)
	assert.Contains(t, rendered, "session_manager_cache_request_total")
	assert.NotContains(t, rendered, "session_identity_total")
	assert.NotContains(t, rendered, "opaque-tag")
	assert.NotContains(t, rendered, cacheRequest.SessionID)
}
