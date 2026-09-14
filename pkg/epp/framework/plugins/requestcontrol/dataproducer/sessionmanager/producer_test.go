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
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

func testProducer(t *testing.T, correlation bool) *Producer {
	t.Helper()
	raw := fmt.Sprintf(`{
		"deploymentID":"prod-a",
		"hmacKeyFile":%q,
		"tokenProducer":"tokens",
		"eventCorrelationEnabled":%t
	}`, writeTestKey(t), correlation)
	created, err := Factory("sessions", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	require.NoError(t, err)
	producer, ok := created.(*Producer)
	require.True(t, ok)
	return producer
}

func eligibleRequest(alias string) *fwksched.InferenceRequest {
	request := &fwksched.InferenceRequest{
		TargetModel: "model",
		Body: &fwkrh.InferenceRequestBody{
			Payload:          fwkrh.PayloadMap{"messages": []any{}},
			TokenizedRequest: fwkrh.NewTokenizedRequest([][]uint32{{1, 2, 3}}),
		},
	}
	if alias != "" {
		request.PutAttribute(agentidentity.AgentIdentityKey, alias)
	}
	return request
}

func bindRequest(t *testing.T, producer *Producer, request *fwksched.InferenceRequest, address, port string) {
	t.Helper()
	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{Address: address, Port: port},
		&fwkdl.Metrics{},
		nil,
	)
	require.NoError(t, producer.PreRequest(t.Context(), request, &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}))
}

func TestProducePublishesIdentityAndEmptyPrefixRequest(t *testing.T) {
	t.Parallel()
	producer := testProducer(t, true)
	request := eligibleRequest("private-alias")
	require.NoError(t, producer.Produce(t.Context(), request, nil))

	identity, ok := requestcontrol.ReadSessionIdentity(request, "sessions")
	require.True(t, ok)
	assert.NotEmpty(t, identity.SessionTag)
	assert.NotContains(t, identity.SessionTag, "private-alias")

	cacheRequest, ok := fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
		request,
		requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions"),
	)
	require.True(t, ok)
	assert.NotEmpty(t, cacheRequest.Stamp)
	assert.False(t, cacheRequest.FullReport)
	assert.Equal(t, 3, cacheRequest.TotalTokens)
	assert.Empty(t, cacheRequest.Prefixes)
	assert.Equal(t, 1, producer.bindings.len())
}

func TestCacheNamespaceRemainsUnsetAndResetInvalidatesDiscoveredEndpoint(t *testing.T) {
	t.Parallel()
	producer := testProducer(t, true)
	assert.Empty(t, producer.CacheNamespace(
		kvevents.EventSource{Endpoint: "10.0.0.2:8000", ModelName: "model"}, ptr.To(0),
	))

	producer.bindings.put("stamp", "session", "model")
	require.True(t, producer.bindings.bindEndpoint("stamp", "10.0.0.2:8000"))
	require.NoError(t, producer.Reset(context.Background(), "10.0.0.2:8000"))
	known, _, mismatch, stale := producer.bindings.observe("stamp", "model", "10.0.0.2:8000")
	assert.True(t, known)
	assert.False(t, mismatch)
	assert.True(t, stale)
}

func TestProduceIdentityOnlyAndFailOpenShapes(t *testing.T) {
	t.Parallel()
	identityOnly := testProducer(t, false)
	request := eligibleRequest("alias")
	require.NoError(t, identityOnly.Produce(t.Context(), request, nil))
	_, identityOK := requestcontrol.ReadSessionIdentity(request, "sessions")
	assert.True(t, identityOK)
	_, cacheOK := fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
		request,
		requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions"),
	)
	assert.False(t, cacheOK)

	correlation := testProducer(t, true)
	missingAlias := eligibleRequest("")
	require.NoError(t, correlation.Produce(t.Context(), missingAlias, nil))
	assert.Empty(t, missingAlias.AttributeKeys())

	missingTokens := eligibleRequest("alias")
	missingTokens.Body.TokenizedRequest = nil
	require.NoError(t, correlation.Produce(t.Context(), missingTokens, nil))
	_, cacheOK = fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
		missingTokens,
		requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions"),
	)
	assert.False(t, cacheOK)

	multiPrompt := eligibleRequest("alias")
	multiPrompt.Body.TokenizedRequest = fwkrh.NewTokenizedRequest([][]uint32{{1}, {2}})
	require.NoError(t, correlation.Produce(t.Context(), multiPrompt, nil))
	_, cacheOK = fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
		multiPrompt,
		requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions"),
	)
	assert.False(t, cacheOK)

	unsupportedEnvelope := eligibleRequest("alias")
	unsupportedEnvelope.Body.Payload = fwkrh.RawPayload("opaque")
	require.NoError(t, correlation.Produce(t.Context(), unsupportedEnvelope, nil))
	_, cacheOK = fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
		unsupportedEnvelope,
		requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions"),
	)
	assert.False(t, cacheOK)
}

func TestProcessEventsCorrelatesKnownScopedStampOnly(t *testing.T) {
	t.Parallel()
	producer := testProducer(t, true)
	request := eligibleRequest("alias")
	require.NoError(t, producer.Produce(t.Context(), request, nil))
	bindRequest(t, producer, request, "pod-a", "8000")
	cacheRequest, ok := fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
		request,
		requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions"),
	)
	require.True(t, ok)
	source := kvevents.EventSource{ModelName: "model", Endpoint: "pod-a:8000"}
	event := &kvevents.BlockStoredEvent{
		SessionID:  ptr.To(cacheRequest.Stamp),
		BlockSize:  16,
		DeviceTier: "GPU",
	}
	require.NoError(t, producer.ProcessEvents(context.Background(), source, kvevents.EventBatch{
		Events: []kvevents.GenericEvent{event},
	}))
	assert.Zero(t, testutil.ToFloat64(producer.metrics.eventOutcomes.WithLabelValues("known")))

	event.BlockHashes = []uint64{1}
	require.NoError(t, producer.ProcessEvents(
		context.Background(),
		kvevents.EventSource{ModelName: "model", Endpoint: "pod-b:8000"},
		kvevents.EventBatch{Events: []kvevents.GenericEvent{event}},
	))
	assert.Equal(t, float64(1), testutil.ToFloat64(
		producer.metrics.eventOutcomes.WithLabelValues("request_mismatch"),
	))

	require.NoError(t, producer.ProcessEvents(context.Background(), source, kvevents.EventBatch{
		Events: []kvevents.GenericEvent{event},
	}))
	assert.Equal(t, float64(1), testutil.ToFloat64(producer.metrics.eventOutcomes.WithLabelValues("known")))

	require.NoError(t, producer.ProcessEvents(context.Background(), source, kvevents.EventBatch{
		Events: []kvevents.GenericEvent{event},
	}))
	assert.Equal(t, float64(1), testutil.ToFloat64(producer.metrics.eventOutcomes.WithLabelValues("duplicate")))

	unknown := "unknown"
	event.SessionID = &unknown
	require.NoError(t, producer.ProcessEvents(context.Background(), source, kvevents.EventBatch{
		Events: []kvevents.GenericEvent{event},
	}))
	assert.Equal(t, float64(1), testutil.ToFloat64(producer.metrics.eventOutcomes.WithLabelValues("unknown")))

	event.SessionID = ptr.To(cacheRequest.Stamp)
	source.ModelName = "other-model"
	require.NoError(t, producer.ProcessEvents(context.Background(), source, kvevents.EventBatch{
		Events: []kvevents.GenericEvent{event},
	}))
	assert.Equal(t, float64(2), testutil.ToFloat64(producer.metrics.eventOutcomes.WithLabelValues("request_mismatch")))

	require.NoError(t, producer.Reset(t.Context(), "pod-a:8000"))
	source.ModelName = "model"
	require.NoError(t, producer.ProcessEvents(context.Background(), source, kvevents.EventBatch{
		Events: []kvevents.GenericEvent{event},
	}))
	assert.Equal(t, float64(1), testutil.ToFloat64(producer.metrics.eventOutcomes.WithLabelValues("reset_stale")))
}

func TestProcessEventsFiltersMalformedStoresWithoutConsumingBinding(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*kvevents.BlockStoredEvent)
		valid  bool
	}{
		{name: "omitted legacy tier", valid: true},
		{name: "cpu", mutate: func(event *kvevents.BlockStoredEvent) { event.DeviceTier = "cpu" }},
		{name: "remote", mutate: func(event *kvevents.BlockStoredEvent) { event.Locality = "remote" }},
		{name: "owned", mutate: func(event *kvevents.BlockStoredEvent) { event.Ownership = "connector" }},
		{name: "invalid block size", mutate: func(event *kvevents.BlockStoredEvent) { event.BlockSize = 0 }},
		{name: "unsupported spec", mutate: func(event *kvevents.BlockStoredEvent) {
			event.KVCacheSpecKind = kvevents.KVCacheSpecKindSlidingWindow
		}},
		{name: "unstamped", mutate: func(event *kvevents.BlockStoredEvent) { event.SessionID = nil }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			producer := testProducer(t, true)
			request := eligibleRequest("alias")
			require.NoError(t, producer.Produce(t.Context(), request, nil))
			bindRequest(t, producer, request, "pod-a", "8000")
			cacheRequest, ok := fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
				request,
				requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions"),
			)
			require.True(t, ok)
			event := &kvevents.BlockStoredEvent{
				SessionID:   ptr.To(cacheRequest.Stamp),
				BlockHashes: []uint64{1},
				BlockSize:   16,
			}
			if test.mutate != nil {
				test.mutate(event)
			}
			source := kvevents.EventSource{ModelName: "model", Endpoint: "pod-a:8000"}
			require.NoError(t, producer.ProcessEvents(t.Context(), source, kvevents.EventBatch{
				Events: []kvevents.GenericEvent{event},
			}))
			known := testutil.ToFloat64(producer.metrics.eventOutcomes.WithLabelValues("known"))
			if test.valid {
				assert.Equal(t, float64(1), known)
				return
			}
			assert.Zero(t, known)
			require.NoError(t, producer.ProcessEvents(t.Context(), source, kvevents.EventBatch{
				Events: []kvevents.GenericEvent{&kvevents.BlockStoredEvent{
					SessionID:   ptr.To(cacheRequest.Stamp),
					BlockHashes: []uint64{1},
					BlockSize:   16,
				}},
			}))
			assert.Equal(t, float64(1), testutil.ToFloat64(
				producer.metrics.eventOutcomes.WithLabelValues("known"),
			))
		})
	}
}

func TestProcessEventsRejectsExpiredBinding(t *testing.T) {
	t.Parallel()
	producer := testProducer(t, true)
	now := time.Unix(100, 0)
	producer.bindings.now = func() time.Time { return now }
	request := eligibleRequest("alias")
	require.NoError(t, producer.Produce(t.Context(), request, nil))
	bindRequest(t, producer, request, "pod-a", "8000")
	cacheRequest, ok := fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
		request,
		requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions"),
	)
	require.True(t, ok)
	now = now.Add(producer.bindings.ttl)
	require.NoError(t, producer.ProcessEvents(
		t.Context(),
		kvevents.EventSource{ModelName: "model", Endpoint: "pod-a:8000"},
		kvevents.EventBatch{Events: []kvevents.GenericEvent{&kvevents.BlockStoredEvent{
			SessionID:   ptr.To(cacheRequest.Stamp),
			BlockHashes: []uint64{1},
			BlockSize:   16,
		}}},
	))
	assert.Equal(t, float64(1), testutil.ToFloat64(producer.metrics.eventOutcomes.WithLabelValues("unknown")))
}

func TestDumpStateContainsNoIdentityData(t *testing.T) {
	t.Parallel()
	producer := testProducer(t, true)
	request := eligibleRequest("private-alias")
	require.NoError(t, producer.Produce(t.Context(), request, nil))
	identity, ok := requestcontrol.ReadSessionIdentity(request, "sessions")
	require.True(t, ok)
	cacheRequest, ok := fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
		request,
		requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions"),
	)
	require.True(t, ok)

	dump, err := producer.DumpState()
	require.NoError(t, err)
	assert.JSONEq(t, `{"activeBindings":1,"maxBindings":100000}`, string(dump))
	assert.NotContains(t, string(dump), "private-alias")
	assert.NotContains(t, string(dump), identity.SessionTag)
	assert.NotContains(t, string(dump), cacheRequest.Stamp)
}

func TestMetricsContainNoIdentityData(t *testing.T) {
	t.Parallel()
	registry := prometheus.NewRegistry()
	raw := fmt.Sprintf(`{
		"deploymentID":"prod-a",
		"hmacKeyFile":%q,
		"tokenProducer":"tokens",
		"eventCorrelationEnabled":true
	}`, writeTestKey(t))
	handle := fwkplugin.NewEppHandle(t.Context(), nil, fwkplugin.WithMetricsRecorder(registry))
	created, err := Factory("sessions", fwkplugin.StrictDecoder(json.RawMessage(raw)), handle)
	require.NoError(t, err)
	producer := created.(*Producer)
	request := eligibleRequest("private-alias")
	require.NoError(t, producer.Produce(t.Context(), request, nil))
	identity, _ := requestcontrol.ReadSessionIdentity(request, "sessions")
	cacheRequest, _ := fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
		request,
		requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName("sessions"),
	)

	families, err := registry.Gather()
	require.NoError(t, err)
	require.NotEmpty(t, families)
	rendered := fmt.Sprint(families)
	assert.Contains(t, rendered, "session_manager_identity_total")
	assert.Contains(t, rendered, "session_manager_cache_request_total")
	assert.Contains(t, rendered, "session_manager_active_bindings")
	assert.Contains(t, rendered, `value:"published"`)
	assert.NotContains(t, rendered, "private-alias")
	assert.NotContains(t, rendered, identity.SessionTag)
	assert.NotContains(t, rendered, cacheRequest.Stamp)
}
