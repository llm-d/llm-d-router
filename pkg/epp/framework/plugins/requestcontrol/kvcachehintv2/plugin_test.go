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

package kvcachehintv2

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrinterturn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/interturn"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
)

// mainPrediction is a human-paced main-session fit of the
// session-interturn-latency-producer.
var mainPrediction = attrinterturn.InterTurnPrediction{
	SessionType: "main",
	LogMean:     2.28,
	LogStd:      1.34,
}

// boundedConfig protects a 1000-token head per session out of a 2500-token
// budget, so two sessions fit and a third does not.
var boundedConfig = Config{
	ParentSessionHeader: "x-parent-session-id",
	Queues: []QueueConfig{{
		SessionType: "main",
		Retain:      &RetainConfig{Quantile: 0.9, MinTTL: "10s", MaxTTL: "1m"},
	}},
	MaxProtectedPrefixTokens: 1000,
	ProtectionBudgetTokens:   2500,
}

func newTestPlugin(t *testing.T, cfg Config) *Plugin {
	t.Helper()
	p, err := NewPlugin("test", cfg)
	require.NoError(t, err)
	return p
}

// sessionRequest carries the attributes the session-id-producer and the
// session-interturn-latency-producer publish, plus a tokenized prompt of
// promptTokens tokens.
func sessionRequest(sessionID string, prediction *attrinterturn.InterTurnPrediction, promptTokens int) *fwksched.InferenceRequest {
	request := &fwksched.InferenceRequest{
		RequestID: "req-1",
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{"model": "m", "messages": []any{}},
		},
	}
	if promptTokens > 0 {
		request.Body.TokenizedRequest = &fwkrh.TokenizedRequest{
			Prompts: []fwkrh.PromptTokens{{TokenIDs: make([]uint32, promptTokens)}},
		}
	}
	if sessionID != "" {
		request.PutAttribute(attrsession.SessionIDDataKey, attrsession.SessionID(sessionID))
	}
	if prediction != nil {
		request.PutAttribute(attrinterturn.InterTurnPredictionDataKey, *prediction)
	}
	return request
}

func directive(t *testing.T, request *fwksched.InferenceRequest) map[string]any {
	t.Helper()
	payload, ok := request.Body.Payload.AsMap()
	require.True(t, ok)
	directives, ok := payload[retentionDirectivesField].([]any)
	require.True(t, ok, "retention_directives must be present")
	require.Len(t, directives, 1)
	d, ok := directives[0].(map[string]any)
	require.True(t, ok)
	return d
}

func TestFactory(t *testing.T) {
	t.Parallel()

	valid := `{
		"parentSessionHeader": "x-parent-session-id",
		"queues": [{"sessionType": "main", "retain": {"quantile": 0.9, "minTTL": "10s", "maxTTL": "1m"}}],
		"maxProtectedPrefixTokens": 16384,
		"protectionBudgetTokens": 1000000
	}`
	plg, err := Factory(PluginType, fwkplugin.StrictDecoder(json.RawMessage(valid)), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, plg.TypedName().Type)

	for name, params := range map[string]string{
		"negative prefix cap":     `{"maxProtectedPrefixTokens": -1}`,
		"budget below prefix cap": `{"maxProtectedPrefixTokens": 1000, "protectionBudgetTokens": 500}`,
		"quantile out of range":   `{"queues":[{"sessionType":"a","retain":{"quantile":1.5}}]}`,
		"unknown field":           `{"unknown":true}`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			_, err := Factory(PluginType, fwkplugin.StrictDecoder(json.RawMessage(params)), nil)
			assert.Error(t, err)
		})
	}
}

func TestPreRequest_BoundsProtectedRange(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		promptTokens int
		rawBody      []byte
		wantEnd      int64
	}{
		{name: "long prompt clamped to cap", promptTokens: 50000, wantEnd: 1000},
		{name: "short prompt protected fully", promptTokens: 400, wantEnd: 400},
		{name: "raw-body estimate", rawBody: bytes.Repeat([]byte("x"), 2000), wantEnd: 500},
		{name: "unknown size charges the cap", wantEnd: 1000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			p := newTestPlugin(t, boundedConfig)
			request := sessionRequest("s1", &mainPrediction, tt.promptTokens)
			request.Body.RawBody = tt.rawBody

			require.NoError(t, p.PreRequest(t.Context(), request, nil))

			d := directive(t, request)
			assert.Equal(t, 0, d["start"])
			assert.Equal(t, tt.wantEnd, d["end"])
			assert.Equal(t, defaultPriority, d["priority"])
		})
	}
}

func TestPreRequest_BudgetExhausted(t *testing.T) {
	t.Parallel()

	p := newTestPlugin(t, boundedConfig)

	// Two 1000-token reservations fit the 2500-token budget.
	for _, session := range []string{"s1", "s2"} {
		request := sessionRequest(session, &mainPrediction, 50000)
		require.NoError(t, p.PreRequest(t.Context(), request, nil))
		assert.True(t, request.Body.Mutated, "session %s must be protected", session)
	}

	// A third would need 1000 more; only 500 remain.
	rejected := sessionRequest("s3", &mainPrediction, 50000)
	require.NoError(t, p.PreRequest(t.Context(), rejected, nil))
	assert.False(t, rejected.Body.Mutated)

	// Renewing an existing reservation costs nothing and still emits.
	renewal := sessionRequest("s1", &mainPrediction, 50000)
	require.NoError(t, p.PreRequest(t.Context(), renewal, nil))
	assert.True(t, renewal.Body.Mutated)
	assert.Equal(t, int64(1000), directive(t, renewal)["end"])
}

func TestPreRequest_BudgetFreesOnExpiry(t *testing.T) {
	t.Parallel()

	p := newTestPlugin(t, boundedConfig)
	now := time.Now()
	p.budget.now = func() time.Time { return now }

	for _, session := range []string{"s1", "s2"} {
		request := sessionRequest(session, &mainPrediction, 50000)
		require.NoError(t, p.PreRequest(t.Context(), request, nil))
		require.True(t, request.Body.Mutated)
	}
	rejected := sessionRequest("s3", &mainPrediction, 50000)
	require.NoError(t, p.PreRequest(t.Context(), rejected, nil))
	require.False(t, rejected.Body.Mutated)

	// The boundedConfig maxTTL is 1m, so both reservations lapse.
	now = now.Add(2 * time.Minute)
	admitted := sessionRequest("s3", &mainPrediction, 50000)
	require.NoError(t, p.PreRequest(t.Context(), admitted, nil))
	assert.True(t, admitted.Body.Mutated)
	used, _ := p.budget.usage()
	assert.Equal(t, int64(1000), used)
}

func TestBudgetTracker_GrowthFallsBackWhenFull(t *testing.T) {
	t.Parallel()

	b := newBudgetTracker(1000)
	charged, ok := b.reserve("s1", 800, time.Minute)
	require.True(t, ok)
	assert.Equal(t, int64(800), charged)

	// Growing to 1200 would exceed the budget; the reservation keeps its
	// size and only refreshes the TTL.
	charged, ok = b.reserve("s1", 1200, time.Minute)
	require.True(t, ok)
	assert.Equal(t, int64(800), charged)

	// A smaller renewal keeps the larger charge: the wider head is still
	// protected on the backend until its TTL lapses.
	charged, ok = b.reserve("s1", 100, time.Minute)
	require.True(t, ok)
	assert.Equal(t, int64(800), charged)

	used, budget := b.usage()
	assert.Equal(t, int64(800), used)
	assert.Equal(t, int64(1000), budget)
}

func TestPreRequest_SkipsWithoutAttributes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request *fwksched.InferenceRequest
	}{
		{name: "no session id", request: sessionRequest("", &mainPrediction, 100)},
		{name: "no prediction", request: sessionRequest("s1", nil, 100)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			p := newTestPlugin(t, boundedConfig)

			require.NoError(t, p.PreRequest(t.Context(), tt.request, nil))
			assert.False(t, tt.request.Body.Mutated)
		})
	}
}

func TestPreRequest_ClientDirectivesWin(t *testing.T) {
	t.Parallel()

	p := newTestPlugin(t, boundedConfig)
	request := sessionRequest("s1", &mainPrediction, 100)
	clientDirectives := []any{map[string]any{"start": 0, "priority": 99}}
	payload, _ := request.Body.Payload.AsMap()
	payload[retentionDirectivesField] = clientDirectives

	require.NoError(t, p.PreRequest(t.Context(), request, nil))

	assert.False(t, request.Body.Mutated)
	assert.Equal(t, clientDirectives, payload[retentionDirectivesField])
	assert.NotContains(t, payload, retentionScopeField)

	// A rejected mutation must not hold budget.
	used, _ := p.budget.usage()
	assert.Equal(t, int64(0), used)
}

func TestPreRequest_ParentSessionScope(t *testing.T) {
	t.Parallel()

	p := newTestPlugin(t, boundedConfig)
	request := sessionRequest("sub-1", &mainPrediction, 100)
	request.Headers = map[string]string{"x-parent-session-id": "main-1"}

	require.NoError(t, p.PreRequest(t.Context(), request, nil))

	payload, _ := request.Body.Payload.AsMap()
	assert.Equal(t, "main-1", payload[retentionScopeField])
}
