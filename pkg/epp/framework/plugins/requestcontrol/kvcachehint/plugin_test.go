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

package kvcachehint

import (
	"encoding/json"
	"math"
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

// subagentPrediction is the Claude Code subagent-trace fit of the
// session-interturn-latency-producer.
var subagentPrediction = attrinterturn.InterTurnPrediction{
	SessionType: "agentic-subagent",
	LogMean:     0.69,
	LogStd:      1.13,
}

// subagentConfig mirrors the README example: a machine-paced queue whose KV
// never outlives seconds-scale gaps.
var subagentConfig = Config{
	ParentSessionHeader: "x-parent-session-id",
	Queues: []QueueConfig{{
		SessionType: "agentic-subagent",
		Retain:      &RetainConfig{Quantile: 0.95, MinTTL: "1s", MaxTTL: "10s"},
	}},
}

func newTestPlugin(t *testing.T, cfg Config) *Plugin {
	t.Helper()
	p, err := NewPlugin("test", cfg)
	require.NoError(t, err)
	return p
}

// sessionRequest carries the attributes the session-id-producer and the
// session-interturn-latency-producer publish.
func sessionRequest(sessionID string, prediction *attrinterturn.InterTurnPrediction) *fwksched.InferenceRequest {
	request := &fwksched.InferenceRequest{
		RequestID: "req-1",
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{"model": "m", "messages": []any{}},
		},
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
		"queues": [
			{"sessionType": "agentic-subagent",
			 "retain": {"quantile": 0.95, "minTTL": "1s", "maxTTL": "10s"}}
		]
	}`
	plg, err := Factory("kv-cache-hint", fwkplugin.StrictDecoder(json.RawMessage(valid)), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, plg.TypedName().Type)

	for name, params := range map[string]string{
		"quantile out of range": `{"queues":[{"sessionType":"a","retain":{"quantile":1.5}}]}`,
		"priority out of range": `{"queues":[{"sessionType":"a","retain":{"priority":200}}]}`,
		"min above max":         `{"queues":[{"sessionType":"a","retain":{"minTTL":"1m","maxTTL":"10s"}}]}`,
		"bad duration":          `{"queues":[{"sessionType":"a","retain":{"minTTL":"soon"}}]}`,
		// The second entry lacks sessionType; the first cannot, because
		// decoding merges it over the default agentic queue.
		"missing sessionType":   `{"queues":[{"sessionType":"a"},{"retain":{"quantile":0.9}}]}`,
		"duplicate sessionType": `{"queues":[{"sessionType":"a"},{"sessionType":"A"}]}`,
		"unknown field":         `{"unknown":true}`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			_, err := Factory("kv-cache-hint", fwkplugin.StrictDecoder(json.RawMessage(params)), nil)
			assert.Error(t, err)
		})
	}
}

func TestPreRequest_InjectsDirective(t *testing.T) {
	t.Parallel()

	// maxTTL of 1m keeps the 0.95 quantile (~12.8s) unclamped, so the
	// duration assertion exercises the quantile computation itself.
	cfg := subagentConfig
	cfg.Queues = []QueueConfig{{
		SessionType: "agentic-subagent",
		Retain:      &RetainConfig{Quantile: 0.95, MinTTL: "1s", MaxTTL: "1m"},
	}}
	p := newTestPlugin(t, cfg)
	request := sessionRequest("s1", &subagentPrediction)

	require.NoError(t, p.PreRequest(t.Context(), request, nil))

	assert.True(t, request.Body.Mutated)
	payload, _ := request.Body.Payload.AsMap()
	assert.Equal(t, "s1", payload[retentionScopeField])

	d := directive(t, request)
	assert.Equal(t, 0, d["start"])
	assert.Nil(t, d["end"])
	assert.Equal(t, defaultPriority, d["priority"])

	// The duration is the 0.95 quantile of the published prediction.
	wantSeconds := math.Exp(0.69 + 1.13*math.Sqrt2*math.Erfinv(2*0.95-1))
	assert.InDelta(t, wantSeconds, d["duration"], 1e-6)
}

func TestPreRequest_ParentSessionScope(t *testing.T) {
	t.Parallel()

	p := newTestPlugin(t, subagentConfig)
	request := sessionRequest("sub-1", &subagentPrediction)
	request.Headers = map[string]string{"x-parent-session-id": "main-1"}

	require.NoError(t, p.PreRequest(t.Context(), request, nil))

	payload, _ := request.Body.Payload.AsMap()
	assert.Equal(t, "main-1", payload[retentionScopeField])
}

func TestPreRequest_QueueMatching(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		cfg         Config
		sessionType string
		wantHint    bool
	}{
		{name: "matching queue", cfg: subagentConfig, sessionType: "agentic-subagent", wantHint: true},
		{name: "no matching queue", cfg: subagentConfig, sessionType: "chat", wantHint: false},
		{name: "catch-all matches any type", cfg: Config{}, sessionType: "chat", wantHint: true},
		{name: "catch-all matches untyped", cfg: Config{}, sessionType: "", wantHint: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			p := newTestPlugin(t, tt.cfg)
			prediction := subagentPrediction
			prediction.SessionType = tt.sessionType
			request := sessionRequest("s1", &prediction)

			require.NoError(t, p.PreRequest(t.Context(), request, nil))
			assert.Equal(t, tt.wantHint, request.Body.Mutated)
		})
	}
}

func TestPreRequest_SkipsWithoutAttributes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request *fwksched.InferenceRequest
	}{
		{name: "no session id", request: sessionRequest("", &subagentPrediction)},
		{name: "no prediction", request: sessionRequest("s1", nil)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			p := newTestPlugin(t, subagentConfig)

			require.NoError(t, p.PreRequest(t.Context(), tt.request, nil))
			assert.False(t, tt.request.Body.Mutated)
		})
	}
}

func TestPreRequest_ClientDirectivesWin(t *testing.T) {
	t.Parallel()

	p := newTestPlugin(t, subagentConfig)
	request := sessionRequest("s1", &subagentPrediction)
	clientDirectives := []any{map[string]any{"start": 0, "priority": 99}}
	payload, _ := request.Body.Payload.AsMap()
	payload[retentionDirectivesField] = clientDirectives

	require.NoError(t, p.PreRequest(t.Context(), request, nil))

	assert.False(t, request.Body.Mutated)
	assert.Equal(t, clientDirectives, payload[retentionDirectivesField])
	assert.NotContains(t, payload, retentionScopeField)
}

func TestPreRequest_NonMapPayloadUntouched(t *testing.T) {
	t.Parallel()

	p := newTestPlugin(t, subagentConfig)
	request := sessionRequest("s1", &subagentPrediction)
	request.Body.Payload = fwkrh.RawPayload(`{}`)

	require.NoError(t, p.PreRequest(t.Context(), request, nil))
	assert.False(t, request.Body.Mutated)
}

func TestRetainTTL_Clamped(t *testing.T) {
	t.Parallel()

	// The subagent fit puts the 0.95 quantile near 12.8s; the configured
	// maxTTL of 10s caps it.
	pMax := newTestPlugin(t, subagentConfig)
	assert.Equal(t, 10*time.Second, pMax.cfg.queues["agentic-subagent"].ttl(subagentPrediction))

	cfg := subagentConfig
	cfg.Queues = []QueueConfig{{
		SessionType: "agentic-subagent",
		Retain:      &RetainConfig{Quantile: 0.95, MinTTL: "30s", MaxTTL: "1m"},
	}}
	pMin := newTestPlugin(t, cfg)
	assert.Equal(t, 30*time.Second, pMin.cfg.queues["agentic-subagent"].ttl(subagentPrediction))
}
