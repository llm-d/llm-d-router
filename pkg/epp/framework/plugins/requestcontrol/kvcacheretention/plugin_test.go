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

package kvcacheretention

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

// seedPrediction is the CC-Bench seed fit of the
// session-interturn-latency-producer.
var seedPrediction = attrinterturn.InterTurnPrediction{LogMean: 2.28, LogStd: 1.34}

func newTestPlugin(t *testing.T, mutate func(*Config)) *Plugin {
	t.Helper()
	cfg := DefaultConfig
	if mutate != nil {
		mutate(&cfg)
	}
	p, err := NewPlugin("test", cfg)
	require.NoError(t, err)
	return p
}

// sessionRequest carries the attributes the session-id-producer and the
// session-interturn-latency-producer publish.
func sessionRequest(sessionID string) *fwksched.InferenceRequest {
	request := &fwksched.InferenceRequest{
		RequestID: "req-1",
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{"model": "m", "messages": []any{}},
		},
	}
	if sessionID != "" {
		request.PutAttribute(attrsession.SessionIDDataKey, attrsession.SessionID(sessionID))
	}
	request.PutAttribute(attrinterturn.InterTurnPredictionDataKey, seedPrediction)
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

	plg, err := Factory("kv-cache-retention", fwkplugin.StrictDecoder(json.RawMessage(`{"priority":50,"quantile":0.8}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, plg.TypedName().Type)

	_, err = Factory("kv-cache-retention", fwkplugin.StrictDecoder(json.RawMessage(`{"priority":200}`)), nil)
	assert.ErrorContains(t, err, "priority")

	_, err = Factory("kv-cache-retention", fwkplugin.StrictDecoder(json.RawMessage(`{"quantile":1.5}`)), nil)
	assert.ErrorContains(t, err, "quantile")

	_, err = Factory("kv-cache-retention", fwkplugin.StrictDecoder(json.RawMessage(`{"unknown":true}`)), nil)
	assert.ErrorContains(t, err, "failed to parse")
}

func TestPreRequest_InjectsDirective(t *testing.T) {
	t.Parallel()

	p := newTestPlugin(t, nil)
	request := sessionRequest("s1")

	require.NoError(t, p.PreRequest(t.Context(), request, nil))

	assert.True(t, request.Body.Mutated)
	payload, _ := request.Body.Payload.AsMap()
	assert.Equal(t, "s1", payload[retentionScopeField])

	d := directive(t, request)
	assert.Equal(t, 0, d["start"])
	assert.Nil(t, d["end"])
	assert.Equal(t, 70, d["priority"])

	// The duration is the 0.9 quantile of the published prediction.
	wantSeconds := math.Exp(2.28 + 1.34*math.Sqrt2*math.Erfinv(2*0.9-1))
	assert.InDelta(t, wantSeconds, d["duration"], 1e-6)
}

func TestPreRequest_SkipsWithoutAttributes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request *fwksched.InferenceRequest
	}{
		{name: "no session id", request: sessionRequest("")},
		{name: "no prediction", request: func() *fwksched.InferenceRequest {
			r := &fwksched.InferenceRequest{
				RequestID: "req-1",
				Body: &fwkrh.InferenceRequestBody{
					Payload: fwkrh.PayloadMap{"model": "m"},
				},
			}
			r.PutAttribute(attrsession.SessionIDDataKey, attrsession.SessionID("s1"))
			return r
		}()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			p := newTestPlugin(t, nil)

			require.NoError(t, p.PreRequest(t.Context(), tt.request, nil))
			assert.False(t, tt.request.Body.Mutated)
		})
	}
}

func TestPreRequest_ClientDirectivesWin(t *testing.T) {
	t.Parallel()

	p := newTestPlugin(t, nil)
	request := sessionRequest("s1")
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

	p := newTestPlugin(t, nil)
	request := sessionRequest("s1")
	request.Body.Payload = fwkrh.RawPayload(`{}`)

	require.NoError(t, p.PreRequest(t.Context(), request, nil))
	assert.False(t, request.Body.Mutated)
}

func TestRetentionDuration_Clamped(t *testing.T) {
	t.Parallel()

	// The seed prediction puts the 0.9 quantile near 54s; clamps override it.
	pMin := newTestPlugin(t, func(cfg *Config) { cfg.MinRetention = "2m"; cfg.MaxRetention = "5m" })
	assert.Equal(t, 2*time.Minute, pMin.retentionDuration(seedPrediction))

	pMax := newTestPlugin(t, func(cfg *Config) { cfg.MaxRetention = "10s" })
	assert.Equal(t, 10*time.Second, pMax.retentionDuration(seedPrediction))
}
