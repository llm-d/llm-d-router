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

package interturnlatency

import (
	"encoding/json"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrinterturn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/interturn"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
)

func newTestProducer(t *testing.T, mutate func(*Config)) *Producer {
	t.Helper()
	cfg := DefaultConfig
	if mutate != nil {
		mutate(&cfg)
	}
	p, err := NewProducer(InterTurnLatencyProducerType, cfg)
	require.NoError(t, err)
	return p
}

// agenticRequest carries the SessionID attribute the session-id-producer
// publishes plus the workload-type header.
func agenticRequest(sessionID string) *fwksched.InferenceRequest {
	request := &fwksched.InferenceRequest{
		RequestID: "req-1",
		Headers:   map[string]string{"x-session-type": "agentic"},
	}
	if sessionID != "" {
		request.PutAttribute(attrsession.SessionIDDataKey, attrsession.SessionID(sessionID))
	}
	return request
}

func TestFactory(t *testing.T) {
	t.Parallel()

	plg, err := Factory("interturn", fwkplugin.StrictDecoder(json.RawMessage(`{"emaFactor":0.5}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, InterTurnLatencyProducerType, plg.TypedName().Type)

	_, err = Factory("interturn", fwkplugin.StrictDecoder(json.RawMessage(`{"emaFactor":2}`)), nil)
	assert.ErrorContains(t, err, "emaFactor")

	_, err = Factory("interturn", fwkplugin.StrictDecoder(json.RawMessage(`{"minSamples":1}`)), nil)
	assert.ErrorContains(t, err, "minSamples")

	_, err = Factory("interturn", fwkplugin.StrictDecoder(json.RawMessage(`{"sessionTypeHeader":""}`)), nil)
	assert.ErrorContains(t, err, "sessionTypeHeader")

	_, err = Factory("interturn", fwkplugin.StrictDecoder(json.RawMessage(`{"unknown":true}`)), nil)
	assert.ErrorContains(t, err, "failed to parse")
}

func TestProduce_PublishesSeedPrediction(t *testing.T) {
	t.Parallel()

	p := newTestProducer(t, nil)
	request := agenticRequest("s1")

	require.NoError(t, p.Produce(t.Context(), request, nil))

	prediction, ok := attrinterturn.ReadInterTurnPrediction(request)
	require.True(t, ok, "prediction must be published")
	assert.Equal(t, 2.28, prediction.LogMean)
	assert.Equal(t, 1.34, prediction.LogStd)
	assert.EqualValues(t, 0, prediction.Observations)
}

func TestProduce_SkipsNonMatchingRequests(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request *fwksched.InferenceRequest
	}{
		{name: "no session id", request: agenticRequest("")},
		{name: "wrong session type", request: func() *fwksched.InferenceRequest {
			r := agenticRequest("s1")
			r.Headers["x-session-type"] = "chat"
			return r
		}()},
		{name: "missing session type", request: func() *fwksched.InferenceRequest {
			r := agenticRequest("s1")
			delete(r.Headers, "x-session-type")
			return r
		}()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			p := newTestProducer(t, nil)

			require.NoError(t, p.Produce(t.Context(), tt.request, nil))

			_, ok := attrinterturn.ReadInterTurnPrediction(tt.request)
			assert.False(t, ok, "no prediction must be published")
			assert.Equal(t, 0, p.tracker.size())
		})
	}
}

func TestProduce_SessionTypeEmptyMatchesAll(t *testing.T) {
	t.Parallel()

	p := newTestProducer(t, func(cfg *Config) { cfg.SessionType = "" })
	request := agenticRequest("s1")
	delete(request.Headers, "x-session-type")

	require.NoError(t, p.Produce(t.Context(), request, nil))

	_, ok := attrinterturn.ReadInterTurnPrediction(request)
	assert.True(t, ok)
}

func TestGapObservation_MeasuresIdleFromResponseCompletion(t *testing.T) {
	t.Parallel()

	// EMAFactor 1 and MinSamples 2 make the fitted logMean equal the sample
	// mean of the observed gaps, exposing the measured values directly.
	p := newTestProducer(t, func(cfg *Config) { cfg.EMAFactor = 1.0; cfg.MinSamples = 2 })
	base := time.Unix(1000, 0)
	clock := base
	p.now = func() time.Time { return clock }

	request := agenticRequest("s1")
	require.NoError(t, p.Produce(t.Context(), request, nil))

	// Each turn completes 20s after arrival; the next arrives 30s after
	// completion. The observed gap must be the 30s idle time, not the 50s
	// arrival-to-arrival time.
	clock = base.Add(20 * time.Second)
	p.ResponseBody(t.Context(), request, &requestcontrol.Response{EndOfStream: true}, nil)
	clock = base.Add(50 * time.Second)
	require.NoError(t, p.Produce(t.Context(), request, nil))

	clock = base.Add(70 * time.Second)
	p.ResponseBody(t.Context(), request, &requestcontrol.Response{EndOfStream: true}, nil)
	clock = base.Add(100 * time.Second)
	require.NoError(t, p.Produce(t.Context(), request, nil))

	prediction, ok := attrinterturn.ReadInterTurnPrediction(request)
	require.True(t, ok)
	assert.EqualValues(t, 2, prediction.Observations)
	assert.InDelta(t, math.Log(30), prediction.LogMean, 1e-9)
}

func TestGapObservation_DiscardsSubMinIntervals(t *testing.T) {
	t.Parallel()

	p := newTestProducer(t, nil)
	base := time.Unix(1000, 0)
	clock := base
	p.now = func() time.Time { return clock }

	request := agenticRequest("s1")
	require.NoError(t, p.Produce(t.Context(), request, nil))

	clock = base.Add(10 * time.Millisecond)
	require.NoError(t, p.Produce(t.Context(), request, nil))

	_, _, observed := p.estimator.snapshot()
	assert.EqualValues(t, 0, observed)
}

func TestResponseBody_IgnoresNonFinalChunks(t *testing.T) {
	t.Parallel()

	p := newTestProducer(t, nil)
	base := time.Unix(1000, 0)
	clock := base
	p.now = func() time.Time { return clock }

	request := agenticRequest("s1")
	require.NoError(t, p.Produce(t.Context(), request, nil))

	clock = base.Add(20 * time.Second)
	p.ResponseBody(t.Context(), request, &requestcontrol.Response{EndOfStream: false}, nil)

	// The gap origin stays at the request arrival because no final chunk was
	// seen.
	clock = base.Add(50 * time.Second)
	gap, ok := p.tracker.observe("s1", clock)
	require.True(t, ok)
	assert.Equal(t, 50*time.Second, gap)
}

func TestDumpState(t *testing.T) {
	t.Parallel()

	p := newTestProducer(t, nil)
	request := agenticRequest("s1")
	require.NoError(t, p.Produce(t.Context(), request, nil))

	raw, err := p.DumpState()
	require.NoError(t, err)

	var state debugState
	require.NoError(t, json.Unmarshal(raw, &state))
	assert.Equal(t, 2.28, state.LogMean)
	assert.Equal(t, 1.34, state.LogStd)
	assert.Equal(t, 1, state.TrackedSessions)
}
