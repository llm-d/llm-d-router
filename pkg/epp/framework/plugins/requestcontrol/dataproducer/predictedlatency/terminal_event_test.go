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

package predictedlatency

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
)

func TestPredictedLatency_TerminalCleanupDoesNotAddToken(t *testing.T) {
	for _, cause := range []requestcontrol.TerminationCause{
		requestcontrol.TerminationCauseClientDisconnect,
		requestcontrol.TerminationCauseEvicted,
		requestcontrol.TerminationCauseError,
		requestcontrol.TerminationCauseNatural,
		"",
	} {
		name := string(cause)
		if name == "" {
			name = "unspecified"
		}
		t.Run(name, func(t *testing.T) {
			router := createTestRouter()
			router.config.StreamingMode = true
			predictor := new(mockPredictor)
			router.latencypredictor = predictor
			endpoint := createTestEndpoint("terminal-pod", 1, 1, 1)
			request := createTestInferenceRequest("unique-request-id", 100, 50)
			result := createTestSchedulingResult(endpoint.GetMetadata())
			state := newPredictedLatencyContext(request)
			router.setPredictedLatencyContextForRequest(request, state)
			require.NoError(t, router.PreRequest(context.Background(), request, result))

			// One real token has been observed. The synthetic cleanup callback
			// for a truncated stream carries no additional data event.
			state.requestReceivedTimestamp = time.Now().Add(-time.Second)
			processFirstTokenForLatencyPrediction(context.Background(), predictor, true, router.config.EndpointRoleLabel, state, state.requestReceivedTimestamp.Add(100*time.Millisecond))
			require.Equal(t, 1, state.generatedTokenCount)
			predictor.capturedTrainingEntries = nil
			events, wantTokens, wantTraining := 1, 1, 0
			if cause == requestcontrol.TerminationCauseNatural || cause == "" {
				// A natural final body chunk may contain another token.
				events, wantTokens, wantTraining = 2, 2, 1
			}
			router.ResponseBody(context.Background(), request, &requestcontrol.Response{
				EndOfStream: true, StreamedEvents: events, TerminationCause: cause,
			}, endpoint.GetMetadata())

			assert.Equal(t, wantTokens, state.generatedTokenCount, "cleanup counted a token that was not observed")
			assert.Len(t, predictor.capturedTrainingEntries, wantTraining, "a single observed token cannot supply a TPOT sample")
			_, err := router.getPredictedLatencyContextForRequest(request)
			require.Error(t, err, "terminal state must still be cleaned up")
		})
	}
}
