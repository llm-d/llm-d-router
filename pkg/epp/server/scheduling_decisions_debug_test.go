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

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/scheduling/decisionhistory"
)

func TestSchedulingDecisionsDebugHandlerReturnsDecisions(t *testing.T) {
	withFrozenNow(t, time.Date(2025, 1, 2, 3, 4, 5, 0, time.UTC))

	recorder := decisionhistory.NewRecorder(2, 5)
	c := recorder.Begin("req-1", "model-a")
	c.ProfileStarted("default", 3)
	c.FilterRan("test-filter", "my-filter", 3, 1)
	c.Picked([]string{"pod1"}, []float64{1}, []string{"pod1"})
	recorder.Commit(c, time.Date(2025, 1, 2, 3, 4, 6, 0, time.UTC), "pod1", "")

	handler := NewSchedulingDecisionsDebugHandler(recorder)
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, SchedulingDecisionsDebugPath, nil))

	require.Equal(t, http.StatusOK, rr.Code)
	require.Equal(t, "application/json", rr.Header().Get("Content-Type"))

	var body struct {
		Timestamp string                     `json:"timestamp"`
		Decisions []decisionhistory.Decision `json:"decisions"`
	}
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
	require.Equal(t, "2025-01-02T03:04:05Z", body.Timestamp)
	require.Len(t, body.Decisions, 1)

	d := body.Decisions[0]
	require.Equal(t, "req-1", d.RequestID)
	require.Equal(t, "pod1", d.TargetEndpoint)
	require.Empty(t, d.FailureReason)
	require.Len(t, d.Profiles, 1)
	require.Equal(t, "default", d.Profiles[0].Profile)
	require.Equal(t, []decisionhistory.FilterStage{{Type: "test-filter", Name: "my-filter", Before: 3, After: 1}}, d.Profiles[0].Filters)
}

func TestSchedulingDecisionsDebugHandlerEmptyHistory(t *testing.T) {
	withFrozenNow(t, time.Date(2025, 1, 2, 3, 4, 5, 0, time.UTC))

	handler := NewSchedulingDecisionsDebugHandler(decisionhistory.NewRecorder(2, 5))
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, SchedulingDecisionsDebugPath, nil))

	require.Equal(t, http.StatusOK, rr.Code)
	require.JSONEq(t, `{"timestamp":"2025-01-02T03:04:05Z","decisions":[]}`, rr.Body.String())
}

func TestSchedulingDecisionsDebugHandlerMethodNotAllowed(t *testing.T) {
	handler := NewSchedulingDecisionsDebugHandler(decisionhistory.NewRecorder(2, 5))
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, SchedulingDecisionsDebugPath, nil))

	require.Equal(t, http.StatusMethodNotAllowed, rr.Code)
}

func TestSetupSchedulingDecisionsDebugHandlerRequiresRegistrarAndRecorder(t *testing.T) {
	recorder := decisionhistory.NewRecorder(2, 5)
	require.ErrorContains(t, SetupSchedulingDecisionsDebugHandler(nil, recorder), "registrar")
	require.ErrorContains(t, SetupSchedulingDecisionsDebugHandler(&testMetricsRegistrar{}, nil), "recorder")
	require.NoError(t, SetupSchedulingDecisionsDebugHandler(&testMetricsRegistrar{}, recorder))
}

type testMetricsRegistrar struct {
	addedPath string
}

func (r *testMetricsRegistrar) AddMetricsServerExtraHandler(path string, handler http.Handler) error {
	r.addedPath = path
	return nil
}
