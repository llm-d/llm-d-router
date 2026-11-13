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
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/scheduling/decisionhistory"
)

const SchedulingDecisionsDebugPath = "/debug/scheduling/recent"

// SetupSchedulingDecisionsDebugHandler registers the recent scheduling
// decisions endpoint on the metrics/admin server. It returns an error when
// the registrar or the recorder is nil.
func SetupSchedulingDecisionsDebugHandler(registrar MetricsHandlerRegistrar, recorder *decisionhistory.Recorder) error {
	if registrar == nil {
		return errors.New("metrics handler registrar is not configured")
	}
	if recorder == nil {
		return errors.New("scheduling decision recorder is not configured")
	}
	return registrar.AddMetricsServerExtraHandler(SchedulingDecisionsDebugPath, NewSchedulingDecisionsDebugHandler(recorder))
}

func NewSchedulingDecisionsDebugHandler(recorder *decisionhistory.Recorder) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		decisions := recorder.Snapshot()
		if decisions == nil {
			decisions = []decisionhistory.Decision{}
		}
		payload, err := json.Marshal(schedulingDecisionsDebugResponse{
			Timestamp: nowFunc().UTC().Format(time.RFC3339Nano),
			Decisions: decisions,
		})
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to encode scheduling decisions: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	})
}

type schedulingDecisionsDebugResponse struct {
	Timestamp string                     `json:"timestamp"`
	Decisions []decisionhistory.Decision `json:"decisions"`
}
