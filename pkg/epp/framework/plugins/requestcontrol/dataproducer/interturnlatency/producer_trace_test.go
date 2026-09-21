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
	"bufio"
	"encoding/json"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrinterturn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/interturn"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
)

// traceSession is one line of a WekaTrace JSONL corpus, e.g.
// huggingface.co/datasets/semianalysisai/cc-traces-weka-061326. Request
// timestamps are seconds from the session start; api_time is the request's
// turnaround, so t+api_time is the response completion the next turn's idle
// gap is measured from. A request of type "subagent" is a container: its
// nested requests are API calls issued under the subagent's own session
// identifier, which shares the workload-type label but not the parent's
// prefix cache.
type traceSession struct {
	ID       string         `json:"id"`
	Requests []traceRequest `json:"requests"`
}

type traceRequest struct {
	T        float64        `json:"t"`
	APITime  float64        `json:"api_time"`
	Type     string         `json:"type"`
	AgentID  string         `json:"agent_id"`
	Requests []traceRequest `json:"requests"`
}

// traceEvent is one replayed hook invocation. Sessions are replayed as if
// co-started: their relative timestamps interleave on one global clock.
type traceEvent struct {
	at        float64
	sessionID string
	arrival   bool
}

// TestProducer_TraceCalibration replays a real agentic trace through the
// producer and checks that the predicted quantile of the next turn's idle gap
// is calibrated: the fraction of actual gaps at or below the predicted
// q-quantile must be close to q, and the online fit must converge to the
// offline maximum-likelihood estimate over the same gaps.
//
// The trace is not vendored; the test is skipped unless INTERTURN_TRACE_PATH
// points at a local traces.jsonl.
func TestProducer_TraceCalibration(t *testing.T) {
	t.Parallel()

	path := os.Getenv("INTERTURN_TRACE_PATH")
	if path == "" {
		t.Skip("set INTERTURN_TRACE_PATH to a WekaTrace traces.jsonl to run trace-replay checks")
	}

	const (
		quantile = 0.9
		// warmup is the observation count after which predictions are scored;
		// each queue's estimator needs to have left its seed behind.
		warmup = 1000
		// subagentType is the workload type the orchestrator assigns to
		// subagent traffic: same producer, separate queue.
		subagentType = "agentic-subagent"
	)
	// Queue seeds: the main queue uses the CC-Bench fit from the SAECache
	// paper (the producer default); the subagent queue is seeded with the
	// machine-paced fit of the WekaTrace subagent population.
	subagentSeed := attrinterturn.InterTurnPrediction{LogMean: 0.69, LogStd: 1.13}

	events := loadTraceEvents(t, path)
	p := newTestProducer(t, func(cfg *Config) {
		cfg.Queues = []QueueConfig{
			{SessionType: "agentic"},
			{SessionType: subagentType, InitialLogMean: &subagentSeed.LogMean, InitialLogStd: &subagentSeed.LogStd},
		}
	})
	base := time.Unix(0, 0)
	clock := base
	p.now = func() time.Time { return clock }

	isSubagent := func(sessionID string) bool { return strings.Contains(sessionID, "/") }
	requests := map[string]*fwksched.InferenceRequest{}
	requestFor := func(sessionID string) *fwksched.InferenceRequest {
		if r, ok := requests[sessionID]; ok {
			return r
		}
		sessionType := "agentic"
		if isSubagent(sessionID) {
			sessionType = subagentType
		}
		r := &fwksched.InferenceRequest{
			RequestID: sessionID,
			Headers:   map[string]string{"x-session-type": sessionType},
		}
		r.PutAttribute(attrsession.SessionIDDataKey, attrsession.SessionID(sessionID))
		requests[sessionID] = r
		return r
	}

	// lastActivity mirrors the tracker so each arrival's actual idle gap is
	// known to the test; predictions holds the prediction published at the
	// session's previous arrival, which is what should cover that gap.
	lastActivity := map[string]time.Time{}
	predictions := map[string]attrinterturn.InterTurnPrediction{}
	// The frozen per-queue seeds are the baseline: online learning must not
	// predict worse than the static fits.
	seeds := map[bool]attrinterturn.InterTurnPrediction{
		false: {LogMean: DefaultConfig.InitialLogMean, LogStd: DefaultConfig.InitialLogStd},
		true:  subagentSeed,
	}
	var scored, covered, seedCovered int
	// Per-population split: subagent sessions are machine-paced, main
	// sessions include human pauses; each population is scored against its
	// own queue's prediction.
	var scoredMain, coveredMain, scoredSub, coveredSub int
	logGaps := map[bool][]float64{}

	for _, event := range events {
		clock = base.Add(time.Duration(event.at * float64(time.Second)))
		request := requestFor(event.sessionID)

		if !event.arrival {
			p.ResponseBody(t.Context(), request, &requestcontrol.Response{EndOfStream: true}, nil)
			lastActivity[event.sessionID] = clock
			continue
		}

		if last, seen := lastActivity[event.sessionID]; seen {
			gap := clock.Sub(last)
			if gap >= p.cfg.minInterval && gap <= p.cfg.maxIdle {
				subagent := isSubagent(event.sessionID)
				logGaps[subagent] = append(logGaps[subagent], math.Log(gap.Seconds()))
				prediction, has := predictions[event.sessionID]
				if has && prediction.Observations >= warmup {
					scored++
					hit := gap.Seconds() <= prediction.Quantile(quantile)
					if hit {
						covered++
					}
					if gap.Seconds() <= seeds[subagent].Quantile(quantile) {
						seedCovered++
					}
					if subagent {
						scoredSub++
						if hit {
							coveredSub++
						}
					} else {
						scoredMain++
						if hit {
							coveredMain++
						}
					}
				}
			}
		}
		lastActivity[event.sessionID] = clock

		require.NoError(t, p.Produce(t.Context(), request, nil))
		prediction, ok := attrinterturn.ReadInterTurnPrediction(request)
		require.True(t, ok)
		predictions[event.sessionID] = prediction
	}

	require.Greater(t, scored, 1000, "trace too small to score calibration")
	coverage := float64(covered) / float64(scored)
	seedCoverage := float64(seedCovered) / float64(scored)
	mainCoverage := float64(coveredMain) / float64(max(scoredMain, 1))
	subCoverage := float64(coveredSub) / float64(max(scoredSub, 1))

	t.Logf("scored=%d coverage=%.3f seedCoverage=%.3f (target %.2f)", scored, coverage, seedCoverage, quantile)
	t.Logf("main sessions: scored=%d coverage=%.3f; subagent sessions: scored=%d coverage=%.3f",
		scoredMain, mainCoverage, scoredSub, subCoverage)
	for _, queue := range []struct {
		name     string
		subagent bool
	}{{"agentic", false}, {subagentType, true}} {
		offlineMean, offlineStd := logMoments(logGaps[queue.subagent])
		logMean, logStd, observed := p.estimators[queue.name].snapshot()
		t.Logf("queue %s: online mu=%.3f sigma=%.3f n=%d; offline MLE mu=%.3f sigma=%.3f n=%d",
			queue.name, logMean, logStd, observed, offlineMean, offlineStd, len(logGaps[queue.subagent]))
	}

	// Calibration is the predictability criterion: each population's
	// q-quantile prediction must cover close to a q fraction of its actual
	// gaps. The online fits are recency-weighted by construction (the EMA
	// tracks the current session mix, not the all-time average), so they are
	// scored on prediction quality, not on convergence to whole-trace MLEs.
	assert.InDelta(t, quantile, coverage, 0.05, "predicted quantile is not calibrated")
	assert.InDelta(t, quantile, mainCoverage, 0.07, "main-session quantile is not calibrated")
	assert.InDelta(t, quantile, subCoverage, 0.07, "subagent-session quantile is not calibrated")
	assert.GreaterOrEqual(t, math.Abs(quantile-seedCoverage)-math.Abs(quantile-coverage), -0.02,
		"online learning is worse calibrated than the frozen seeds")
}

// loadTraceEvents parses the JSONL trace into a single time-ordered event
// stream of arrivals and response completions. Completions sort before
// arrivals at equal timestamps, matching the serving order that produced
// them (a tool result re-arrives the instant the response finishes).
func loadTraceEvents(t *testing.T, path string) []traceEvent {
	t.Helper()

	file, err := os.Open(path)
	require.NoError(t, err)
	defer func() { require.NoError(t, file.Close()) }()

	var events []traceEvent
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 1<<20), 1<<28)
	for scanner.Scan() {
		var session traceSession
		require.NoError(t, json.Unmarshal(scanner.Bytes(), &session))
		require.NotEmpty(t, session.ID)
		for _, request := range session.Requests {
			if request.Type == "subagent" {
				// The container row is not an API call; its nested requests
				// run under the subagent's own session identifier.
				require.NotEmpty(t, request.AgentID)
				subagentID := session.ID + "/" + request.AgentID
				for _, nested := range request.Requests {
					events = append(events,
						traceEvent{at: nested.T, sessionID: subagentID, arrival: true},
						traceEvent{at: nested.T + nested.APITime, sessionID: subagentID})
				}
				continue
			}
			events = append(events,
				traceEvent{at: request.T, sessionID: session.ID, arrival: true},
				traceEvent{at: request.T + request.APITime, sessionID: session.ID})
		}
	}
	require.NoError(t, scanner.Err())
	require.NotEmpty(t, events)

	sort.SliceStable(events, func(i, j int) bool {
		if events[i].at != events[j].at {
			return events[i].at < events[j].at
		}
		return !events[i].arrival && events[j].arrival
	})
	return events
}
