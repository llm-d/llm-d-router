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

package endpointhealth

import (
	"sync"
	"time"

	"github.com/go-logr/logr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// endpointState is the health state of one endpoint.
type endpointState struct {
	// address is the endpoint's host and port when the state started.
	address string
	// pollUnhealthy is the poll health decision of the latest evaluation.
	pollUnhealthy bool
	// responseFailures counts consecutive failed responses since the last
	// successful response or ejection.
	responseFailures int
	// ejections counts response ejections since the last successful
	// response. It scales the next ejection duration.
	ejections int
	// ejectedUntil is when the current response ejection ends. Zero when
	// the endpoint is not ejected.
	ejectedUntil time.Time
}

func (s *endpointState) ejected() bool {
	return !s.ejectedUntil.IsZero()
}

func (s *endpointState) excluded() bool {
	return s.pollUnhealthy || s.ejected()
}

// tracker holds the health state of every endpoint one filter instance has
// seen. The filter and response hooks run concurrently for many requests, so
// all state is guarded by mu.
type tracker struct {
	cfg        config
	pluginName string

	mu        sync.Mutex
	endpoints map[fwkdl.ID]*endpointState
	// excluded counts the endpoints whose state is excluded.
	excluded int
}

func newTracker(cfg config, pluginName string) *tracker {
	return &tracker{
		cfg:        cfg,
		pluginName: pluginName,
		endpoints:  make(map[fwkdl.ID]*endpointState),
	}
}

// pollUnhealthy reports whether a source's failure run is long enough to mark
// the endpoint unhealthy: at least pollFailureThreshold failed polls spanning at
// least pollFailureDuration. Requiring both keeps a single failed poll of a
// source with a long interval, and a burst of fast failures, from marking it.
func (t *tracker) pollUnhealthy(status *fwkdl.PollStatus, now time.Time) bool {
	if t.cfg.pollFailureThreshold == 0 || status == nil {
		return false
	}
	return status.ConsecutiveFailures >= t.cfg.pollFailureThreshold &&
		now.Sub(status.FailingSince) >= t.cfg.pollFailureDuration
}

// evaluate updates the endpoint's state from its latest poll status and the
// passage of time, and reports whether the endpoint is excluded.
func (t *tracker) evaluate(logger logr.Logger, id fwkdl.ID, address string, status *fwkdl.PollStatus, now time.Time) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	st := t.state(id, address)
	wasExcluded := st.excluded()

	if unhealthy := t.pollUnhealthy(status, now); unhealthy != st.pollUnhealthy {
		st.pollUnhealthy = unhealthy
		if unhealthy {
			ejections.WithLabelValues(t.pluginName, reasonPoll).Inc()
			logger.Info("Endpoint metric polls keep failing, marking it unhealthy",
				"endpoint", id, "source", t.cfg.pollSource,
				"consecutiveFailures", status.ConsecutiveFailures, "failingFor", now.Sub(status.FailingSince))
		} else {
			readmissions.WithLabelValues(t.pluginName, reasonPoll).Inc()
			logger.Info("Endpoint metric polls recovered, clearing its poll unhealthy mark",
				"endpoint", id, "source", t.cfg.pollSource, "excluded", st.ejected())
		}
	}

	if st.ejected() && !now.Before(st.ejectedUntil) {
		st.ejectedUntil = time.Time{}
		readmissions.WithLabelValues(t.pluginName, reasonResponse).Inc()
		logger.Info("Endpoint response ejection expired",
			"endpoint", id, "excluded", st.pollUnhealthy)
	}

	t.updateExcluded(wasExcluded, st.excluded())
	return st.excluded()
}

// observeResponse records the outcome of one response the endpoint returned.
func (t *tracker) observeResponse(logger logr.Logger, id fwkdl.ID, address string, failed bool, now time.Time) {
	if t.cfg.responseFailureThreshold == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	// A response from the address the endpoint had before it was recreated
	// says nothing about the endpoint now.
	if st, ok := t.endpoints[id]; ok && st.address != address {
		return
	}
	st := t.state(id, address)
	// Responses to requests scheduled before the ejection neither extend it
	// nor change the failure count and backoff of the next one.
	if st.ejected() {
		return
	}
	if !failed {
		st.responseFailures = 0
		st.ejections = 0
		return
	}
	st.responseFailures++
	if st.responseFailures < t.cfg.responseFailureThreshold {
		return
	}

	wasExcluded := st.excluded()
	st.ejections++
	duration := min(t.cfg.ejectionDuration*time.Duration(st.ejections), t.cfg.maxEjectionDuration)
	st.ejectedUntil = now.Add(duration)
	ejections.WithLabelValues(t.pluginName, reasonResponse).Inc()
	logger.Info("Endpoint returned consecutive failed responses, ejecting it",
		"endpoint", id, "consecutiveFailures", st.responseFailures, "ejectionDuration", duration)
	st.responseFailures = 0
	t.updateExcluded(wasExcluded, st.excluded())
}

// prune drops the state of endpoints that are no longer in the pool.
func (t *tracker) prune(active []fwkdl.ID) {
	keep := make(map[fwkdl.ID]struct{}, len(active))
	for _, id := range active {
		keep[id] = struct{}{}
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	for id, st := range t.endpoints {
		if _, ok := keep[id]; ok {
			continue
		}
		t.updateExcluded(st.excluded(), false)
		delete(t.endpoints, id)
	}
}

// state returns the endpoint's state. It starts a new state on first use, and
// when the endpoint is recreated at another address under the same ID, such as
// a restarted pod that keeps its name. Callers hold mu.
func (t *tracker) state(id fwkdl.ID, address string) *endpointState {
	st, ok := t.endpoints[id]
	if ok && st.address == address {
		return st
	}
	if ok {
		t.updateExcluded(st.excluded(), false)
	}
	st = &endpointState{address: address}
	t.endpoints[id] = st
	return st
}

// updateExcluded keeps the excluded count and gauge in step with one
// endpoint's transition. Callers hold mu.
func (t *tracker) updateExcluded(was, is bool) {
	if was == is {
		return
	}
	if is {
		t.excluded++
	} else {
		t.excluded--
	}
	excludedEndpoints.WithLabelValues(t.pluginName).Set(float64(t.excluded))
}
