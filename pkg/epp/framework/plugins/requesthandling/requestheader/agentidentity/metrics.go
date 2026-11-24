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

package agentidentity

import (
	"errors"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// sessionHeaderNone is the session_header label of a request that matched
	// no session header.
	sessionHeaderNone = "none"
	// harnessUnknown is the harness label of a request with no user-agent.
	harnessUnknown = "unknown"
	// harnessOther is the harness label of an unparsable user-agent and of
	// values beyond the label cap.
	harnessOther = metricsutil.OverflowValue
	// maxHarnessLabelValues caps the distinct harness label values.
	maxHarnessLabelValues = 50
	// maxHarnessTokenLength bounds a parsed user-agent product token.
	maxHarnessTokenLength = 64
)

// newRequestsCounter builds llm_d_epp_agent_identity_requests_total. Neither
// label carries a session ID: session_header is a configured header name (or
// "none") and harness is a bounded user-agent product token.
func newRequestsCounter() *prometheus.CounterVec {
	return prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "agent_identity_requests_total",
		Help: metricsutil.HelpMsgWithStability(
			"Requests seen by the agent-identity plugin, by the client's user-agent product token (harness) and the header the session identity was resolved from (session_header, \"none\" when no session header matched).",
			compbasemetrics.ALPHA),
	}, []string{"harness", "session_header"})
}

// newActiveSessionsGauge builds llm_d_epp_agent_identity_active_sessions.
func newActiveSessionsGauge() *prometheus.GaugeVec {
	return prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "agent_identity_active_sessions",
		Help: metricsutil.HelpMsgWithStability(
			"Distinct session identities that sent a request in the last 5 minutes, by the harness of the session's first request. Refreshed every 15 seconds.",
			compbasemetrics.ALPHA),
	}, []string{"harness"})
}

// errMetricsAlreadyEnabled rejects a second metrics-enabled agent-identity
// instance. Each instance keeps its own active-session set and may resolve
// different identities, so sharing the collectors would double-count requests
// and make the gauge alternate between the instances' values.
var errMetricsAlreadyEnabled = errors.New("metrics are already enabled on another agent-identity instance; enable metrics on one instance only")

// registerMetrics registers both collectors on reg (when non-nil).
func registerMetrics(reg prometheus.Registerer) (*prometheus.CounterVec, *prometheus.GaugeVec, error) {
	requests := newRequestsCounter()
	active := newActiveSessionsGauge()
	if reg == nil {
		return requests, active, nil
	}
	if err := register(reg, requests); err != nil {
		return nil, nil, err
	}
	if err := register(reg, active); err != nil {
		reg.Unregister(requests)
		return nil, nil, err
	}
	return requests, active, nil
}

func register(reg prometheus.Registerer, c prometheus.Collector) error {
	err := reg.Register(c)
	var already prometheus.AlreadyRegisteredError
	if errors.As(err, &already) {
		return errMetricsAlreadyEnabled
	}
	return err
}

func newHarnessLabels(limit int) *metricsutil.BoundedLabel {
	labels := metricsutil.NewBoundedLabel(limit)
	labels.Pin(harnessUnknown)
	labels.Pin(harnessOther)
	return labels
}

// harness returns the bounded harness label for a request's user-agent.
func (p *Plugin) harness(headers map[string]string) string {
	value := strings.TrimSpace(headers["user-agent"])
	if value == "" {
		return harnessUnknown
	}
	return p.harnessLabels.Bound(parseHarnessToken(value))
}

// parseHarnessToken returns the product token of a user-agent value: the text
// before the first "/" or space, lowercased, at most maxHarnessTokenLength
// characters of [a-z0-9._-], otherwise harnessOther. For example
// "claude-cli/2.1.3 (external, cli)" -> "claude-cli" and
// "codex_cli_rs/0.131.0 (Mac OS)" -> "codex_cli_rs".
func parseHarnessToken(value string) string {
	if i := strings.IndexAny(value, "/ "); i >= 0 {
		value = value[:i]
	}
	value = strings.ToLower(value)
	if value == "" || len(value) > maxHarnessTokenLength {
		return harnessOther
	}
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
		default:
			return harnessOther
		}
	}
	return value
}
