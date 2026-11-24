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

// Package qoscontract provides a RequestHeaderProcessor plugin that enforces per-team QoS
// contracts before flow control: it maps a team to a service class (a priority band), and moves
// traffic outside the team's contract (size limits, concurrency and a daily budget) to the next
// lower class.
package qoscontract

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// PluginType is the registered plugin type.
	PluginType = "qos-contract"

	// ClassHeader reports the class that served the request, or the class a rejection applied to.
	ClassHeader = "x-llm-d-qos-class"
	// ConcurrencyRemainingHeader reports the team's unused concurrency.
	ConcurrencyRemainingHeader = "x-llm-d-qos-concurrency-remaining"
	// RetryAfterHeader carries the seconds a rejected client should wait.
	RetryAfterHeader = "retry-after"

	defaultTeamID = "default"

	decisionWithinContract      = "within_contract"
	decisionDowngraded          = "downgraded"
	decisionRejectedConcurrency = "rejected_concurrency"
	decisionRejectedSize        = "rejected_size"
	decisionRejectedDaily       = "rejected_daily"

	// The reason label names the limit a downgraded or rejected request exceeded. When it exceeds
	// several, the first of size, daily budget and concurrency is reported, the same order that
	// picks the rejection.
	reasonSize        = "size"
	reasonDaily       = "daily"
	reasonConcurrency = "concurrency"
)

var (
	_ requesthandling.RequestHeaderProcessor  = &Plugin{}
	_ requesthandling.ResponseHeaderProcessor = &Plugin{}
	_ requesthandling.ResponseBodyProcessor   = &Plugin{}
	_ requesthandling.RequestAbandoned        = &Plugin{}
	_ plugin.ProducerPlugin                   = &Plugin{}
)

// teamState is the mutable accounting for one team contract.
type teamState struct {
	cfg        TeamConfig
	label      string
	classIndex int

	mu        sync.Mutex
	inflight  int
	dailyDay  int
	dailyUsed float64
}

// hold is the accounting held for one request until it completes.
type hold struct {
	team        *teamState
	servedClass string
	reserved    float64
	// prompt is the estimated prompt tokens, charged for a stream cut before its usage chunk.
	prompt float64
	// committed is true when the request was served within the contract and so holds a
	// concurrency slot and a daily-budget charge.
	committed bool
	// day is the UTC day whose budget the reservation was charged to.
	day int
	// errorStatus is set when the model server answered with a non-2xx status.
	errorStatus atomic.Bool
}

// Plugin enforces per-team QoS contracts. See the package README.
type Plugin struct {
	typedName    plugin.TypedName
	reportOnly   bool
	teamHeader   string
	outputWeight float64
	defaultOut   int64
	classes      []ClassConfig
	teams        map[string]*teamState
	defaultTeam  *teamState
	now          func() time.Time

	holds sync.Map // key: RequestID, value: *hold
}

// PluginFactory builds the plugin from its parameters.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	params := defaultParameters()
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PluginType, err)
		}
	}
	p, err := newPlugin(name, params, time.Now)
	if err != nil {
		return nil, err
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
	}
	return p, nil
}

func newPlugin(name string, params Parameters, now func() time.Time) (*Plugin, error) {
	if params.DefaultTeam.ID == "" {
		params.DefaultTeam.ID = defaultTeamID
	}
	if err := params.validate(); err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", PluginType, name, err)
	}
	classIndex := make(map[string]int, len(params.Classes))
	for i, c := range params.Classes {
		classIndex[c.Name] = i
	}
	newState := func(cfg TeamConfig) *teamState {
		return &teamState{cfg: cfg, label: metricsutil.BoundFairnessID(cfg.ID), classIndex: classIndex[cfg.Class]}
	}
	teams := make(map[string]*teamState, len(params.Teams))
	for _, t := range params.Teams {
		teams[t.ID] = newState(t)
	}
	return &Plugin{
		typedName:    plugin.TypedName{Type: PluginType, Name: name},
		reportOnly:   params.Mode == ModeReportOnly,
		teamHeader:   params.TeamHeader,
		outputWeight: params.OutputTokenWeight,
		defaultOut:   params.DefaultOutputReservation,
		classes:      params.Classes,
		teams:        teams,
		defaultTeam:  newState(params.DefaultTeam),
		now:          now,
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *Plugin) TypedName() plugin.TypedName { return p.typedName }

func (p *Plugin) lookupTeam(request *scheduling.InferenceRequest) (string, *teamState) {
	id, _ := metadata.GetLowerCaseHeaderValue(request.Headers, p.teamHeader)
	if st, ok := p.teams[id]; ok {
		return id, st
	}
	return id, p.defaultTeam
}

// estimateCost returns the estimated prompt tokens and the weighted cost reserved for the request.
// The prompt is estimated from the body size because tokenization runs after flow control.
// The size includes JSON structure, so the estimate runs high for a short prompt with large
// request fields (tools, stop sequences) and low for multi-byte UTF-8 text.
func (p *Plugin) estimateCost(request *scheduling.InferenceRequest, cfg TeamConfig) (prompt int64, cost float64) {
	prompt = int64(request.RequestSizeBytes / fwkrequest.BytesPerToken)
	var output int64
	switch {
	case request.Body != nil && request.Body.MaxOutputTokens != nil:
		output = *request.Body.MaxOutputTokens
	case cfg.MaxOutputTokens > 0:
		output = cfg.MaxOutputTokens
	default:
		output = p.defaultOut
	}
	return prompt, float64(prompt) + p.outputWeight*float64(output)
}

// RequestHeader decides the request's class. A request within the team's contract (its size
// limits and its commits) is served at the committed class and charged; a request outside it
// moves one class down without a charge, and is rejected in the lowest class.
func (p *Plugin) RequestHeader(_ context.Context, request *scheduling.InferenceRequest) error {
	if request == nil {
		return nil
	}
	teamID, team := p.lookupTeam(request)
	committed := p.classes[team.classIndex]
	prompt, cost := p.estimateCost(request, team.cfg)
	overSize := (team.cfg.MaxPromptTokens > 0 && prompt > team.cfg.MaxPromptTokens) ||
		(team.cfg.MaxOutputTokens > 0 && request.Body != nil && request.Body.MaxOutputTokens != nil && *request.Body.MaxOutputTokens > team.cfg.MaxOutputTokens)

	now := p.now()
	team.mu.Lock()
	if day := dayNumber(now); day != team.dailyDay {
		team.dailyDay, team.dailyUsed = day, 0
	}
	budgetDay := team.dailyDay
	overDaily := team.cfg.DailyTokens > 0 && team.dailyUsed+cost > team.cfg.DailyTokens
	overConcurrency := team.cfg.MaxConcurrency > 0 && team.inflight >= team.cfg.MaxConcurrency
	served, decision, withinContract := committed, decisionWithinContract, !overSize && !overDaily && !overConcurrency
	reason := ""
	switch {
	case overSize:
		reason = reasonSize
	case overDaily:
		reason = reasonDaily
	case overConcurrency:
		reason = reasonConcurrency
	}
	switch {
	case withinContract:
		team.inflight++
		team.dailyUsed += cost
		inflightRequests.WithLabelValues(team.label).Set(float64(team.inflight))
	case team.classIndex+1 < len(p.classes):
		served, decision = p.classes[team.classIndex+1], decisionDowngraded
	case overSize:
		team.mu.Unlock()
		return p.reject(team, committed, decisionRejectedSize, reason, p.sizeError(team, committed.Name))
	case overDaily:
		team.mu.Unlock()
		return p.reject(team, committed, decisionRejectedDaily, reason, p.quotaError(team, committed.Name, untilNextDay(now), "daily token budget exhausted"))
	default:
		team.mu.Unlock()
		return p.reject(team, committed, decisionRejectedConcurrency, reason, p.quotaError(team, committed.Name, time.Second, "concurrency limit reached"))
	}
	team.mu.Unlock()

	contractDecisions.WithLabelValues(team.label, committed.Name, served.Name, decision, reason, p.mode()).Inc()
	rec := &hold{team: team, servedClass: served.Name, reserved: cost, prompt: float64(prompt), committed: withinContract, day: budgetDay}
	// The hold is settled exactly once: by RequestAbandoned when the request leaves the director
	// before dispatch, otherwise at end of stream, which the handler forces for every dispatched
	// request, completed or not.
	p.holds.Store(request.RequestID, rec)

	// Report-only mode only records the decision: the request keeps the priority its
	// InferenceObjective set and the fairness ID it arrived with.
	if !p.reportOnly {
		setClass(request, teamID, served)
	}
	return nil
}

// Produces declares the priority and tenant the plugin sets. The Director applies them to the
// request after the plugin runs.
func (p *Plugin) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{
		requesthandling.PriorityKey: 0,
		requesthandling.TenantKey:   "",
	}
}

// setClass serves the request at class and makes the team the fairness flow when the request
// has no fairness ID.
func setClass(request *scheduling.InferenceRequest, teamID string, class ClassConfig) {
	request.PutAttribute(requesthandling.PriorityKey, class.Priority)
	if request.FairnessID == "" && teamID != "" {
		request.PutAttribute(requesthandling.TenantKey, teamID)
	}
}

// ResponseHeader records whether the model server answered with an error status, for
// settlement, and in enforce-limits mode reports the served class and the team's remaining quota.
func (p *Plugin) ResponseHeader(_ context.Context, request *scheduling.InferenceRequest, response *requesthandling.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || response.Headers == nil {
		return
	}
	v, ok := p.holds.Load(request.RequestID)
	if !ok {
		return
	}
	rec := v.(*hold)
	rec.errorStatus.Store(isErrorStatus(response.Headers))
	if p.reportOnly {
		return
	}
	response.Headers[ClassHeader] = rec.servedClass
	for k, val := range p.quotaHeaders(rec.team) {
		response.Headers[k] = val
	}
}

// RequestAbandoned refunds a request that left the director before dispatch: a flow-control
// rejection, a client that left while queued, a scheduling failure, or a later plugin's error.
func (p *Plugin) RequestAbandoned(_ context.Context, request *scheduling.InferenceRequest) {
	if request == nil {
		return
	}
	p.finish(request.RequestID, nil)
}

// ResponseBody settles the reservation at end of stream; see settledCost.
func (p *Plugin) ResponseBody(_ context.Context, request *scheduling.InferenceRequest, response *requesthandling.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	p.finish(request.RequestID, response)
}

// settledCost returns the weighted tokens a dispatched request is charged at end of stream:
//   - an error status from the model server with no usage is charged nothing;
//   - a request with usage or streamed events is charged its prompt (the reported input tokens,
//     else the estimate) plus its output (the reported output tokens, else the streamed events),
//     however the stream ended. Usage can be partial: an Anthropic stream reports input tokens in
//     message_start and output tokens only in the final message_delta, so a stream cut between
//     them falls back to the streamed events for its output;
//   - a stream cut short (client disconnect, eviction, error) with neither is charged nothing;
//   - a natural end with neither, or a request whose response the EPP does not observe, keeps
//     the reservation, the only estimate available.
func (p *Plugin) settledCost(rec *hold, response *requesthandling.Response) float64 {
	usage := response.Usage
	hasUsage := usage.PromptTokens > 0 || usage.CompletionTokens > 0
	switch {
	case !hasUsage && rec.errorStatus.Load():
		return 0
	case hasUsage || response.StreamedEvents > 0:
		prompt := rec.prompt
		if usage.PromptTokens > 0 {
			prompt = float64(usage.PromptTokens)
		}
		output := usage.CompletionTokens
		if output <= 0 {
			output = response.StreamedEvents
		}
		return prompt + p.outputWeight*float64(output)
	case response.TerminationCause == requesthandling.TerminationCauseClientDisconnect,
		response.TerminationCause == requesthandling.TerminationCauseEvicted,
		response.TerminationCause == requesthandling.TerminationCauseError:
		return 0
	default:
		return rec.reserved
	}
}

// finish settles a request exactly once. For a committed request it releases the concurrency slot
// and, when the reservation was made on the current UTC day, settles the daily charge. A nil response, for a request that was never dispatched, refunds
// the reservation in full.
func (p *Plugin) finish(requestID string, response *requesthandling.Response) {
	v, ok := p.holds.LoadAndDelete(requestID)
	if !ok {
		return
	}
	rec := v.(*hold)
	actual := 0.0
	if response != nil {
		actual = p.settledCost(rec, response)
	}
	team := rec.team
	if rec.committed {
		team.mu.Lock()
		team.inflight--
		// A reservation from an earlier UTC day was charged to that day's budget, which has been
		// reset since: its refund or overrun must not move today's budget.
		if team.dailyDay == rec.day {
			team.dailyUsed = math.Max(0, team.dailyUsed-(rec.reserved-actual))
		}
		inflightRequests.WithLabelValues(team.label).Set(float64(team.inflight))
		team.mu.Unlock()
	}
	if actual > 0 {
		chargedTokens.WithLabelValues(team.label, rec.servedClass).Add(actual)
	}
}

func (p *Plugin) mode() string {
	if p.reportOnly {
		return ModeReportOnly
	}
	return ModeEnforceLimits
}

// reject records a rejection and returns err. In report-only mode it returns nil instead.
func (p *Plugin) reject(team *teamState, committed ClassConfig, decision, reason string, err errcommon.Error) error {
	contractDecisions.WithLabelValues(team.label, committed.Name, "", decision, reason, p.mode()).Inc()
	if !p.reportOnly {
		return err
	}
	// Report-only mode serves a would-be rejection unchanged. Like a rejection, it holds no
	// concurrency slot and no budget charge, so later decisions match enforce-limits mode.
	return nil
}

func (p *Plugin) sizeError(team *teamState, class string) errcommon.Error {
	return errcommon.Error{
		Code: errcommon.BadRequest,
		Msg:  fmt.Sprintf("request exceeds the size limits of team %q: max prompt tokens %d, max output tokens %d", team.cfg.ID, team.cfg.MaxPromptTokens, team.cfg.MaxOutputTokens),
		Headers: map[string]string{
			errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonSizeLimit),
			ClassHeader:                             class,
		},
	}
}

func (p *Plugin) quotaError(team *teamState, class string, retryAfter time.Duration, reason string) errcommon.Error {
	headers := p.quotaHeaders(team)
	headers[errcommon.RequestDroppedReasonHeaderKey] = string(errcommon.RequestDroppedReasonOverQuota)
	headers[ClassHeader] = class
	headers[RetryAfterHeader] = strconv.FormatInt(int64(math.Max(1, math.Ceil(retryAfter.Seconds()))), 10)
	return errcommon.Error{
		Code:    errcommon.ResourceExhausted,
		Msg:     fmt.Sprintf("team %q: %s", team.cfg.ID, reason),
		Headers: headers,
	}
}

// quotaHeaders returns the remaining-commit headers for team. The caller must not hold team.mu.
func (p *Plugin) quotaHeaders(team *teamState) map[string]string {
	headers := map[string]string{}
	team.mu.Lock()
	defer team.mu.Unlock()
	if team.cfg.MaxConcurrency > 0 {
		headers[ConcurrencyRemainingHeader] = strconv.Itoa(max(0, team.cfg.MaxConcurrency-team.inflight))
	}
	return headers
}

// isErrorStatus reports whether the response headers carry a non-2xx HTTP status.
func isErrorStatus(headers map[string]string) bool {
	status := headers[":status"]
	if status == "" {
		status = headers["status"]
	}
	code, err := strconv.Atoi(status)
	return err == nil && (code < 200 || code >= 300)
}

// dayNumber identifies the UTC day containing t.
func dayNumber(t time.Time) int {
	return int(t.UTC().Unix() / 86400)
}

// untilNextDay returns the time from t to the next UTC midnight. Truncate counts from the zero
// time, a UTC midnight, so the result does not depend on t's location.
func untilNextDay(t time.Time) time.Duration {
	return t.Truncate(24 * time.Hour).Add(24 * time.Hour).Sub(t)
}
