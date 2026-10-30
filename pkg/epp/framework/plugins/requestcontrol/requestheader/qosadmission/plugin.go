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

// Package qosadmission provides a RequestHeaderProcessor plugin that enforces per-team QoS
// contracts before flow control: it maps a team to a service class (a priority band), applies
// size caps, a concurrency cap, a committed token rate and a daily budget, and moves traffic
// above the commit to the next lower class.
package qosadmission

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"sync"
	"time"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// PluginType is the registered plugin type.
	PluginType = "qos-admission"

	// ClassHeader reports the class that served the request, or the class a rejection applied to.
	ClassHeader = "x-llm-d-qos-class"
	// RemainingTokensHeader reports the weighted tokens left in the team's bucket.
	RemainingTokensHeader = "x-ratelimit-remaining-tokens" //#nosec G101 -- header name, not a credential
	// ResetTokensHeader reports the seconds until the team's bucket is full.
	ResetTokensHeader = "x-ratelimit-reset-tokens" //#nosec G101 -- header name, not a credential
	// ConcurrencyRemainingHeader reports the team's unused concurrency.
	ConcurrencyRemainingHeader = "x-llm-d-qos-concurrency-remaining"
	// RetryAfterHeader carries the seconds a rejected client should wait.
	RetryAfterHeader = "retry-after"

	defaultTeamID = "default"

	decisionAdmitted            = "admitted"
	decisionDowngraded          = "downgraded"
	decisionRejectedQuota       = "rejected_quota"
	decisionRejectedConcurrency = "rejected_concurrency"
	decisionRejectedSize        = "rejected_size"
	decisionRejectedDaily       = "rejected_daily"
)

var (
	_ requestcontrol.RequestHeaderProcessor  = &Plugin{}
	_ requestcontrol.ResponseHeaderProcessor = &Plugin{}
	_ requestcontrol.ResponseBodyProcessor   = &Plugin{}
)

// teamState is the mutable accounting for one team contract.
type teamState struct {
	cfg        TeamConfig
	label      string
	classIndex int

	mu        sync.Mutex
	bucket    *tokenBucket // nil when the team has no commit
	inflight  int
	dailyDay  int
	dailyUsed float64
}

// admission is the accounting held for one admitted request until it completes.
type admission struct {
	team          *teamState
	servedClass   string
	reserved      float64
	bucketCharged bool
	stop          func() bool
}

// Plugin enforces per-team QoS contracts. See the package README.
type Plugin struct {
	typedName    plugin.TypedName
	shadow       bool
	teamHeader   string
	outputWeight float64
	defaultOut   int64
	classes      []ClassConfig
	teams        map[string]*teamState
	defaultTeam  *teamState
	now          func() time.Time

	admissions sync.Map // key: RequestID, value: *admission
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
	start := now()
	newState := func(cfg TeamConfig) *teamState {
		st := &teamState{cfg: cfg, label: metricsutil.BoundFairnessID(cfg.ID), classIndex: classIndex[cfg.Class]}
		if cfg.CommitTokensPerMinute > 0 {
			st.bucket = newTokenBucket(cfg.CommitTokensPerMinute, cfg.BurstSeconds, start)
		}
		return st
	}
	teams := make(map[string]*teamState, len(params.Teams))
	for _, t := range params.Teams {
		teams[t.ID] = newState(t)
	}
	return &Plugin{
		typedName:    plugin.TypedName{Type: PluginType, Name: name},
		shadow:       params.Mode == ModeShadow,
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

// estimateCost returns the estimated prompt tokens and the weighted cost reserved at admission.
// The prompt is estimated from the body size because tokenization runs after flow control.
func (p *Plugin) estimateCost(request *scheduling.InferenceRequest, cfg TeamConfig) (prompt int64, cost float64) {
	prompt = int64(request.RequestSizeBytes / fwkrh.BytesPerToken)
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

// RequestHeader decides the request's class and charges its reservation.
func (p *Plugin) RequestHeader(ctx context.Context, request *scheduling.InferenceRequest) error {
	if request == nil {
		return nil
	}
	teamID, team := p.lookupTeam(request)
	committed := p.classes[team.classIndex]
	prompt, cost := p.estimateCost(request, team.cfg)

	if (team.cfg.MaxPromptTokens > 0 && prompt > team.cfg.MaxPromptTokens) ||
		(team.cfg.MaxOutputTokens > 0 && request.Body != nil && request.Body.MaxOutputTokens != nil && *request.Body.MaxOutputTokens > team.cfg.MaxOutputTokens) {
		return p.reject(team, committed.Name, decisionRejectedSize, errcommon.Error{
			Code: errcommon.BadRequest,
			Msg:  fmt.Sprintf("request exceeds the size limits of team %q: max prompt tokens %d, max output tokens %d", team.cfg.ID, team.cfg.MaxPromptTokens, team.cfg.MaxOutputTokens),
			Headers: map[string]string{
				errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonSizeLimit),
				ClassHeader:                             committed.Name,
			},
		})
	}

	now := p.now()
	team.mu.Lock()
	if day := dayNumber(now); day != team.dailyDay {
		team.dailyDay, team.dailyUsed = day, 0
	}
	if team.cfg.DailyTokens > 0 && team.dailyUsed+cost > team.cfg.DailyTokens {
		team.mu.Unlock()
		return p.reject(team, committed.Name, decisionRejectedDaily, p.quotaError(team, committed.Name, untilNextDay(now), "daily token budget exhausted"))
	}
	if team.cfg.MaxConcurrency > 0 && team.inflight >= team.cfg.MaxConcurrency {
		team.mu.Unlock()
		return p.reject(team, committed.Name, decisionRejectedConcurrency, p.quotaError(team, committed.Name, time.Second, "concurrency limit reached"))
	}
	served, decision, charged := committed, decisionAdmitted, false
	if team.bucket != nil {
		switch {
		case team.bucket.tryTake(cost, now):
			charged = true
		case team.classIndex+1 < len(p.classes):
			served, decision = p.classes[team.classIndex+1], decisionDowngraded
		default:
			wait := team.bucket.waitFor(cost, now)
			team.mu.Unlock()
			return p.reject(team, committed.Name, decisionRejectedQuota, p.quotaError(team, committed.Name, wait, "token rate commit exceeded"))
		}
		bucketTokens.WithLabelValues(team.label).Set(team.bucket.remaining(now))
	}
	team.inflight++
	team.dailyUsed += cost
	inflightRequests.WithLabelValues(team.label).Set(float64(team.inflight))
	team.mu.Unlock()

	admissionDecisions.WithLabelValues(team.label, committed.Name, served.Name, decision, p.mode()).Inc()
	rec := &admission{team: team, servedClass: served.Name, reserved: cost, bucketCharged: charged}
	p.admissions.Store(request.RequestID, rec)
	// A request that never reaches end of stream (flow-control rejection, scheduling failure)
	// is released in full when its stream ends.
	requestID := request.RequestID
	rec.stop = context.AfterFunc(ctx, func() { p.finish(requestID, nil) })

	if p.shadow {
		return nil
	}
	request.Objectives.Priority = served.Priority
	if request.FairnessID == "" && teamID != "" {
		request.FairnessID = teamID
	}
	return nil
}

// ResponseHeader reports the served class and the team's remaining quota.
func (p *Plugin) ResponseHeader(_ context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, _ *datalayer.EndpointMetadata) {
	if p.shadow || request == nil || response == nil || response.Headers == nil {
		return
	}
	v, ok := p.admissions.Load(request.RequestID)
	if !ok {
		return
	}
	rec := v.(*admission)
	response.Headers[ClassHeader] = rec.servedClass
	for k, val := range p.quotaHeaders(rec.team) {
		response.Headers[k] = val
	}
}

// ResponseBody settles the reservation against the reported usage at end of stream.
func (p *Plugin) ResponseBody(_ context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	usage := response.Usage
	p.finish(request.RequestID, &usage)
}

// finish releases a request's concurrency slot and settles its charge exactly once. A nil usage
// refunds the reservation in full; a usage with no token counts keeps the reservation.
func (p *Plugin) finish(requestID string, usage *fwkrh.Usage) {
	v, ok := p.admissions.LoadAndDelete(requestID)
	if !ok {
		return
	}
	rec := v.(*admission)
	if rec.stop != nil {
		rec.stop()
	}
	actual := 0.0
	if usage != nil {
		actual = rec.reserved
		if usage.PromptTokens > 0 || usage.CompletionTokens > 0 {
			actual = float64(usage.PromptTokens) + p.outputWeight*float64(usage.CompletionTokens)
		}
	}
	refund := rec.reserved - actual
	now := p.now()
	team := rec.team
	team.mu.Lock()
	team.inflight--
	team.dailyUsed = math.Max(0, team.dailyUsed-refund)
	if rec.bucketCharged {
		team.bucket.adjust(refund, now)
		bucketTokens.WithLabelValues(team.label).Set(team.bucket.remaining(now))
	}
	inflightRequests.WithLabelValues(team.label).Set(float64(team.inflight))
	team.mu.Unlock()
	if actual > 0 {
		chargedTokens.WithLabelValues(team.label, rec.servedClass).Add(actual)
	}
}

func (p *Plugin) mode() string {
	if p.shadow {
		return ModeShadow
	}
	return ModeEnforce
}

// reject records a rejection and returns err, or nil in shadow mode.
func (p *Plugin) reject(team *teamState, class, decision string, err errcommon.Error) error {
	admissionDecisions.WithLabelValues(team.label, class, "", decision, p.mode()).Inc()
	if p.shadow {
		return nil
	}
	return err
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

// quotaHeaders returns the remaining-quota headers for team. The caller must not hold team.mu.
func (p *Plugin) quotaHeaders(team *teamState) map[string]string {
	headers := map[string]string{}
	now := p.now()
	team.mu.Lock()
	defer team.mu.Unlock()
	if team.bucket != nil {
		headers[RemainingTokensHeader] = strconv.FormatInt(int64(team.bucket.remaining(now)), 10)
		headers[ResetTokensHeader] = strconv.FormatInt(int64(math.Ceil(team.bucket.untilFull(now).Seconds())), 10)
	}
	if team.cfg.MaxConcurrency > 0 {
		headers[ConcurrencyRemainingHeader] = strconv.Itoa(max(0, team.cfg.MaxConcurrency-team.inflight))
	}
	return headers
}

// dayNumber identifies the UTC day containing t.
func dayNumber(t time.Time) int {
	return int(t.UTC().Unix() / 86400)
}

func untilNextDay(t time.Time) time.Duration {
	u := t.UTC()
	next := time.Date(u.Year(), u.Month(), u.Day()+1, 0, 0, 0, 0, time.UTC)
	return next.Sub(u)
}
