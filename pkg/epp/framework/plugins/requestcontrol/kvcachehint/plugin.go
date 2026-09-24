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

// Package kvcachehint provides a PreRequest plugin that annotates requests
// from multi-turn sessions with KV-cache retention hints for backends
// implementing the vLLM Context-Aware KV-Cache Retention API
// (https://github.com/vllm-project/vllm/pull/38514).
//
// The plugin consumes the SessionID attribute published by the
// session-id-producer and the InterTurnPrediction attribute published by the
// session-interturn-latency-producer. Retention policy is configured per
// workload type: each queue matches the prediction's SessionType and sets the
// retention duration to a quantile of that queue's predicted inter-turn
// interval, clamped to configured bounds. The session's KV blocks stay
// protected while the next turn is likely to arrive and fall back to LRU once
// the session is far into the distribution's tail.
package kvcachehint

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrinterturn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/interturn"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
)

const (
	// PluginType is the plugin type registered with the framework.
	PluginType = "kv-cache-hint"

	// retentionDirectivesField and retentionScopeField are the request-body
	// fields of the vLLM KV-Cache Retention API.
	retentionDirectivesField = "retention_directives"
	retentionScopeField      = "retention_scope"
)

var (
	_ requestcontrol.PreRequest = &Plugin{}
	_ fwkplugin.ConsumerPlugin  = &Plugin{}
)

// RetainConfig sets one queue's retention policy: how long the session's KV
// blocks are protected from eviction after a turn completes. Durations are
// time.ParseDuration strings, e.g. "30s". Unset fields inherit DefaultRetain.
type RetainConfig struct {
	// Quantile of the predicted inter-turn distribution used as the retention
	// duration, in (0, 1).
	Quantile float64 `json:"quantile,omitempty"`
	// MinTTL and MaxTTL clamp the computed duration.
	MinTTL string `json:"minTTL,omitempty"`
	MaxTTL string `json:"maxTTL,omitempty"`
	// Priority is the eviction priority (0-100) written into the directive.
	Priority *int `json:"priority,omitempty"`
}

// QueueConfig declares the retention policy for one workload type. Requests
// whose InterTurnPrediction was fitted by the producer queue of the same
// SessionType get this queue's hint; other requests pass through untouched.
type QueueConfig struct {
	// SessionType is the workload type this queue's policy applies to.
	SessionType string `json:"sessionType"`
	// Retain is the queue's retention policy. Unset inherits DefaultRetain.
	Retain *RetainConfig `json:"retain,omitempty"`
}

// Config holds the plugin parameters. See README.md for what each one does.
type Config struct {
	// ParentSessionHeader carries the identifier of the session that spawned
	// this one. When present on a request, the retention scope is the parent
	// session, so a subagent's blocks are grouped under the parent's scope.
	// Empty disables parent scoping.
	ParentSessionHeader string `json:"parentSessionHeader,omitempty"`
	// Queues are the per-workload-type retention policies. Empty configures a
	// single catch-all queue applied to every request with a prediction.
	Queues []QueueConfig `json:"queues,omitempty"`
}

// DefaultRetain fills the unset fields of every queue's RetainConfig.
var DefaultRetain = RetainConfig{
	Quantile: 0.9,
	MinTTL:   "1s",
	MaxTTL:   "10m",
}

// defaultPriority is the directive eviction priority when unset.
const defaultPriority = 70

// DefaultConfig is decoded over by the factory.
var DefaultConfig = Config{
	ParentSessionHeader: "x-parent-session-id",
	Queues:              []QueueConfig{{SessionType: "agentic"}},
}

// catchAllQueue keys the single queue used when no queues are configured.
const catchAllQueue = ""

// retainPolicy is RetainConfig after parsing and validation.
type retainPolicy struct {
	quantile float64
	minTTL   time.Duration
	maxTTL   time.Duration
	priority int
}

// resolvedConfig is Config after parsing and validation. queues maps the
// lowercased workload type to its policy; a single catchAllQueue entry
// matches every type.
type resolvedConfig struct {
	parentSessionHeader string
	queues              map[string]retainPolicy
}

func (c Config) resolve() (resolvedConfig, error) {
	out := resolvedConfig{
		// Request headers are stored with lowercased keys (see
		// handlers.HandleRequestHeaders), so the configured name must match.
		parentSessionHeader: strings.ToLower(strings.TrimSpace(c.ParentSessionHeader)),
		queues:              make(map[string]retainPolicy, len(c.Queues)),
	}

	if len(c.Queues) == 0 {
		policy, err := resolveRetain(catchAllQueue, nil)
		if err != nil {
			return out, err
		}
		out.queues[catchAllQueue] = policy
		return out, nil
	}
	for _, queue := range c.Queues {
		sessionType := strings.ToLower(strings.TrimSpace(queue.SessionType))
		if sessionType == "" {
			return out, errors.New("queues entries must set sessionType")
		}
		if _, dup := out.queues[sessionType]; dup {
			return out, fmt.Errorf("duplicate queue for sessionType %q", sessionType)
		}
		policy, err := resolveRetain(sessionType, queue.Retain)
		if err != nil {
			return out, err
		}
		out.queues[sessionType] = policy
	}
	return out, nil
}

func resolveRetain(sessionType string, retain *RetainConfig) (retainPolicy, error) {
	merged := DefaultRetain
	if retain != nil {
		if retain.Quantile != 0 {
			merged.Quantile = retain.Quantile
		}
		if retain.MinTTL != "" {
			merged.MinTTL = retain.MinTTL
		}
		if retain.MaxTTL != "" {
			merged.MaxTTL = retain.MaxTTL
		}
		merged.Priority = retain.Priority
	}

	var out retainPolicy
	var err error
	if out.minTTL, err = positiveDuration("retain.minTTL", merged.MinTTL); err != nil {
		return out, queueErr(sessionType, err)
	}
	if out.maxTTL, err = positiveDuration("retain.maxTTL", merged.MaxTTL); err != nil {
		return out, queueErr(sessionType, err)
	}

	out.priority = defaultPriority
	if merged.Priority != nil {
		out.priority = *merged.Priority
	}
	switch {
	case merged.Quantile <= 0 || merged.Quantile >= 1:
		return out, queueErr(sessionType, fmt.Errorf("retain.quantile must be in (0, 1), got %v", merged.Quantile))
	case out.minTTL > out.maxTTL:
		return out, queueErr(sessionType, fmt.Errorf("retain.minTTL (%v) must be <= retain.maxTTL (%v)", out.minTTL, out.maxTTL))
	case out.priority < 0 || out.priority > 100:
		return out, queueErr(sessionType, fmt.Errorf("retain.priority must be in [0, 100], got %d", out.priority))
	}
	out.quantile = merged.Quantile
	return out, nil
}

func queueErr(sessionType string, err error) error {
	if sessionType == catchAllQueue {
		return err
	}
	return fmt.Errorf("queue %q: %w", sessionType, err)
}

func positiveDuration(field, raw string) (time.Duration, error) {
	d, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid %s %q: %w", field, raw, err)
	}
	if d <= 0 {
		return 0, fmt.Errorf("%s must be > 0, got %v", field, d)
	}
	return d, nil
}

// Plugin annotates session requests with retention hints.
type Plugin struct {
	typedName fwkplugin.TypedName
	cfg       resolvedConfig
}

// Factory builds a Plugin from raw plugin parameters.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for plugin %q: %w", name, err)
		}
	}
	plugin, err := NewPlugin(name, cfg)
	if err != nil {
		return nil, fmt.Errorf("invalid parameters for plugin %q: %w", name, err)
	}
	return plugin, nil
}

// NewPlugin initializes a Plugin from a validated Config.
func NewPlugin(name string, cfg Config) (*Plugin, error) {
	resolved, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	return &Plugin{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		cfg:       resolved,
	}, nil
}

// TypedName returns the type and name of the plugin.
func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

// Consumes declares the SessionID and InterTurnPrediction attributes this
// plugin turns into retention hints.
func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			attrsession.SessionIDDataKey:             attrsession.SessionID(""),
			attrinterturn.InterTurnPredictionDataKey: attrinterturn.InterTurnPrediction{},
		},
	}
}

// PreRequest writes the retention directive into the request body. Requests
// without a session identifier, without an inter-turn prediction, or whose
// prediction matches no configured queue pass through untouched.
//
// Always returns nil: a returned error fails the request, and failing to
// annotate retention is never a reason to reject one.
func (p *Plugin) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	if request == nil {
		return nil
	}
	sessionID, ok := attrsession.ReadSessionID(request)
	if !ok || sessionID == "" {
		return nil
	}
	prediction, ok := attrinterturn.ReadInterTurnPrediction(request)
	if !ok {
		return nil
	}
	policy, ok := p.matchQueue(prediction.SessionType)
	if !ok {
		return nil
	}

	if request.Body == nil || request.Body.Payload == nil {
		return nil
	}
	payload, ok := request.Body.Payload.AsMap()
	if !ok {
		return nil
	}
	if _, exists := payload[retentionDirectivesField]; exists {
		// Client-supplied directives win.
		return nil
	}

	scope := p.retentionScope(request, sessionID)
	duration := policy.ttl(prediction)
	request.Body.MutatePayloadMap(func(m fwkrh.PayloadMap) {
		m[retentionDirectivesField] = []any{map[string]any{
			"start":    0,
			"end":      nil,
			"priority": policy.priority,
			"duration": duration.Seconds(),
		}}
		m[retentionScopeField] = scope
	})

	if debugLogger := log.FromContext(ctx).V(logutil.DEBUG); debugLogger.Enabled() {
		debugLogger.Info("kv-cache hint set", "sessionID", string(sessionID),
			"scope", scope, "sessionType", prediction.SessionType,
			"durationSeconds", duration.Seconds(), "priority", policy.priority,
			"logMean", prediction.LogMean, "logStd", prediction.LogStd,
			"observations", prediction.Observations)
	}
	return nil
}

// matchQueue returns the policy for the prediction's workload type. A single
// catch-all queue matches every type.
func (p *Plugin) matchQueue(sessionType string) (retainPolicy, bool) {
	if policy, ok := p.cfg.queues[sessionType]; ok {
		return policy, true
	}
	policy, ok := p.cfg.queues[catchAllQueue]
	return policy, ok
}

// retentionScope is the parent session when the request names one, so blocks
// of a spawned session expire with the session that spawned it; otherwise the
// session itself.
func (p *Plugin) retentionScope(request *fwksched.InferenceRequest, sessionID attrsession.SessionID) string {
	if p.cfg.parentSessionHeader != "" && request.Headers != nil {
		if parent := strings.TrimSpace(request.Headers[p.cfg.parentSessionHeader]); parent != "" {
			return parent
		}
	}
	return string(sessionID)
}

// ttl returns the policy quantile of the predicted distribution, clamped to
// the policy bounds. Clamping happens on the float value so an extreme
// quantile never overflows the Duration conversion.
func (r retainPolicy) ttl(prediction attrinterturn.InterTurnPrediction) time.Duration {
	seconds := prediction.Quantile(r.quantile)
	if math.IsNaN(seconds) || seconds < r.minTTL.Seconds() {
		return r.minTTL
	}
	if seconds > r.maxTTL.Seconds() {
		return r.maxTTL
	}
	return time.Duration(seconds * float64(time.Second))
}
