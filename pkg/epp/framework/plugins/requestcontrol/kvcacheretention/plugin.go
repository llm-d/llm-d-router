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

// Package kvcacheretention provides a PreRequest plugin that annotates
// requests from multi-turn sessions with KV-cache retention directives for
// backends implementing the vLLM Context-Aware KV-Cache Retention API
// (https://github.com/vllm-project/vllm/issues/37003).
//
// The plugin consumes the SessionID attribute published by the
// session-id-producer and the InterTurnPrediction attribute published by the
// session-interturn-latency-producer, and sets each request's retention
// duration to a configured quantile of the predicted inter-turn interval:
// the session's KV blocks stay protected while the next turn is likely to
// arrive and fall back to LRU once the session is far into the distribution's
// tail.
package kvcacheretention

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
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
	PluginType = "kv-cache-retention"

	// retentionDirectivesField and retentionScopeField are the request-body
	// fields of the vLLM KV-Cache Retention API.
	retentionDirectivesField = "retention_directives"
	retentionScopeField      = "retention_scope"
)

var (
	_ requestcontrol.PreRequest = &Plugin{}
	_ fwkplugin.ConsumerPlugin  = &Plugin{}
)

// Config holds the plugin parameters. Durations are time.ParseDuration
// strings, e.g. "30s". See README.md for what each one does.
type Config struct {
	// Priority is the eviction priority (0-100) written into the directive.
	Priority int `json:"priority,omitempty"`
	// Quantile of the predicted inter-turn distribution used as the retention
	// duration, in (0, 1).
	Quantile float64 `json:"quantile,omitempty"`
	// MinRetention and MaxRetention clamp the computed duration.
	MinRetention string `json:"minRetention,omitempty"`
	MaxRetention string `json:"maxRetention,omitempty"`
}

// DefaultConfig is decoded over by the factory.
var DefaultConfig = Config{
	Priority:     70,
	Quantile:     0.9,
	MinRetention: "1s",
	MaxRetention: "10m",
}

// resolvedConfig is Config after parsing and validation.
type resolvedConfig struct {
	priority     int
	quantile     float64
	minRetention time.Duration
	maxRetention time.Duration
}

func (c Config) resolve() (resolvedConfig, error) {
	var out resolvedConfig
	var err error

	if out.minRetention, err = positiveDuration("minRetention", c.MinRetention); err != nil {
		return out, err
	}
	if out.maxRetention, err = positiveDuration("maxRetention", c.MaxRetention); err != nil {
		return out, err
	}

	switch {
	case c.Priority < 0 || c.Priority > 100:
		return out, fmt.Errorf("priority must be in [0, 100], got %d", c.Priority)
	case c.Quantile <= 0 || c.Quantile >= 1:
		return out, fmt.Errorf("quantile must be in (0, 1), got %v", c.Quantile)
	case out.minRetention > out.maxRetention:
		return out, fmt.Errorf("minRetention (%v) must be <= maxRetention (%v)", out.minRetention, out.maxRetention)
	}

	out.priority = c.Priority
	out.quantile = c.Quantile
	return out, nil
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

// Plugin annotates session requests with retention directives.
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
// plugin turns into retention directives.
func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			attrsession.SessionIDDataKey:             attrsession.SessionID(""),
			attrinterturn.InterTurnPredictionDataKey: attrinterturn.InterTurnPrediction{},
		},
	}
}

// PreRequest writes the retention directive into the request body. Requests
// without a session identifier or an inter-turn prediction pass through
// untouched.
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

	duration := p.retentionDuration(prediction)
	request.Body.MutatePayloadMap(func(m fwkrh.PayloadMap) {
		m[retentionDirectivesField] = []any{map[string]any{
			"start":    0,
			"end":      nil,
			"priority": p.cfg.priority,
			"duration": duration.Seconds(),
		}}
		m[retentionScopeField] = string(sessionID)
	})

	if debugLogger := log.FromContext(ctx).V(logutil.DEBUG); debugLogger.Enabled() {
		debugLogger.Info("kv-cache-retention directive set", "sessionID", string(sessionID),
			"durationSeconds", duration.Seconds(), "priority", p.cfg.priority,
			"logMean", prediction.LogMean, "logStd", prediction.LogStd,
			"observations", prediction.Observations)
	}
	return nil
}

// retentionDuration returns the configured quantile of the predicted
// distribution, clamped to the configured bounds. Clamping happens on the
// float value so an extreme quantile never overflows the Duration conversion.
func (p *Plugin) retentionDuration(prediction attrinterturn.InterTurnPrediction) time.Duration {
	seconds := prediction.Quantile(p.cfg.quantile)
	if math.IsNaN(seconds) || seconds < p.cfg.minRetention.Seconds() {
		return p.cfg.minRetention
	}
	if seconds > p.cfg.maxRetention.Seconds() {
		return p.cfg.maxRetention
	}
	return time.Duration(seconds * float64(time.Second))
}
