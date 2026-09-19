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

// Package interturnlatency provides a DataProducer that learns the
// distribution of a multi-turn workload's inter-turn intervals online and
// publishes an InterTurnPrediction attribute on the InferenceRequest
// attribute store, so consumers can predict when the session's next turn is
// likely to arrive.
//
// For each session the producer measures the idle gap between one turn's
// response completion and the next turn's arrival. Gaps feed a log-normal
// fit: a sliding window over ln(gap) provides the maximum-likelihood sample
// estimate, blended into the running parameters with an exponential moving
// average. Session identity comes from the SessionID attribute published by
// the session-id-producer; the workload type comes from a request header
// supplied by the orchestrator.
package interturnlatency

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrinterturn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/interturn"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	interturnlatencyconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/interturnlatency/constants"
)

// InterTurnLatencyProducerType is the plugin type registered with the framework.
const InterTurnLatencyProducerType = interturnlatencyconstants.InterTurnLatencyProducerType

var (
	_ requestcontrol.DataProducer          = &Producer{}
	_ requestcontrol.ResponseBodyProcessor = &Producer{}
	_ fwkplugin.ConsumerPlugin             = &Producer{}
	_ fwkplugin.StateDumper                = &Producer{}
)

// Config holds the producer parameters. Durations are time.ParseDuration
// strings, e.g. "30s". See README.md for what each one does.
type Config struct {
	// SessionTypeHeader carries the orchestrator-assigned workload type.
	SessionTypeHeader string `json:"sessionTypeHeader,omitempty"`
	// SessionType is the workload type this producer acts on; requests with
	// any other value pass through untouched. Empty matches every request
	// that carries a session identifier.
	SessionType string `json:"sessionType,omitempty"`

	// InitialLogMean and InitialLogStd seed the estimator, in log-seconds.
	// The defaults are the CC-Bench agentic-trace fit reported in the
	// SAECache paper (arXiv:2605.18825, mu=2.28, sigma=1.34).
	InitialLogMean float64 `json:"initialLogMean,omitempty"`
	InitialLogStd  float64 `json:"initialLogStd,omitempty"`
	// EMAFactor is the blend weight of each new sample estimate, in (0, 1].
	EMAFactor float64 `json:"emaFactor,omitempty"`
	// MinSamples gates estimator updates until the window holds this many
	// observations.
	MinSamples int `json:"minSamples,omitempty"`
	// WindowSize is the sliding-window capacity of the sample estimate.
	WindowSize int `json:"windowSize,omitempty"`

	// MinInterval discards gap observations below the timestamp-precision
	// floor (tool-result auto-fills arriving effectively instantly).
	MinInterval string `json:"minInterval,omitempty"`
	// MaxIdle bounds a usable gap observation and the session sweep horizon.
	MaxIdle string `json:"maxIdle,omitempty"`
	// MaxSessions is the soft cap on tracked sessions.
	MaxSessions int `json:"maxSessions,omitempty"`
}

// DefaultConfig is decoded over by the factory.
var DefaultConfig = Config{
	SessionTypeHeader: "x-session-type",
	SessionType:       "agentic",
	InitialLogMean:    2.28,
	InitialLogStd:     1.34,
	EMAFactor:         0.1,
	MinSamples:        20,
	WindowSize:        200,
	MinInterval:       "100ms",
	MaxIdle:           "1h",
	MaxSessions:       100000,
}

// resolvedConfig is Config after parsing and validation.
type resolvedConfig struct {
	sessionTypeHeader string
	sessionType       string
	initialLogMean    float64
	initialLogStd     float64
	emaFactor         float64
	minSamples        int
	windowSize        int
	minInterval       time.Duration
	maxIdle           time.Duration
	maxSessions       int
}

func (c Config) resolve() (resolvedConfig, error) {
	var out resolvedConfig
	var err error

	if out.minInterval, err = positiveDuration("minInterval", c.MinInterval); err != nil {
		return out, err
	}
	if out.maxIdle, err = positiveDuration("maxIdle", c.MaxIdle); err != nil {
		return out, err
	}

	sessionType := strings.TrimSpace(c.SessionType)
	switch {
	case sessionType != "" && strings.TrimSpace(c.SessionTypeHeader) == "":
		return out, errors.New("sessionTypeHeader must not be empty when sessionType is set")
	case c.InitialLogStd <= 0:
		return out, fmt.Errorf("initialLogStd must be > 0, got %v", c.InitialLogStd)
	case c.EMAFactor <= 0 || c.EMAFactor > 1:
		return out, fmt.Errorf("emaFactor must be in (0, 1], got %v", c.EMAFactor)
	case c.MinSamples < 2:
		return out, fmt.Errorf("minSamples must be >= 2, got %d", c.MinSamples)
	case c.WindowSize < c.MinSamples:
		return out, fmt.Errorf("windowSize (%d) must be >= minSamples (%d)", c.WindowSize, c.MinSamples)
	case c.MaxSessions <= 0:
		return out, fmt.Errorf("maxSessions must be > 0, got %d", c.MaxSessions)
	}

	// Request headers are stored with lowercased keys (see
	// handlers.HandleRequestHeaders), so the configured name must match.
	out.sessionTypeHeader = strings.ToLower(strings.TrimSpace(c.SessionTypeHeader))
	out.sessionType = sessionType
	out.initialLogMean = c.InitialLogMean
	out.initialLogStd = c.InitialLogStd
	out.emaFactor = c.EMAFactor
	out.minSamples = c.MinSamples
	out.windowSize = c.WindowSize
	out.maxSessions = c.MaxSessions
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

// Producer publishes an inter-turn prediction for session requests. It is a
// process-lifetime singleton serving every request, so all shared state is
// guarded.
type Producer struct {
	typedName fwkplugin.TypedName
	dk        fwkplugin.DataKey
	cfg       resolvedConfig
	estimator *logNormalEstimator
	tracker   *sessionTracker
	now       func() time.Time
}

// Factory builds a Producer from raw plugin parameters.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for plugin %q: %w", name, err)
		}
	}
	producer, err := NewProducer(name, cfg)
	if err != nil {
		return nil, fmt.Errorf("invalid parameters for plugin %q: %w", name, err)
	}
	return producer, nil
}

// NewProducer initializes a Producer from a validated Config.
func NewProducer(name string, cfg Config) (*Producer, error) {
	resolved, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	return &Producer{
		typedName: fwkplugin.TypedName{Type: InterTurnLatencyProducerType, Name: name},
		dk:        attrinterturn.InterTurnPredictionDataKey.WithNonEmptyProducerName(name),
		cfg:       resolved,
		estimator: newLogNormalEstimator(resolved.initialLogMean, resolved.initialLogStd,
			resolved.emaFactor, resolved.minSamples, resolved.windowSize),
		tracker: newSessionTracker(resolved.maxSessions, resolved.maxIdle),
		now:     time.Now,
	}, nil
}

// TypedName returns the type and name of the plugin.
func (p *Producer) TypedName() fwkplugin.TypedName { return p.typedName }

// Produces declares the InterTurnPrediction attribute key written by this
// producer.
func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.dk: attrinterturn.InterTurnPrediction{}}
}

// Consumes declares the SessionID attribute this producer keys its
// per-session tracking on.
func (p *Producer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{attrsession.SessionIDDataKey: attrsession.SessionID("")},
	}
}

// Produce feeds the session's idle gap to the estimator and publishes the
// current prediction on the request's attribute store. Requests without a
// session identifier or with a non-matching workload type get no prediction;
// consumers must handle absence as "not a tracked session".
func (p *Producer) Produce(_ context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	sessionID := p.sessionID(request)
	if sessionID == "" {
		return nil
	}
	if gap, ok := p.tracker.observe(sessionID, p.now()); ok && gap >= p.cfg.minInterval {
		p.estimator.observe(gap.Seconds())
	}

	logMean, logStd, observed := p.estimator.snapshot()
	request.PutAttribute(p.dk, attrinterturn.InterTurnPrediction{
		LogMean:      logMean,
		LogStd:       logStd,
		Observations: observed,
	})
	return nil
}

// ResponseBody records the session's activity when the response completes, so
// the next turn's gap measures idle time rather than turnaround time.
func (p *Producer) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *requestcontrol.Response, _ *fwkdl.EndpointMetadata) {
	if response == nil || !response.EndOfStream {
		return
	}
	sessionID := p.sessionID(request)
	if sessionID == "" {
		return
	}
	p.tracker.touch(sessionID, p.now())
}

// sessionID returns the request's session identifier, or empty when the
// request carries none or its workload type does not match.
func (p *Producer) sessionID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	if p.cfg.sessionType != "" {
		if request.Headers == nil ||
			!strings.EqualFold(strings.TrimSpace(request.Headers[p.cfg.sessionTypeHeader]), p.cfg.sessionType) {
			return ""
		}
	}
	id, ok := attrsession.ReadSessionID(request)
	if !ok {
		return ""
	}
	return string(id)
}

type debugState struct {
	LogMean         float64 `json:"logMean"`
	LogStd          float64 `json:"logStd"`
	Observations    int64   `json:"observations"`
	TrackedSessions int     `json:"trackedSessions"`
}

// DumpState implements [fwkplugin.StateDumper] for /debug/plugins/state.
func (p *Producer) DumpState() (json.RawMessage, error) {
	logMean, logStd, observed := p.estimator.snapshot()
	return json.Marshal(debugState{
		LogMean:         logMean,
		LogStd:          logStd,
		Observations:    observed,
		TrackedSessions: p.tracker.size(),
	})
}
