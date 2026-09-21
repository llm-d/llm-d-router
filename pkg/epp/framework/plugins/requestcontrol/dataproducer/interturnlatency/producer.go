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
// average. Fits are kept per workload type in configured queues, so
// workloads with different rhythms do not pollute each other. Session
// identity comes from the SessionID attribute published by the
// session-id-producer; the workload type comes from a request header
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

// QueueConfig declares one per-workload-type estimator queue. Requests whose
// workload type matches SessionType feed and read this queue's estimator, so
// workloads with different rhythms (a human-paced main session, a
// machine-paced subagent session) do not pollute each other's fit.
type QueueConfig struct {
	// SessionType is the workload type routed to this queue.
	SessionType string `json:"sessionType"`
	// InitialLogMean and InitialLogStd seed this queue's estimator, in
	// log-seconds. Unset values inherit the top-level seeds.
	InitialLogMean *float64 `json:"initialLogMean,omitempty"`
	InitialLogStd  *float64 `json:"initialLogStd,omitempty"`
}

// Config holds the producer parameters. Durations are time.ParseDuration
// strings, e.g. "30s". See README.md for what each one does.
type Config struct {
	// SessionTypeHeader carries the orchestrator-assigned workload type.
	SessionTypeHeader string `json:"sessionTypeHeader,omitempty"`
	// Queues are the per-workload-type estimator queues; requests with any
	// other type pass through untouched. Empty configures a single catch-all
	// queue fed by every request that carries a session identifier.
	Queues []QueueConfig `json:"queues,omitempty"`

	// InitialLogMean and InitialLogStd seed the estimators of queues without
	// their own seeds, in log-seconds. The defaults are the CC-Bench
	// agentic-trace fit reported in the SAECache paper (arXiv:2605.18825,
	// mu=2.28, sigma=1.34).
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
	Queues:            []QueueConfig{{SessionType: "agentic"}},
	InitialLogMean:    2.28,
	InitialLogStd:     1.34,
	EMAFactor:         0.1,
	MinSamples:        20,
	WindowSize:        200,
	MinInterval:       "100ms",
	MaxIdle:           "1h",
	MaxSessions:       100000,
}

// catchAllQueue keys the single queue used when no queues are configured.
const catchAllQueue = ""

// queueSeed is one queue's resolved estimator seed.
type queueSeed struct {
	logMean float64
	logStd  float64
}

// resolvedConfig is Config after parsing and validation. queues maps the
// lowercased workload type to its seed; a single catchAllQueue entry matches
// every type.
type resolvedConfig struct {
	sessionTypeHeader string
	queues            map[string]queueSeed
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

	switch {
	case len(c.Queues) > 0 && strings.TrimSpace(c.SessionTypeHeader) == "":
		return out, errors.New("sessionTypeHeader must not be empty when queues are configured")
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

	out.queues = make(map[string]queueSeed, len(c.Queues))
	if len(c.Queues) == 0 {
		out.queues[catchAllQueue] = queueSeed{logMean: c.InitialLogMean, logStd: c.InitialLogStd}
	}
	for _, queue := range c.Queues {
		sessionType := strings.ToLower(strings.TrimSpace(queue.SessionType))
		if sessionType == "" {
			return out, errors.New("queues entries must set sessionType")
		}
		if _, dup := out.queues[sessionType]; dup {
			return out, fmt.Errorf("duplicate queue for sessionType %q", sessionType)
		}
		seed := queueSeed{logMean: c.InitialLogMean, logStd: c.InitialLogStd}
		if queue.InitialLogMean != nil {
			seed.logMean = *queue.InitialLogMean
		}
		if queue.InitialLogStd != nil {
			seed.logStd = *queue.InitialLogStd
		}
		if seed.logStd <= 0 {
			return out, fmt.Errorf("queue %q initialLogStd must be > 0, got %v", sessionType, seed.logStd)
		}
		out.queues[sessionType] = seed
	}

	// Request headers are stored with lowercased keys (see
	// handlers.HandleRequestHeaders), so the configured name must match.
	out.sessionTypeHeader = strings.ToLower(strings.TrimSpace(c.SessionTypeHeader))
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
// guarded. estimators is keyed like resolvedConfig.queues and is built once
// at construction, so the map itself is read-only.
type Producer struct {
	typedName  fwkplugin.TypedName
	dk         fwkplugin.DataKey
	cfg        resolvedConfig
	estimators map[string]*logNormalEstimator
	tracker    *sessionTracker
	now        func() time.Time
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
	estimators := make(map[string]*logNormalEstimator, len(resolved.queues))
	for sessionType, seed := range resolved.queues {
		estimators[sessionType] = newLogNormalEstimator(seed.logMean, seed.logStd,
			resolved.emaFactor, resolved.minSamples, resolved.windowSize)
	}
	return &Producer{
		typedName:  fwkplugin.TypedName{Type: InterTurnLatencyProducerType, Name: name},
		dk:         attrinterturn.InterTurnPredictionDataKey.WithNonEmptyProducerName(name),
		cfg:        resolved,
		estimators: estimators,
		tracker:    newSessionTracker(resolved.maxSessions, resolved.maxIdle),
		now:        time.Now,
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

// Produce feeds the session's idle gap to its queue's estimator and
// publishes that queue's current prediction on the request's attribute
// store. Requests without a session identifier or with a workload type no
// queue matches get no prediction; consumers must handle absence as "not a
// tracked session".
func (p *Producer) Produce(_ context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	sessionID, queue, estimator := p.session(request)
	if sessionID == "" {
		return nil
	}
	if gap, ok := p.tracker.observe(sessionID, p.now()); ok && gap >= p.cfg.minInterval {
		estimator.observe(gap.Seconds())
	}

	logMean, logStd, observed := estimator.snapshot()
	request.PutAttribute(p.dk, attrinterturn.InterTurnPrediction{
		SessionType:  queue,
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
	sessionID, _, _ := p.session(request)
	if sessionID == "" {
		return
	}
	p.tracker.touch(sessionID, p.now())
}

// session returns the request's session identifier plus the key and
// estimator of the queue its workload type routes to. The identifier is
// empty when the request carries none or no queue matches its type.
func (p *Producer) session(request *fwksched.InferenceRequest) (string, string, *logNormalEstimator) {
	if request == nil {
		return "", "", nil
	}
	queue := catchAllQueue
	estimator, ok := p.estimators[catchAllQueue]
	if !ok {
		if request.Headers != nil {
			queue = strings.ToLower(strings.TrimSpace(request.Headers[p.cfg.sessionTypeHeader]))
		}
		if estimator, ok = p.estimators[queue]; !ok {
			return "", "", nil
		}
	}
	id, ok := attrsession.ReadSessionID(request)
	if !ok {
		return "", "", nil
	}
	return string(id), queue, estimator
}

type queueDebugState struct {
	LogMean      float64 `json:"logMean"`
	LogStd       float64 `json:"logStd"`
	Observations int64   `json:"observations"`
}

type debugState struct {
	// Queues is keyed by workload type; the catch-all queue reports as "*".
	Queues          map[string]queueDebugState `json:"queues"`
	TrackedSessions int                        `json:"trackedSessions"`
}

// DumpState implements [fwkplugin.StateDumper] for /debug/plugins/state.
func (p *Producer) DumpState() (json.RawMessage, error) {
	queues := make(map[string]queueDebugState, len(p.estimators))
	for sessionType, estimator := range p.estimators {
		logMean, logStd, observed := estimator.snapshot()
		if sessionType == catchAllQueue {
			sessionType = "*"
		}
		queues[sessionType] = queueDebugState{LogMean: logMean, LogStd: logStd, Observations: observed}
	}
	return json.Marshal(debugState{
		Queues:          queues,
		TrackedSessions: p.tracker.size(),
	})
}
