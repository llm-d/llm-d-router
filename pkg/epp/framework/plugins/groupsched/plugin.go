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

// Package groupsched is the group scheduler: ONE shared queue, a seat budget per engine, and a
// seat held by each thread of requests while it runs. A thread is a chain of requests in which each
// one extends the previous; it is the fairness unit, the queue key and the thing that holds a seat.
//
// Every cost here is KV BLOCKS of a request's prompt, and this policy never hashes one: thread
// identity renders the request once at arrival, hashes it once, derives the thread from those
// blocks, and PUBLISHES the same per-prompt blocks (ThreadBlocksKey). One render and one hash per
// request, shared by the queue key and by every admission decision; the two plugins' block sizes
// must therefore agree, which the startup gate checks.
//
// READ IT TOP-DOWN:
//
//	plugin.go         this page: the plugin object, its startup checks, and what it is wired to
//	hooks_request.go  the request path, in the order the router calls it
//	hooks_pods.go     the detector and the endpoint life cycle (where capacity is learned)
//	pick.go           THE PASS: tidy → syncQueue → deriveColdCharges → enforceBudgets → seatWaiters → dispatch
//	admission.go      who gets a seat, and on which engine
//	eviction.go       who loses a seat, and when
//	requests.go       request records: released → at a pod → done
//	state.go          the data, its life cycle, and the lock convention
//	blockindex.go     the cache accounting: distinct blocks, their refcounts, the exact cost
//	config.go         the six knobs, their defaults and their validation
//	metrics.go        what an operator sees: the counters at their events, the gauges at scrape
//
// THE ROUTER'S CALLS, and where each one goes:
//
//	Pick              which queued request runs next      hooks_request.go → pick.go
//	Filter            which pod it runs on (its seat)     hooks_request.go
//	PreRequest        it reached a pod                    hooks_request.go → requests.go
//	ResponseBody      first chunk; end of stream          hooks_request.go → requests.go
//	RequestAbandoned  it left before reaching a pod       hooks_request.go → requests.go
//	Saturation        never saturated; learns capacity    hooks_pods.go
//	Extract           endpoint added or deleted           hooks_pods.go → eviction.go
package groupsched

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
)

// PluginType is the canonical type name used to register the plugin.
const PluginType = "group-scheduler"

var (
	_ fwkrc.PreRequest            = &GroupScheduler{}
	_ fwkrc.ResponseBodyProcessor = &GroupScheduler{}
	_ fwkplugin.StateDumper       = &GroupScheduler{}
	_ fwkdl.Registrant            = &GroupScheduler{}
	_ fwkdl.EndpointExtractor     = &GroupScheduler{}
	_ fwkfc.FairnessPolicy        = &GroupScheduler{}
	_ fwkfc.SaturationDetector    = &GroupScheduler{}
	_ fwksched.Filter             = &GroupScheduler{}
)

// GroupScheduler is one named instance shared by every hookup, so the three answers cannot
// disagree:
//   - Pick (the flow-control fairness policy): which queued request dispatches next;
//   - Filter (the scheduling filter): which engine may serve it — the thread's seat;
//   - Saturation (the flow-control detector): is the POOL saturated. It answers 0 always, and it is
//     where capacity is learned from the live candidates: once per new engine it takes the
//     scheduler's mutex, and every later poll (~1 kHz) touches only its own sets.
type GroupScheduler struct {
	typedName fwkplugin.TypedName
	cfg       Config

	sched   *Scheduler
	metrics *groupMetrics
	handle  fwkplugin.Handle
	// logger is the plugin's logger: startup state at the default level (a run's configuration and
	// its wiring must be readable from its own log), per-request detail at V(DEBUG) — off unless the
	// operator asks for it.
	logger logr.Logger

	fcfsOnce sync.Once

	// now is the plugin's clock: time.Now in production, and a seam a test can drive so a
	// decision trace does not depend on wall time.
	now func() time.Time

	// ---- capacity ----
	// endpoints is the set of KNOWN endpoints (the capacity_missing gauge reports them). Capacity is
	// learned from the candidates Saturation is handed, at the moment it is handed them.
	capMu     sync.Mutex
	endpoints map[string]struct{} // the KNOWN endpoints (capacity may still be unknown)
	// learned and rejected make Saturation's hot path cheap and quiet: a learned engine needs no
	// second look, and a REJECTED one (its measured block size disagrees with the configured one)
	// is not worth a scheduler lock and a log line a thousand times a second. A pod's geometry is
	// fixed for its life and a rollout brings a new endpoint name, so a rejection is once per
	// engine, not forever.
	learned  map[string]struct{}
	rejected map[string]struct{}
}

// Factory builds a GroupScheduler.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultConfig()
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", PluginType, err)
		}
	}
	sched, err := NewScheduler(cfg)
	if err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' plugin: %w", PluginType, err)
	}

	logCtx := context.Background()
	if handle != nil {
		logCtx = handle.Context()
	}
	r := &GroupScheduler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		logger:    log.FromContext(logCtx).WithName(PluginType + "/" + name),
		cfg:       cfg,
		sched:     sched,
		now:       time.Now,
		// Every map the hot path reads is created HERE, once: Saturation may be called from several
		// shard goroutines at the same time, so nothing on that path may create one lazily.
		endpoints: map[string]struct{}{},
		learned:   map[string]struct{}{},
		rejected:  map[string]struct{}{},
		// the handle is held for the RegisterDependencies-time gate: plugin construction order is
		// NOT the config order (the runner may build the fairness policy first), so the sibling
		// checks cannot run at Factory time — RegisterDependencies runs after ALL plugins are
		// constructed and added.
		handle: handle,
	}
	r.metrics = newPluginMetrics(r)
	// The decoded config is echoed at the DEFAULT level, always: a run's configuration must be
	// verifiable from its own log.
	r.logger.V(logutil.DEFAULT).Info("group-scheduler: effective config " + configEcho(cfg))

	// The observation wiring is TWO assignments: the core calls the metrics directly (every method
	// is nil-safe, so a unit-mode run records the same accounting as a live one — the affinity and
	// engine-gone counters are CORRECTNESS signals, not decoration) and logs the two lines a run's
	// artifacts have to be able to point at.
	r.sched.metrics = r.metrics
	r.sched.logger = r.logger
	if handle != nil {
		// THE DETECTOR-IDENTITY GATE.
		if others := r.otherDetectors(); len(others) > 0 {
			r.logger.V(logutil.DEFAULT).Info(
				"group-scheduler: WARNING DETECTOR CONFLICT — another plugin answers the flow-control saturation gate; point flowControl.saturationDetector.pluginRef at this one",
				"others", others)
		} else {
			r.logger.V(logutil.DEFAULT).Info("group-scheduler: this plugin answers the flow-control saturation gate; it answers NOT saturated (the seat budget is the admission control)")
		}
		if reg := handle.Metrics(); reg != nil {
			if err := r.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", PluginType, err)
			}
		}
		// the per-id SEAT/EXIT event stream, at V(DEBUG): off by default, because one line per seat
		// and per exit is measurement detail, not startup state.
		r.sched.onEvent = func(ev Event) {
			if !r.logger.V(logutil.DEBUG).Enabled() {
				return
			}
			r.logger.V(logutil.DEBUG).Info("group-scheduler event",
				"t", ev.T.UTC().Format(time.RFC3339Nano), "kind", ev.Kind, "id", ev.ID,
				"detail", ev.Detail, "engine", ev.Engine)
		}
	}
	return r, nil
}

// ---- the startup checks: what this plugin needs from its siblings, and what it refuses to run
// without ----

// otherDetectors lists the OTHER plugins in this config that implement the framework's
// SaturationDetector (the detector-identity gate's input).
func (r *GroupScheduler) otherDetectors() []string {
	if r.handle == nil {
		return nil
	}
	var out []string
	for _, p := range r.handle.GetAllPlugins() {
		if p == nil || p.TypedName().Type == PluginType {
			continue
		}
		if _, ok := p.(fwkfc.SaturationDetector); ok {
			out = append(out, p.TypedName().String())
		}
	}
	sort.Strings(out)
	return out
}

// validateThreadProducer is the fail-closed thread-source gate: the thread-identity plugin
// must be in the same config, it is the ONLY source accepted, and its block length must be
// ours. Both plugins charge the SAME blocks — thread identity hashes them once at arrival and
// this policy reads them — so a block-size disagreement would charge every request the wrong
// cost, silently. It is checked here, at startup, not discovered under load.
func (r *GroupScheduler) validateThreadProducer() error {
	if r.handle == nil {
		return nil // unit/test mode
	}
	thread, agent := false, false
	theirBlockSize := 0
	for _, p := range r.handle.GetAllPlugins() {
		if p == nil {
			continue
		}
		switch p.TypedName().Type {
		case threadidentity.PluginType:
			thread = true
			if tp, ok := p.(*threadidentity.Plugin); ok {
				theirBlockSize = tp.BlockSize()
			}
		case agentidentity.PluginType:
			agent = true
		}
	}
	if !thread {
		if agent {
			return fmt.Errorf("the '%s' plugin requires the '%s' plugin: agent identity alone is refused, because an agent id is a multi-chain member (its parallel sub-agents are separate chains) and this policy keeps ONE chain per tracked id — add the '%s' plugin",
				PluginType, threadidentity.PluginType, threadidentity.PluginType)
		}
		return fmt.Errorf("the '%s' plugin requires the '%s' plugin: with no thread identity every request is anonymous and Pick passes every head through unseated, which looks healthy in every outcome metric",
			PluginType, threadidentity.PluginType)
	}
	if theirBlockSize != 0 && theirBlockSize != r.cfg.BlockSizeTokens {
		return fmt.Errorf("the '%s' plugin and the '%s' plugin must agree on the block size: thread identity hashes the request's blocks at %d tokens and this policy charges them at %d, so every cost would be wrong — set both to the engine's block size",
			PluginType, threadidentity.PluginType, theirBlockSize, r.cfg.BlockSizeTokens)
	}
	if agent {
		r.logger.V(logutil.DEBUG).Info("group-scheduler: agent identity is configured but unused — the thread is thread identity's derived thread on every request")
	}
	return nil
}

// TypedName is the plugin's name in the framework (the registration key).
func (r *GroupScheduler) TypedName() fwkplugin.TypedName { return r.typedName }

// DumpState reports the group snapshot.
func (r *GroupScheduler) DumpState() (json.RawMessage, error) {
	return json.Marshal(r.sched.Snapshot(r.now()))
}
