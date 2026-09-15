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

// Package thunderagent consolidates program-aware admission control for
// agentic workloads into one plugin. A program (an agent trajectory
// identified by the request FairnessID) is bound to a pod, its KV token
// footprint is tracked from real usage reports plus in-flight estimates, and
// one shared program table backs every hookup:
//
//   - Scorer: sticky placement on the bound pod, least token load otherwise.
//   - SaturationDetector: per-pod working set against real scraped capacity,
//     with a hysteresis latch so admission does not oscillate.
//   - FairnessPolicy: resumes in-trajectory programs before admitting new
//     ones, smallest footprint first, with a starvation guard.
//   - PreRequest / ResponseBody: token accounting and session lifecycle.
//
// vllm:kv_cache_usage_perc counts only the blocks held by requests the engine
// is currently running. A program between turns (waiting on a tool) still
// owns its context in the prefix cache, but those blocks sit on the free list
// and are reported as unused. On an agentic workload most programs are idle
// at any instant, so engine-reported utilization stays near zero while the
// cache is in fact full. The committed-token footprint tracked here counts
// idle programs, which is the quantity that determines whether the next
// program fits.
package thunderagent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// ThunderAgentPluginType is the plugin type registered with the framework.
	ThunderAgentPluginType = "thunder-agent"

	// initialBytesPerToken seeds the request-size-to-token estimator before
	// the first usage observation. Refined with momentum from actual usage.
	initialBytesPerToken = 4.0

	// Bounds keeping the estimator sane against outlier usage samples.
	minBytesPerToken = 1.0
	maxBytesPerToken = 64.0

	// estimatorMomentum is the weight of the previous ratio in each update.
	estimatorMomentum = 0.8
)

// inflightEstimateKey is the per-request attribute under which PreRequest
// stashes the token estimate it added to a program, so ResponseBody can
// remove exactly that amount when the request completes or aborts.
var inflightEstimateKey = fwkplugin.NewDataKey("inflight-estimate", ThunderAgentPluginType)

var (
	_ fwksched.Scorer             = &ThunderAgent{}
	_ fwkrc.PreRequest            = &ThunderAgent{}
	_ fwkrc.ResponseBodyProcessor = &ThunderAgent{}
	_ fwkfc.SaturationDetector    = &ThunderAgent{}
	_ fwkfc.FairnessPolicy        = &ThunderAgent{}
	_ fwkplugin.StateDumper       = &ThunderAgent{}
)

// ThunderAgent is the single plugin instance behind all hookups. One named
// instance is referenced from the scheduling profile (scorer), the flow
// control saturation detector, and the priority band fairness policy, so all
// three read and write the same program table.
type ThunderAgent struct {
	typedName fwkplugin.TypedName

	capacityTokens         float64
	requireRealCapacity    bool
	utilThreshold          float64
	bufferTokensPerProgram float64
	shedIdle               time.Duration
	headWaitStarvationMs   float64
	sessionFinalHeader     string
	parentSessionHeader    string
	profileName            string

	table   *programTable
	metrics *thunderMetrics
}

// Factory builds a ThunderAgent from raw plugin parameters, registers its
// metrics, and starts its eviction sweep for the plugin's lifetime.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultConfig()
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", ThunderAgentPluginType, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' plugin: %w", ThunderAgentPluginType, err)
	}

	a := newThunderAgent(name, cfg)
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			if err := a.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", ThunderAgentPluginType, err)
			}
		}
		go a.runEviction(handle.Context(), time.Duration(cfg.EvictionSweepSeconds*float64(time.Second)))
	}
	return a, nil
}

func newThunderAgent(name string, cfg Config) *ThunderAgent {
	return &ThunderAgent{
		typedName:              fwkplugin.TypedName{Type: ThunderAgentPluginType, Name: name},
		capacityTokens:         float64(cfg.CapacityTokens),
		requireRealCapacity:    cfg.RequireRealCapacity,
		utilThreshold:          cfg.UtilThreshold,
		bufferTokensPerProgram: float64(cfg.BufferTokensPerProgram),
		shedIdle:               time.Duration(cfg.ShedIdleSeconds * float64(time.Second)),
		headWaitStarvationMs:   cfg.HeadWaitStarvationMs,
		sessionFinalHeader:     strings.ToLower(strings.TrimSpace(cfg.SessionFinalHeader)),
		parentSessionHeader:    strings.ToLower(strings.TrimSpace(cfg.ParentSessionHeader)),
		profileName:            cfg.ProfileName,
		table:                  newProgramTable(cfg),
		metrics:                newThunderMetrics(),
	}
}

func (a *ThunderAgent) TypedName() fwkplugin.TypedName {
	return a.typedName
}

// NewState satisfies the FairnessPolicy contract. Pick reads the shared
// program table instead of per-band state, so no scoped state is needed.
func (a *ThunderAgent) NewState(_ context.Context) any { return nil }

// programID returns the program identifier for a request, or "" for requests
// carrying no explicit identity. Anonymous traffic is not attributed to any
// program; pair this plugin with a general load scorer to account for it.
func programID(request *fwksched.InferenceRequest) string {
	if request == nil || request.FairnessID == "" || request.FairnessID == metadata.DefaultFairnessID {
		return ""
	}
	return request.FairnessID
}
