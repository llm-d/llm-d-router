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

// Package loraresidency narrows candidates to the endpoints where the
// requested LoRA adapter is already resident, and decides when a new copy of
// the adapter may be created elsewhere.
package loraresidency

import (
	"context"
	"encoding/json"
	"fmt"

	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	schedplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling"
)

const (
	PluginType = "lora-residency-filter"
)

var (
	_ fwksched.Filter          = &Plugin{}
	_ fwkplugin.ConsumerPlugin = &Plugin{}
)

// Config holds the filter parameters.
type Config struct {
	// MaxReplicas caps how many endpoints may hold a copy of one adapter. Once an
	// adapter has that many homes the filter keeps requests on them even when
	// they are saturated. 0 means unbounded.
	MaxReplicas int `json:"maxReplicas,omitempty"`
	// QueueThreshold is the waiting-queue depth above which an endpoint counts as
	// saturated. 0 disables the queue check.
	QueueThreshold int `json:"queueThreshold,omitempty"`
	// KVCacheThreshold is the KV cache utilization, in [0, 1], above which an
	// endpoint counts as saturated. 0 disables the KV cache check.
	KVCacheThreshold float64 `json:"kvCacheThreshold,omitempty"`
}

var DefaultConfig = Config{
	MaxReplicas:      0,
	QueueThreshold:   8,
	KVCacheThreshold: 0.8,
}

func (c *Config) validate() error {
	if c.MaxReplicas < 0 {
		return fmt.Errorf("maxReplicas must be >= 0, got %d", c.MaxReplicas)
	}
	if c.QueueThreshold < 0 {
		return fmt.Errorf("queueThreshold must be >= 0, got %d", c.QueueThreshold)
	}
	if c.KVCacheThreshold < 0 || c.KVCacheThreshold > 1 {
		return fmt.Errorf("kvCacheThreshold must be in [0, 1], got %f", c.KVCacheThreshold)
	}
	return nil
}

// Plugin is the lora-residency-filter.
type Plugin struct {
	typedName fwkplugin.TypedName
	config    Config
}

// Factory builds the filter from its JSON parameters.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
	}
	return New(name, config), nil
}

// New builds the filter directly from a Config.
func New(name string, config Config) *Plugin {
	return &Plugin{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		config:    config,
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Consumes declares the per-endpoint metrics the filter reads, all published
// by the core-metrics-extractor.
func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.LoadedModelsKey, metrics.MetricsExtractorType):        map[string]fwkdl.LoraLoadState{},
			fwkplugin.NewDataKey(metrics.WaitingQueueSizeKey, metrics.MetricsExtractorType):    int(0),
			fwkplugin.NewDataKey(metrics.KVCacheUsagePercentKey, metrics.MetricsExtractorType): float64(0),
		},
	}
}

// Filter keeps the endpoints where the adapter occupies a GPU slot. A host-cache
// copy is not a home: activating it evicts a GPU resident and, measured on
// Qwen3-32B, costs about as much as a load from disk. When every home is
// saturated it opens one new copy on the unsaturated endpoints, host-cache
// copies first, unless MaxReplicas has been reached or no endpoint has room,
// in which case the homes are kept. With no home at all, every endpoint passes
// so the scorer can choose the first one.
func (p *Plugin) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)
	_, span := tracing.Tracer(schedplugins.TracerScope).Start(ctx, "filter_lora_residency",
		trace.WithSpanKind(trace.SpanKindInternal))
	defer span.End()
	span.SetAttributes(semconv.LLMDEPPFilterCandidateEndpoints(len(endpoints)))

	decide := func(outcome string, kept []fwksched.Endpoint) []fwksched.Endpoint {
		recordDecision(p.typedName.Name, outcome)
		span.SetAttributes(semconv.LLMDEPPFilterDecision(outcome))
		logger.V(logutil.DEBUG).Info("LoraResidencyFilter", "outcome", outcome, "kept", len(kept), "total", len(endpoints))
		return kept
	}

	if request == nil || request.TargetModel == "" || len(endpoints) <= 1 {
		return decide(outcomeNotApplicable, endpoints)
	}
	span.SetAttributes(semconv.GenAIRequestModel(request.TargetModel))

	var homes, warm, cold []fwksched.Endpoint
	for _, ep := range endpoints {
		switch state, resident := ep.GetMetrics().LoadedModels[request.TargetModel]; {
		case resident && state.Level == fwkdl.LoraLoadLevelGPU:
			homes = append(homes, ep)
		case resident:
			warm = append(warm, ep)
		default:
			cold = append(cold, ep)
		}
	}
	span.SetAttributes(semconv.LLMDEPPFilterStickyEndpoints(len(homes)))

	if len(homes) == 0 {
		return decide(outcomeNoHome, endpoints)
	}
	for _, ep := range homes {
		if !p.saturated(ep) {
			return decide(outcomeSticky, homes)
		}
	}
	if p.config.MaxReplicas > 0 && len(homes) >= p.config.MaxReplicas {
		return decide(outcomeCapBlocked, homes)
	}
	for _, candidates := range [][]fwksched.Endpoint{warm, cold} {
		var spare []fwksched.Endpoint
		for _, ep := range candidates {
			if !p.saturated(ep) {
				spare = append(spare, ep)
			}
		}
		if len(spare) > 0 {
			return decide(outcomeSpread, spare)
		}
	}
	return decide(outcomeFleetSaturated, homes)
}

func (p *Plugin) saturated(ep fwksched.Endpoint) bool {
	m := ep.GetMetrics()
	if p.config.QueueThreshold > 0 && m.WaitingQueueSize > p.config.QueueThreshold {
		return true
	}
	if p.config.KVCacheThreshold > 0 && m.KVCacheUsagePercent > p.config.KVCacheThreshold {
		return true
	}
	return false
}
