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

package groupsched_test

// THE CONFIG-LEVEL STARTUP GATE — the wave-1 killer, rebuilt on exported APIs only.
//
// The original gate lived beside the runner because it called the runner's own
// configuration phases; this rewrite drives the SAME sequence through the exported
// pieces those phases are made of (loader.LoadRawConfig, loader.InstantiateAndConfigure,
// datalayer.CreateMissingDataProducers, ResolveDataProducerPhases,
// ValidateAndOrderDataDependencies), so the property it guards survives on a tree that
// carries nothing outside the two plugin folders: the GSCH-CORE chain — thread-identity
// (the member) → token-producer (the block keys) → the group scheduler behind THREE
// references (fairness policy, filter, saturation detector) — parses, instantiates and
// passes the plugin-DAG / layer-order validation the EPP runs at startup.
//
// The dual/triple reference is the shape that killed wave 1 live (the DAG validator
// rejected a consumer whose producer sat in a later layer; 75 green unit tests missed it
// because they never build the DAG).

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/epp/config/loader"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	extractormetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/bandselection"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/groupsched"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/passthrough"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/systemone"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmhttp"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
)

func init() {
	// The runner registers the flowControl gate and the in-tree plugin factories at
	// startup (runner.go:891-894); the exported loader needs both before a config
	// citing them parses. Registering here is the same act through exported APIs.
	loader.RegisterFeatureGate(flowcontrol.FeatureGate, false)
	fwkplugin.Register(threadidentity.PluginType, fwkplugin.StabilityAlpha, threadidentity.Factory)
	fwkplugin.Register(tokenizer.PluginType, fwkplugin.StabilityBeta, tokenizer.PluginFactory)
	fwkplugin.Register(groupsched.PluginType, fwkplugin.StabilityAlpha, groupsched.Factory)
	// The loader's scheduling defaults reference the single profile handler; the
	// runner registers every in-tree factory at startup, this test the ones the
	// core chain and the system defaults resolve against.
	// the system defaults the loader fabricates when a slot names nothing: the
	// profile handler, the picker, and the flow-control layer's four defaults.
	fwkplugin.Register(single.SingleProfileHandlerType, fwkplugin.StabilityStable, single.SingleProfileHandlerFactory)
	fwkplugin.Register(maxscore.MaxScorePickerType, fwkplugin.StabilityStable, maxscore.MaxScorePickerFactory)
	fwkplugin.Register(fcfs.FCFSOrderingPolicyType, fwkplugin.StabilityStable, fcfs.FCFSOrderingPolicyFactory)
	fwkplugin.Register(globalstrict.GlobalStrictFairnessPolicyType, fwkplugin.StabilityStable, globalstrict.GlobalStrictFairnessPolicyFactory)
	fwkplugin.Register(usagelimits.StaticUsageLimitPolicyType, fwkplugin.StabilityBeta, usagelimits.StaticPolicyFactory)
	fwkplugin.Register(bandselection.StrictBandSelectionPolicyType, fwkplugin.StabilityBeta, bandselection.StrictPolicyFactory)
	// and the request parsers it fabricates when none are configured
	fwkplugin.Register(openai.OpenAIParserType, fwkplugin.StabilityStable, openai.OpenAIParserPluginFactory)
	fwkplugin.Register(anthropic.AnthropicParserType, fwkplugin.StabilityStable, anthropic.AnthropicParserPluginFactory)
	fwkplugin.Register(vllmhttp.VllmHTTPParserType, fwkplugin.StabilityStable, vllmhttp.VllmHTTPParserPluginFactory)
	fwkplugin.Register(systemone.SystemOneParserType, fwkplugin.StabilityStable, systemone.SystemOneParserPluginFactory)
	fwkplugin.Register(passthrough.PassthroughParserType, fwkplugin.StabilityStable, passthrough.PassthroughParserPluginFactory)
	// and the metrics source + extractor of the data-layer default
	fwkplugin.Register(sourcemetrics.MetricsDataSourceType, fwkplugin.StabilityStable, sourcemetrics.MetricsDataSourceFactory)
	fwkplugin.Register(extractormetrics.MetricsExtractorType, fwkplugin.StabilityStable, extractormetrics.CoreMetricsExtractorFactory)
}

// corePluginConfig is the GSCH-CORE arm's plugin chain, exactly as a deployment renders
// it: thread-identity (the member) → token-producer (the block keys) → the group
// scheduler, the FAIRNESS POLICY, the FILTER (the seat pin) AND the SATURATION
// DETECTOR — one instance behind THREE references — and nothing else that can decide
// anything.
const corePluginConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: thread-identity
  parameters:
    blockSizeTokens: 64
    logRequests: true
    tokenProducer:
      modelName: test-model
      vllm:
        url: http://localhost:8000
- type: token-producer
  parameters:
    modelName: test-model
    vllm:
      url: http://localhost:8000
- type: group-scheduler
  name: group
  parameters:
    groupBudgetFraction: 0.95
    quantum: 4
    idleSeconds: 20
    inFlightCap: 64
    blockSizeTokens: 64
    inflightReapSeconds: 3700
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: group
flowControl:
  defaultRequestTTL: 0s
  priorityBands:
  - priority: 0
    fairnessPolicyRef: group
  saturationDetector:
    pluginRef: group
`

// gateStartedInProcess records that a full config has already been instantiated in THIS
// test binary. The plugins register their series into the controller-runtime metrics
// registry, which is process-global, so a second full start in one process fails on a
// duplicate collector — a property of the harness (one EPP per process), not of a config.
var gateStartedInProcess atomic.Bool

// loadCoreChain runs the runner's configuration sequence through its exported pieces and
// returns the handle holding the instantiated plugins.
func loadCoreChain(t *testing.T) fwkplugin.Handle {
	t.Helper()

	// Phase one, as the runner runs it: parse and validate the raw config.
	rawConfig, _, err := loader.LoadRawConfig([]byte(corePluginConfig), logr.Discard())
	require.NoError(t, err, "the core chain config must parse")

	// Phase two: the exported pieces the runner's parseConfigurationPhaseTwo is made of.
	handle := fwkplugin.NewEppHandle(context.Background(), nil,
		fwkplugin.WithMetricsRecorder(ctrlmetrics.Registry))
	cfg, err := loader.InstantiateAndConfigure(rawConfig, handle, logr.Discard())
	require.NoError(t, err, "the core chain must instantiate its plugins")

	pollStatus := datalayer.NewPollStatusProducer(cfg.DataConfig)
	handle.AddPlugin(pollStatus.TypedName().Name, pollStatus)
	require.NoError(t, datalayer.CreateMissingDataProducers(context.Background(),
		fwkplugin.DefaultProducerRegistry, fwkplugin.Registry, handle),
		"the auto-created default producers must resolve")

	// Producer phases exactly as the runner resolves them (a phase-less validation
	// would silently check a DIFFERENT order than the one startup uses).
	producerPhases := datalayer.ResolveDataProducerPhases(handle.GetAllPlugins(), cfg.DataProducerPhases)

	// RegisterDependencies, as the runner does, against a real datalayer runtime.
	rt := datalayer.NewRuntime(0)
	for _, p := range handle.GetAllPlugins() {
		if registrant, ok := p.(fwkdl.Registrant); ok {
			require.NoError(t, registrant.RegisterDependencies(rt), "plugin %s RegisterDependencies", p.TypedName())
		}
	}

	// The startup DAG gate: valid layer order, no cycles.
	_, err = datalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins(),
		datalayer.WithDataProducerPhases(producerPhases))
	require.NoError(t, err, "the plugin DAG / layer order must be valid")

	require.NoError(t, fwkplugin.ValidatePluginStability(handle, true),
		"alpha plugins with the experimental gate on must pass the stability check")

	gateStartedInProcess.Store(true)
	return handle
}

// TestGroupSchedulerCoreChainStarts is the config-level DAG gate: one instance,
// referenced as the fairness policy, the filter and the saturation detector, and the
// EPP configuration phases complete.
func TestGroupSchedulerCoreChainStarts(t *testing.T) {
	if gateStartedInProcess.Load() {
		t.Skip("a full config already started in this process: the metrics registry is process-global")
	}
	handle := loadCoreChain(t)

	// ONE INSTANCE: exactly one plugin carries the name both references use.
	var found []fwkplugin.Plugin
	for _, p := range handle.GetAllPlugins() {
		if p.TypedName().Name == "group" {
			found = append(found, p)
		}
	}
	require.Len(t, found, 1, "the dual reference must resolve to ONE instance, not two (%d found)", len(found))
	named := handle.Plugin("group")
	require.NotNil(t, named)

	// both interfaces, one object — that is what makes the layer check pass and the
	// pin possible.
	_, isFilter := named.(fwksched.Filter)
	require.True(t, isFilter, "the group scheduler must implement the scheduling Filter")
	_, isPolicy := named.(fwkfc.FairnessPolicy)
	require.True(t, isPolicy, "the group scheduler must implement the FairnessPolicy")
	det, isDetector := named.(fwkfc.SaturationDetector)
	require.True(t, isDetector,
		"the group scheduler must implement the SaturationDetector: a core arm runs ONE decision plugin, and the pool gate is one of its three answers")

	// THE DETECTOR-IDENTITY GATE: the configured detector must be THIS plugin, and the
	// stock utilization-detector must not be in the config at all. (The loader
	// FABRICATES the stock detector when the slot names nothing — loader/defaults.go —
	// so "no error" is not enough: the identity has to be checked.)
	for _, p := range handle.GetAllPlugins() {
		require.NotEqual(t, "utilization-detector", p.TypedName().Type,
			"a core arm must not carry the stock roofline detector: exactly one detector answers the gate, and it is the group scheduler")
	}
	require.Equal(t, 0.0, det.Saturation(context.Background(), nil),
		"the default detector answer is NOT saturated: the seat budget is the admission control")

	// the DAG the EPP builds at startup: the group scheduler must appear in the
	// execution order EXACTLY ONCE (one instance behind three references).
	producerPhases := datalayer.ResolveDataProducerPhases(handle.GetAllPlugins(), nil)
	order, err := datalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins(),
		datalayer.WithDataProducerPhases(producerPhases))
	require.NoError(t, err, "the plugin DAG / layer order must be valid")
	var occurrences int
	for _, name := range order {
		if strings.Contains(name, "group") {
			occurrences++
		}
	}
	require.Equal(t, 1, occurrences, "the group scheduler must appear once in the execution order, got %v", order)
}
