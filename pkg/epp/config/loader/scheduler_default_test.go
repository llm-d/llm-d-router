/*
Copyright 2025 The Kubernetes Authors.
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

package loader

import (
	"context"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// TestScheduleDefaultProfileFilterWithPrefixAffinity runs the default scheduling
// profile end to end, built from the real default config: the utilization
// detector filters candidates before the queue, KV cache, and prefix cache
// scorers run with the default weights. It pins the interaction between the
// filter and prefix affinity, which the per-plugin tests do not cover, and
// fails if a default weight changes or the filter stops being injected.
func TestScheduleDefaultProfileFilterWithPrefixAffinity(t *testing.T) {
	registerTestPlugins(t)
	// The config-loading tests register a mock saturation detector and only some
	// of the default scorers. Register the real factories the default config
	// names so the test exercises the shipped plugins.
	fwkplugin.Register(utilization.UtilizationDetectorType, fwkplugin.StabilityStable, utilization.UtilizationDetectorFactory)
	fwkplugin.Register(queuedepth.QueueScorerType, fwkplugin.StabilityBeta, queuedepth.QueueScorerFactory)
	fwkplugin.Register(kvcacheutilization.KvCacheUtilizationScorerType, fwkplugin.StabilityBeta, kvcacheutilization.KvCacheUtilizationScorerFactory)

	logger := logging.NewTestLogger()
	rawConfig, _, err := LoadRawConfig(nil, logger)
	require.NoError(t, err)
	cfg, err := InstantiateAndConfigure(rawConfig, testutils.NewTestHandle(context.Background()), logger)
	require.NoError(t, err)
	scheduler := scheduling.NewSchedulerWithConfig(cfg.SchedulerConfig)

	type endpointSpec struct {
		name       string
		queue      int
		kvCache    float64
		prefixFull bool
	}

	tests := []struct {
		name       string
		endpoints  []endpointSpec
		wantPicked string
		wantScores map[string]float64
	}{
		{
			// The full prefix match scores 3.0 and outweighs the queue advantage
			// (weight 2.0) of the idle endpoint while the queue is below the
			// default threshold of 5.
			name: "full prefix match wins below the queue threshold",
			endpoints: []endpointSpec{
				{name: "pod1", queue: 4, kvCache: 0.1, prefixFull: true},
				{name: "pod2", queue: 0, kvCache: 0.1},
			},
			wantPicked: "pod1",
			wantScores: map[string]float64{"pod1": 4.8, "pod2": 3.8},
		},
		{
			// pod1 is removed before scoring, so the queue scorer normalizes over
			// pod2 (queue 0) and pod3 (queue 2) only: pod3 scores 0, not the 0.6
			// it would score if the filtered endpoint still set the range.
			name: "at the queue threshold the prefix match is filtered before scoring",
			endpoints: []endpointSpec{
				{name: "pod1", queue: 5, kvCache: 0.1, prefixFull: true},
				{name: "pod2", queue: 0, kvCache: 0.1},
				{name: "pod3", queue: 2, kvCache: 0.1},
			},
			wantPicked: "pod2",
			wantScores: map[string]float64{"pod2": 3.8, "pod3": 1.8},
		},
		{
			// Every endpoint is at or over the threshold, so the filter fails open
			// and all of them are scored and ranked as usual.
			name: "all endpoints over the threshold are kept and scored",
			endpoints: []endpointSpec{
				{name: "pod1", queue: 5, kvCache: 0.1, prefixFull: true},
				{name: "pod2", queue: 6, kvCache: 0.1},
			},
			wantPicked: "pod1",
			wantScores: map[string]float64{"pod1": 6.8, "pod2": 1.8},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scrapedAt := time.Now()
			endpoints := make([]fwksched.Endpoint, 0, len(test.endpoints))
			for _, spec := range test.endpoints {
				endpoint := fwksched.NewEndpoint(
					&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: spec.name}},
					&fwkdl.Metrics{
						WaitingQueueSize:    spec.queue,
						KVCacheUsagePercent: spec.kvCache,
						UpdateTime:          scrapedAt,
					}, nil)
				matchedBlocks := 0
				if spec.prefixFull {
					matchedBlocks = 10
				}
				endpoint.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(matchedBlocks, 10, 1))
				endpoints = append(endpoints, endpoint)
			}

			result, err := scheduler.Schedule(context.Background(), &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "any-model",
			}, endpoints)
			require.NoError(t, err)

			profileResult := result.ProfileResults["default"]
			require.NotNil(t, profileResult)
			require.Len(t, profileResult.TargetEndpoints, 1)
			assert.Equal(t, test.wantPicked, profileResult.TargetEndpoints[0].GetMetadata().ID.Name)

			gotScores := make(map[string]float64, len(profileResult.ScoredCandidates))
			for _, candidate := range profileResult.ScoredCandidates {
				gotScores[candidate.GetMetadata().ID.Name] = candidate.Score
			}
			require.Len(t, gotScores, len(test.wantScores))
			for name, want := range test.wantScores {
				assert.InDelta(t, want, gotScores[name], 1e-9, "score of %s", name)
			}
		})
	}
}

func TestScheduleFallbackWithInjectedSaturationDetector(t *testing.T) {
	registerTestPlugins(t)
	fwkplugin.Register(concurrency.ConcurrencyDetectorType, fwkplugin.StabilityStable, concurrency.ConcurrencyDetectorFactory)
	fwkplugin.Register(bylabel.LabelSelectorFilterType, fwkplugin.StabilityStable, bylabel.SelectorFactory)
	fwkplugin.RegisterWithPluginDependencies(disagg.DisaggProfileHandlerType, fwkplugin.StabilityStable,
		disagg.HandlerFactory, disagg.DisaggProfileHandlerConfigParser)

	const configText = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- name: dedicated-decode
  type: label-selector-filter
  parameters: {matchLabels: {llm-d.ai/role: decode}}
- name: aggregated-worker
  type: label-selector-filter
  parameters: {matchLabels: {llm-d.ai/role: prefill-decode}}
- type: disagg-profile-handler
  parameters: {profiles: {fallback: fallback}}
- type: concurrency-detector
  parameters: {maxConcurrency: 1, failOpen: false}
flowControl:
  saturationDetector: {pluginRef: concurrency-detector}
schedulingProfiles:
- name: decode
  plugins: [{pluginRef: dedicated-decode}]
- name: fallback
  plugins: [{pluginRef: aggregated-worker}]
`
	logger := logging.NewTestLogger()
	raw, _, err := LoadRawConfig([]byte(configText), logger)
	require.NoError(t, err)
	cfg, err := InstantiateAndConfigure(raw, testutils.NewTestHandle(t.Context()), logger)
	require.NoError(t, err)
	scheduler := scheduling.NewSchedulerWithConfig(cfg.SchedulerConfig)
	endpoint := func(name, role string, requests int64) fwksched.Endpoint {
		pod := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
			ID: k8stypes.NamespacedName{Name: name}, Labels: map[string]string{bylabel.RoleLabel: role},
		}, &fwkdl.Metrics{UpdateTime: time.Now()}, fwkdl.NewAttributes())
		pod.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: requests})
		return pod
	}
	decode := endpoint("decode", bylabel.RoleDecode, 1)
	backup := endpoint("fallback", bylabel.RolePrefillDecode, 0)
	req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "test-model"}
	_, err = scheduler.Schedule(t.Context(), req, []fwksched.Endpoint{decode, backup})
	var rejection errcommon.Error
	require.ErrorAs(t, err, &rejection)
	assert.Equal(t, errcommon.ServiceUnavailable, rejection.Code)
	result, err := scheduler.Schedule(t.Context(), req, []fwksched.Endpoint{backup})
	require.NoError(t, err)
	assert.Equal(t, "fallback", result.PrimaryProfileName)
}
