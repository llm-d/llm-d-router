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
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func reloadStartup(t *testing.T, text string) (*configapiv1.EndpointPickerConfig, []configapiv1.SchedulingProfile, fwkplugin.Handle) {
	t.Helper()
	logger := logging.NewTestLogger()
	startup, _, err := LoadRawConfig([]byte(text), logger)
	require.NoError(t, err)
	source := startup.DeepCopy()
	handle := testutils.NewTestHandle(context.Background())
	_, err = InstantiateAndConfigure(startup, handle, logger)
	require.NoError(t, err)
	return source, startup.SchedulingProfiles, handle
}

func TestBuildSchedulerForReload(t *testing.T) {
	registerTestPlugins(t)
	RegisterFeatureGate(testFeatureGate, true)
	RegisterFeatureGate(flowcontrol.FeatureGate, false)

	startupSource, startupProfiles, handle := reloadStartup(t, successSchedulerConfigText)

	t.Run("profile change", func(t *testing.T) {
		candidate := strings.Replace(successSchedulerConfigText, "weight: 50", "weight: 25", 1)
		scheduler, err := BuildSchedulerForReload([]byte(candidate), startupSource, startupProfiles, handle)
		require.NoError(t, err)
		require.NotNil(t, scheduler)
	})

	t.Run("static change", func(t *testing.T) {
		candidate := strings.Replace(successSchedulerConfigText, "blockSize: 32", "blockSize: 64", 1)
		_, err := BuildSchedulerForReload([]byte(candidate), startupSource, startupProfiles, handle)
		require.ErrorIs(t, err, ErrUnsupportedReload)
	})

	t.Run("profile rename", func(t *testing.T) {
		candidate := strings.Replace(successSchedulerConfigText, "name: default", "name: renamed", 1)
		_, err := BuildSchedulerForReload([]byte(candidate), startupSource, startupProfiles, handle)
		require.ErrorIs(t, err, ErrUnsupportedReload)
	})

	t.Run("explicit profile removed", func(t *testing.T) {
		candidate := strings.Replace(successSchedulerConfigText, "schedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: testScorer\n    weight: 50\n  - pluginRef: maxScorePicker\n", "", 1)
		require.NotEqual(t, successSchedulerConfigText, candidate)
		_, err := BuildSchedulerForReload([]byte(candidate), startupSource, startupProfiles, handle)
		require.ErrorIs(t, err, ErrUnsupportedReload)
	})

	t.Run("duplicate profile name", func(t *testing.T) {
		candidate := strings.Replace(successSchedulerConfigText, "dataLayer:\n", "- name: default\n  plugins:\n  - pluginRef: maxScorePicker\ndataLayer:\n", 1)
		_, err := BuildSchedulerForReload([]byte(candidate), startupSource, startupProfiles, handle)
		require.ErrorContains(t, err, "duplicate name")
	})

	t.Run("large static integer change", func(t *testing.T) {
		startupText := strings.Replace(successSchedulerConfigText, "blockSize: 32", "blockSize: 9007199254740992", 1)
		largeSource, largeProfiles, largeHandle := reloadStartup(t, startupText)

		candidate := strings.Replace(startupText, "blockSize: 9007199254740992", "blockSize: 9007199254740993", 1)
		_, err := BuildSchedulerForReload([]byte(candidate), largeSource, largeProfiles, largeHandle)
		require.ErrorIs(t, err, ErrUnsupportedReload)
	})

	t.Run("empty file", func(t *testing.T) {
		_, err := BuildSchedulerForReload(nil, startupSource, startupProfiles, handle)
		require.Error(t, err)
		require.NotErrorIs(t, err, ErrUnsupportedReload)
	})

	t.Run("startup non-scorer weight", func(t *testing.T) {
		startupText := strings.Replace(successSchedulerConfigText, "  - pluginRef: maxScorePicker\n", "  - pluginRef: maxScorePicker\n    weight: 2\n", 1)
		require.NotEqual(t, successSchedulerConfigText, startupText)
		weightedSource, weightedProfiles, weightedHandle := reloadStartup(t, startupText)

		_, err := BuildSchedulerForReload([]byte(startupText), weightedSource, weightedProfiles, weightedHandle)
		require.NoError(t, err)
	})
}

func TestBuildSchedulerForReloadRetainsSaturationFilter(t *testing.T) {
	registerTestPlugins(t)
	RegisterFeatureGate(testFeatureGate, true)
	RegisterFeatureGate(flowcontrol.FeatureGate, false)

	const detectorType = "test-saturation-filter"
	fwkplugin.Register(detectorType, fwkplugin.StabilityStable, func(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return &mockSaturationFilter{mockSaturationDetector{mockPlugin{t: fwkplugin.TypedName{Name: name, Type: detectorType}}}}, nil
	})

	startupText := strings.Replace(successSchedulerConfigText, "schedulingProfiles:\n",
		"- name: utilization-detector\n  type: test-saturation-filter\nschedulingProfiles:\n", 1)
	startupSource, startupProfiles, handle := reloadStartup(t, startupText)
	require.Equal(t, "utilization-detector", startupProfiles[0].Plugins[len(startupProfiles[0].Plugins)-1].PluginRef)

	candidate := strings.Replace(startupText, "weight: 50", "weight: 25", 1)
	require.NotEqual(t, startupText, candidate)
	scheduler, err := BuildSchedulerForReload([]byte(candidate), startupSource, startupProfiles, handle)
	require.NoError(t, err)

	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, nil, nil)
	_, err = scheduler.Schedule(context.Background(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{endpoint})
	require.ErrorContains(t, err, "no endpoints available")
}

func TestApplySchedulingDefaultsForReloadKeepsImplicitProfileOrder(t *testing.T) {
	handle := testutils.NewTestHandle(context.Background())
	for _, name := range []string{"scorer-a", "scorer-b", "scorer-c"} {
		handle.AddPlugin(name, &mockScorer{mockPlugin{t: fwkplugin.TypedName{Name: name, Type: testScorerType}}})
	}
	handle.AddPlugin("picker", &mockPicker{mockPlugin{t: fwkplugin.TypedName{Name: "picker", Type: testPickerType}}})
	handle.AddPlugin("handler", single.NewSingleProfileHandler())
	weight := DefaultScorerWeight
	startupProfiles := []configapiv1.SchedulingProfile{{
		Name: "default",
		Plugins: []configapiv1.SchedulingPlugin{
			{PluginRef: "scorer-a", Weight: &weight},
			{PluginRef: "scorer-b", Weight: &weight},
			{PluginRef: "scorer-c", Weight: &weight},
			{PluginRef: "picker"},
		},
	}}

	for range 32 {
		candidate := &configapiv1.EndpointPickerConfig{}
		require.NoError(t, applySchedulingDefaultsForReload(candidate, startupProfiles, handle))
		require.Equal(t, startupProfiles, candidate.SchedulingProfiles)
	}
}

func TestApplySchedulingDefaultsForReloadKeepsStartupPicker(t *testing.T) {
	handle := testutils.NewTestHandle(context.Background())
	handle.AddPlugin("picker-a", &mockPicker{mockPlugin{t: fwkplugin.TypedName{Name: "picker-a", Type: testPickerType}}})
	handle.AddPlugin("picker-b", &mockPicker{mockPlugin{t: fwkplugin.TypedName{Name: "picker-b", Type: testPickerType}}})
	handle.AddPlugin("handler", single.NewSingleProfileHandler())
	startupProfiles := []configapiv1.SchedulingProfile{{
		Name: "default", Plugins: []configapiv1.SchedulingPlugin{{PluginRef: "picker-b"}},
	}}

	candidate := &configapiv1.EndpointPickerConfig{SchedulingProfiles: []configapiv1.SchedulingProfile{{Name: "default"}}}
	require.NoError(t, applySchedulingDefaultsForReload(candidate, startupProfiles, handle))
	require.Equal(t, []configapiv1.SchedulingPlugin{{PluginRef: "picker-b"}}, candidate.SchedulingProfiles[0].Plugins)

	candidate = &configapiv1.EndpointPickerConfig{SchedulingProfiles: []configapiv1.SchedulingProfile{{
		Name: "default", Plugins: []configapiv1.SchedulingPlugin{{PluginRef: "picker-a"}},
	}}}
	require.NoError(t, applySchedulingDefaultsForReload(candidate, startupProfiles, handle))
	require.Equal(t, []configapiv1.SchedulingPlugin{{PluginRef: "picker-a"}}, candidate.SchedulingProfiles[0].Plugins)
}
