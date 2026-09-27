/*
Copyright 2026 The Kubernetes Authors.

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

package runner

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/config/loader"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

type reloadTestPicker struct {
	name   string
	target string
}

func (p *reloadTestPicker) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "reload-test-picker", Name: p.name}
}

func (p *reloadTestPicker) Pick(_ context.Context, candidates []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	for _, candidate := range candidates {
		if candidate.GetMetadata().ID.Name == p.target {
			return &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{candidate.Endpoint}}
		}
	}
	return nil
}

func TestConfigFileReloadChangesPicker(t *testing.T) {
	const startupText = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: picker-a
  type: reload-test-picker
- name: picker-b
  type: reload-test-picker
- name: handler
  type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: picker-a
`
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte(startupText), 0o600))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	startup, _, err := loader.LoadRawConfig([]byte(startupText), logging.NewTestLogger())
	require.NoError(t, err)
	pickerA := &reloadTestPicker{name: "picker-a", target: "endpoint-a"}
	pickerB := &reloadTestPicker{name: "picker-b", target: "endpoint-b"}
	handle := fwkplugin.NewEppHandle(ctx, nil)
	handle.AddPlugin("picker-a", pickerA)
	handle.AddPlugin("picker-b", pickerB)
	profileHandler := single.NewSingleProfileHandler()
	handle.AddPlugin("handler", profileHandler)
	initialProfile := scheduling.NewSchedulerProfile().WithPicker(pickerA)
	initial := scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(profileHandler,
		map[string]fwksched.SchedulerProfile{"default": initialProfile}))
	r := &Runner{
		rawConfig:           startup,
		startupSourceConfig: startup.DeepCopy(),
		startupConfigBytes:  []byte(startupText),
		PluginHandle:        handle,
	}
	opts := runserver.NewOptions()
	opts.ConfigFile = path
	opts.WatchConfigFile = true
	runtimeScheduler := r.newRuntimeScheduler(opts, initial)
	require.NotNil(t, r.reloadableScheduler)
	require.NoError(t, r.startConfigWatcher(ctx, opts))

	endpoints := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "endpoint-a"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "endpoint-b"}}, nil, nil),
	}
	selectedEndpoint := func() string {
		result, err := runtimeScheduler.Schedule(ctx, &fwksched.InferenceRequest{}, endpoints)
		require.NoError(t, err)
		return result.ProfileResults[result.PrimaryProfileName].TargetEndpoints[0].GetMetadata().ID.Name
	}
	require.Equal(t, "endpoint-a", selectedEndpoint())

	candidate := strings.Replace(startupText, "pluginRef: picker-a", "pluginRef: picker-b", 1)
	stagedPath := filepath.Join(dir, "config.yaml.new")
	require.NoError(t, os.WriteFile(stagedPath, []byte(candidate), 0o600))
	require.NoError(t, os.Rename(stagedPath, path))
	require.Eventually(t, func() bool { return r.reloadableScheduler.Generation() == 2 }, 3*time.Second, 10*time.Millisecond)
	require.Equal(t, "endpoint-b", selectedEndpoint())
}

func TestConfigWatcherStopsAfterFileDiscoveryFailure(t *testing.T) {
	dir := t.TempDir()
	endpointsPath := filepath.Join(dir, "endpoints.yaml")
	require.NoError(t, os.WriteFile(endpointsPath, []byte("endpoints: [\n"), 0o600))
	configText := fmt.Sprintf(`apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: file-discovery
  type: file-discovery
  parameters:
    path: %q
    watchFile: false
- name: random-picker
  type: random-picker
- name: max-score-picker
  type: max-score-picker
- name: metrics-source
  type: metrics-data-source
- name: metrics-extractor
  type: core-metrics-extractor
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: random-picker
dataLayer:
  injectDefaults: false
  discovery:
    pluginRef: file-discovery
  sources:
  - pluginRef: metrics-source
    extractors:
    - pluginRef: metrics-extractor
`, endpointsPath)
	configPath := filepath.Join(dir, "config.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(configText), 0o600))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	opts := runserver.NewOptions()
	opts.ConfigFile = configPath
	opts.WatchConfigFile = true
	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)
	err = r.runWithFileDiscovery(ctx, opts, rawConfig)
	require.ErrorContains(t, err, "discovery")
	require.Equal(t, uint64(1), r.reloadableScheduler.Generation())

	candidate := strings.Replace(configText, "pluginRef: random-picker", "pluginRef: max-score-picker", 1)
	require.NoError(t, os.WriteFile(configPath, []byte(candidate), 0o600))
	require.Never(t, func() bool { return r.reloadableScheduler.Generation() != 1 }, time.Second, 10*time.Millisecond)
}
