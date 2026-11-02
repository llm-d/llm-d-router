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

package runner

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/thunderagent"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

func TestThunderAgentSampleConfigLoads(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	configBytes, err := os.ReadFile(filepath.Join("..", "..", "..", "test", "e2e", "artifacts", "config", "thunderagent-config.yaml"))
	require.NoError(t, err)

	opts := runserver.NewOptions()
	opts.ConfigText = string(configBytes)
	opts.PoolName = "thunder-test-pool"

	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)
	require.Empty(t, rawConfig.SchedulingProfiles)

	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
	require.NoError(t, err)

	ta := r.PluginHandle.Plugin(thunderagent.ThunderAgentPluginType)
	require.NotNil(t, ta)
	require.Same(t, ta, eppConfig.SaturationDetector)

	require.NotNil(t, eppConfig.FlowControlConfig)
	require.Same(t, ta, eppConfig.FlowControlConfig.Registry.DefaultPriorityBand.FairnessPolicy)

	require.Len(t, rawConfig.SchedulingProfiles, 1)
	profile := rawConfig.SchedulingProfiles[0]
	require.Equal(t, "default", profile.Name)
	refs := make([]string, 0, len(profile.Plugins))
	for _, p := range profile.Plugins {
		refs = append(refs, p.PluginRef)
	}
	require.ElementsMatch(t, []string{thunderagent.ThunderAgentPluginType, maxscore.MaxScorePickerType}, refs)
	require.NotNil(t, r.PluginHandle.Plugin(single.SingleProfileHandlerType))
	require.Contains(t, eppConfig.SchedulerConfig.String(), thunderagent.ThunderAgentPluginType)
}
