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
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// TestThunderAgentSampleConfigLoads verifies the shipped thunder-agent sample
// config parses, instantiates its plugins, and builds a flow control config
// through the production configuration path. The config references one
// thunder-agent instance from three slots (scorer, saturation detector,
// fairness policy), so a pass proves the by-name plugin resolution accepts
// the same instance in all of them.
func TestThunderAgentSampleConfigLoads(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	configBytes, err := os.ReadFile(filepath.Join("..", "..", "..", "deploy", "config", "thunderagent-config.yaml"))
	require.NoError(t, err)

	opts := runserver.NewOptions()
	opts.ConfigText = string(configBytes)
	opts.PoolName = "thunder-test-pool"

	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)

	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds)
	require.NoError(t, err)
	require.NotNil(t, eppConfig.FlowControlConfig,
		"the sample config enables the flowControl gate, so a flow control config should be built")
}
