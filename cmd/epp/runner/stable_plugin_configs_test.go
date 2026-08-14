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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// stableConfigsGlob matches every frozen plugin config, across major versions.
const stableConfigsGlob = "../../../test/testdata/plugins/stable/v*/*.yaml"

// TestStablePluginConfigs loads every frozen config under test/testdata/plugins/stable/
// and fails if one no longer parses, no longer instantiates its plugins, or pulls in a
// plugin that is not at least Beta.
//
// These configs are the written-down form of the promise made when a plugin is promoted
// to Stable: a configuration valid today stays valid for the whole major version. The
// files must not be edited to make this test pass — a failure means a stable plugin
// changed incompatibly. See test/testdata/plugins/stable/README.md.
func TestStablePluginConfigs(t *testing.T) {
	files, err := filepath.Glob(stableConfigsGlob)
	require.NoError(t, err, "failed to glob stable plugin configs")
	require.NotEmpty(t, files, "no stable plugin configs found at %s", stableConfigsGlob)

	for _, file := range files {
		t.Run(testName(file), func(t *testing.T) {
			configText, err := os.ReadFile(file)
			require.NoError(t, err, "failed to read %s", file)

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			opts := runserver.NewOptions()
			opts.ConfigText = string(configText)
			opts.PoolName = "stable-config-pool"
			// Stable configs must load on a default command line. Leaving this false
			// also asserts that no config here depends on an Alpha plugin.
			opts.AllowExperimentalPlugins = false

			r := NewRunner()
			rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err, "stable config %s no longer parses", file)

			ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
			_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds)
			require.NoError(t, err, "stable config %s no longer instantiates its plugins", file)

			require.NoError(t, fwkplugin.ValidatePluginStability(r.PluginHandle, opts.AllowExperimentalPlugins),
				"stable config %s references a plugin that is not at least Beta", file)
		})
	}
}

// testName renders a config path as "<version>/<plugin type>" so a failure names the
// plugin whose contract broke.
func testName(file string) string {
	return filepath.Join(filepath.Base(filepath.Dir(file)), filepath.Base(file))
}
