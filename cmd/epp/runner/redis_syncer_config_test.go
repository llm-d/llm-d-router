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
	"fmt"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

func TestRedisSyncerPluginNames(t *testing.T) {
	for _, pluginType := range []string{"redis-syncer", "redis-state-store"} {
		for _, configuredName := range []string{"", "shared-state"} {
			t.Run(pluginType+"/"+configuredName, func(t *testing.T) {
				server := miniredis.RunT(t)
				pluginRef := configuredName
				if pluginRef == "" {
					pluginRef = pluginType
				}
				opts := runserver.NewOptions()
				opts.PoolName = "redis-syncer-test"
				opts.ConfigText = fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: %s
    name: %q
    parameters:
      address: %q
dataLayer:
  crossReplica:
    syncerPluginRef: %s
`, pluginType, configuredName, server.Addr(), pluginRef)

				r := NewRunner()
				rawConfig, err := r.parseConfigurationPhaseOne(t.Context(), opts)
				require.NoError(t, err)
				ds := datastore.NewDatastore(t.Context(), r.setupMetricsCollection(opts))
				_, err = r.parseConfigurationPhaseTwo(t.Context(), rawConfig, ds, opts.RefreshMetricsInterval)
				require.NoError(t, err)

				plugin := r.PluginHandle.CrossReplicaSyncer()
				require.NotNil(t, plugin)
				require.Equal(t, "redis-syncer", plugin.TypedName().Type)
				require.Equal(t, pluginRef, plugin.TypedName().Name)
				syncer, ok := plugin.(fwkdl.CrossReplicaSyncer)
				require.True(t, ok)
				value, existed, err := syncer.GetOrSet(t.Context(), "test", "request", "first")
				require.NoError(t, err)
				require.False(t, existed)
				require.Equal(t, "first", value)
				value, existed, err = syncer.GetOrSet(t.Context(), "test", "request", "second")
				require.NoError(t, err)
				require.True(t, existed)
				require.Equal(t, "first", value)
			})
		}
	}
}
