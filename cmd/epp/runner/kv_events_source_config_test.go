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
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	srckvevents "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/kvevents"
	preciseproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/preciseprefixcache"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// TestKVEventsSourceRefConfigLoad loads the producer ahead of the source it
// names, so the loader must order the source first from the pluginRef.
func TestKVEventsSourceRefConfigLoad(t *testing.T) {
	for name, tc := range map[string]struct {
		ref     string
		wantErr string
	}{
		"bound":          {ref: "events"},
		"unknown source": {ref: "missing", wantErr: "missing"},
	} {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			opts := runserver.NewOptions()
			opts.ConfigText = fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: %s
    name: cache
    parameters:
      kvEventsSourceRef: %s
  - type: %s
    name: events
  - type: token-producer
    parameters:
      backend: vllm
      modelName: model
  - type: max-score-picker
  - type: single-profile-handler
`, preciseproducer.PluginType, tc.ref, srckvevents.PluginType)
			opts.PoolName = "kv-events-pool"
			opts.AllowExperimentalPlugins = true

			r := NewRunner()
			rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
			if err == nil {
				ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
				_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
			}
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			source, ok := r.PluginHandle.Plugin("events").(*srckvevents.Source)
			require.True(t, ok, "plugin events must be a %s", srckvevents.PluginType)
			state, err := source.DumpState()
			require.NoError(t, err)
			require.Contains(t, string(state), `"consumers":["cache/`+preciseproducer.PluginType+`"]`)
		})
	}
}
