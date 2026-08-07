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
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// TestMultiClusterConfigLoad drives the documented multicluster wiring
// (framework/plugins/README.md) through both config phases. With the metrics extractor
// present the plugins resolve. Without it, the metric scorers' required pool attributes
// have no producer and load fails (the #2255 guard).
func TestMultiClusterConfigLoad(t *testing.T) {
	tests := []struct {
		name          string
		withExtractor bool
		wantErr       error
	}{
		{name: "full config loads", withExtractor: true},
		{name: "missing metrics extractor fails at load", withExtractor: false, wantErr: datalayer.ErrNoDefaultProducer},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			clustersPath := filepath.Join(dir, "clusters.yaml")
			require.NoError(t, os.WriteFile(clustersPath, []byte(
				"endpoints:\n  - name: a\n    address: gw.cluster-a.example.com\n    port: \"443\"\n"), 0o644))

			opts := runserver.NewOptions()
			opts.ConfigText = multiClusterConfig(clustersPath, tt.withExtractor)
			opts.PoolName = "mc-config-pool"
			opts.PoolNamespace = "mc-config-ns"

			r := NewRunner()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			// Phase one decodes and validates references. The missing producer only
			// surfaces in phase two, which builds the produce/consume graph.
			rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err)

			ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
			_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds)
			if tt.wantErr != nil {
				require.ErrorIs(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)

			for name, wantType := range map[string]string{
				"discovery":         "multicluster-file-discovery",
				"metrics-source":    "multicluster-metrics-data-source",
				"metrics-extractor": "multicluster-metrics-extractor",
				"session-affinity":  "multicluster-session-affinity-filter",
				"kv-cache":          "multicluster-kv-cache-utilization-scorer",
				"queue":             "multicluster-queue-scorer",
			} {
				p := r.PluginHandle.Plugin(name)
				require.NotNil(t, p, "plugin %q must resolve", name)
				require.Equal(t, wantType, p.TypedName().Type, "plugin %q resolved to the wrong type", name)
			}
		})
	}
}

// multiClusterConfig mirrors the example in framework/plugins/README.md. When
// withExtractor is false it drops the metrics extractor and its source binding.
func multiClusterConfig(clustersPath string, withExtractor bool) string {
	var extractorPlugin, extractorBinding string
	if withExtractor {
		extractorPlugin = "  - type: multicluster-metrics-extractor\n    name: metrics-extractor\n"
		extractorBinding = "      extractors:\n        - pluginRef: metrics-extractor\n"
	}
	return fmt.Sprintf(`apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: multicluster-file-discovery
    name: discovery
    parameters:
      path: %q
  - type: multicluster-metrics-data-source
    name: metrics-source
    parameters:
      scheme: https
%s  - type: multicluster-session-affinity-filter
    name: session-affinity
  - type: multicluster-kv-cache-utilization-scorer
    name: kv-cache
  - type: multicluster-queue-scorer
    name: queue
  - type: max-score-picker
  - type: single-profile-handler
dataLayer:
  discovery:
    pluginRef: discovery
  sources:
    - pluginRef: metrics-source
%s`, clustersPath, extractorPlugin, extractorBinding)
}
