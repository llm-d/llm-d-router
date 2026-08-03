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

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// TestMultiClusterConfigLoads guards the multicluster wiring documented in
// framework/plugins/README.md: the discovery, metrics source, metrics extractor,
// and the two metric scorers instantiate against the registry and resolve to their
// expected types as one EndpointPickerConfig.
func TestMultiClusterConfigLoads(t *testing.T) {
	dir := t.TempDir()
	clustersPath := filepath.Join(dir, "clusters.yaml")
	require.NoError(t, os.WriteFile(clustersPath, []byte(
		"endpoints:\n"+
			"  - name: a\n"+
			"    address: gw.cluster-a.example.com\n"+
			"    port: \"443\"\n"), 0o644))

	// Mirrors the example config in framework/plugins/README.md, so this test fails if
	// that documented wiring stops loading. caCertPath is omitted because it points at a
	// deploy-specific file. CA loading is covered by the metrics source's own tests.
	configText := fmt.Sprintf(`apiVersion: llm-d.ai/v1alpha1
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
  - type: multicluster-metrics-extractor
    name: metrics-extractor
  - type: multicluster-session-affinity-filter
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
      extractors:
        - pluginRef: metrics-extractor
`, clustersPath)

	opts := runserver.NewOptions()
	opts.ConfigText = configText
	opts.PoolName = "mc-config-pool"
	opts.PoolNamespace = "mc-config-ns"

	r := NewRunner()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err, "documented multicluster config must parse and validate")
	require.NotNil(t, rawConfig.DataLayer)
	require.NotNil(t, rawConfig.DataLayer.Discovery)
	require.Len(t, rawConfig.DataLayer.Sources, 1)

	// Phase two instantiates each plugin against the registry, so it fails if a
	// multicluster factory is unregistered or a type is misspelled. Phase one alone
	// only decodes the YAML and would not catch that.
	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds)
	require.NoError(t, err, "documented multicluster config must instantiate against the registry")

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
}
