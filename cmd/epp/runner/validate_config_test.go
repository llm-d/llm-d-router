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
	"strings"
	"testing"

	"github.com/spf13/pflag"
	"github.com/stretchr/testify/require"

	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// validateConfigBase is test/e2e/artifacts/config/sim-epp-config.yaml.
const validateConfigBase = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
  parameters:
    maxPrefixTokensToMatch: 16384
    lruCapacityPerServer: 31250
- type: prefix-cache-scorer
- type: decode-filter
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
`

func TestValidateConfig(t *testing.T) {
	// k8s-peer-discovery rejects a namespace that differs from NAMESPACE.
	t.Setenv("NAMESPACE", "")
	require.NoError(t, os.Unsetenv("NAMESPACE"))

	endpointsFile := filepath.Join(t.TempDir(), "endpoints.yaml")
	require.NoError(t, os.WriteFile(endpointsFile, []byte(`endpoints:
- name: e0
  address: 127.0.0.1
  port: "8000"
`), 0o600))

	invalidObjectivesFile := filepath.Join(t.TempDir(), "objectives.yaml")
	require.NoError(t, os.WriteFile(invalidObjectivesFile, []byte("objectives:\n- priority: 10\n"), 0o600))

	fileDiscoveryConfig := editBase(t, "plugins:\n- type: approx-prefix-cache-producer",
		"plugins:\n- name: file-disc\n  type: file-discovery\n  parameters:\n    path: "+endpointsFile+
			"\n- type: approx-prefix-cache-producer") + `dataLayer:
  discovery:
    endpoints:
      pluginRef: file-disc
`

	peersConfig := func(ref string) string {
		return editBase(t, "plugins:\n- type: approx-prefix-cache-producer",
			"plugins:\n- name: peers\n  type: k8s-peer-discovery\n  parameters:\n    selector: app=epp\n    port: \"9002\"\n    namespace: peer-ns"+
				"\n- type: approx-prefix-cache-producer") + `dataLayer:
  discovery:
    peers:
      pluginRef: ` + ref + "\n"
	}

	tests := []struct {
		name           string
		configText     string
		poolName       string
		poolNamespace  string
		objectivesFile string
		wantErr        string
	}{
		{
			name:       "valid config",
			configText: validateConfigBase,
			poolName:   "epp",
		},
		{
			name:       "valid config with file discovery",
			configText: fileDiscoveryConfig,
		},
		{
			name:       "unregistered plugin type",
			configText: editBase(t, "type: max-score-picker", "type: max-score-pickr"),
			poolName:   "epp",
			wantErr:    "plugin type 'max-score-pickr' is not registered",
		},
		{
			name:       "profile references an undefined plugin",
			configText: editBase(t, "pluginRef: decode-filter", "pluginRef: no-such-plugin"),
			poolName:   "epp",
			wantErr:    "no-such-plugin",
		},
		{
			name:       "kubernetes mode without pool-name or endpoint-selector",
			configText: validateConfigBase,
			wantErr:    "either pool-name or endpoint-selector must be set",
		},
		{
			name:           "objectives file in kubernetes mode",
			configText:     validateConfigBase,
			poolName:       "epp",
			objectivesFile: invalidObjectivesFile,
			wantErr:        "objectives-file requires file discovery",
		},
		{
			name:           "invalid objectives file with file discovery",
			configText:     fileDiscoveryConfig,
			objectivesFile: invalidObjectivesFile,
			wantErr:        "objective 0: name is required",
		},
		{
			name:           "missing objectives file with file discovery",
			configText:     fileDiscoveryConfig,
			objectivesFile: filepath.Join(t.TempDir(), "missing.yaml"),
			wantErr:        "missing.yaml",
		},
		{
			name:       "peer discovery references an undefined plugin",
			configText: peersConfig("no-such-peers"),
			poolName:   "epp",
			wantErr:    `peerDiscovery: no plugin found with name "no-such-peers"`,
		},
		{
			name:       "peer discovery references a plugin that is not a PeerDiscovery",
			configText: peersConfig("decode-filter"),
			poolName:   "epp",
			wantErr:    `peerDiscovery: plugin "decode-filter" does not implement PeerDiscovery`,
		},
		{
			name:          "peer discovery namespace differs from the pool namespace",
			configText:    peersConfig("peers"),
			poolName:      "epp",
			poolNamespace: "pool-ns",
			wantErr:       `namespace "peer-ns" must match pool namespace "pool-ns"`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			opts := runserver.NewOptions()
			opts.ConfigText = tc.configText
			opts.PoolName = tc.poolName
			opts.PoolNamespace = tc.poolNamespace
			opts.ObjectivesFile = tc.objectivesFile
			opts.ValidateConfig = true

			r := NewRunner()
			rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err)

			err = r.validateConfig(ctx, opts, rawConfig)
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorContains(t, err, tc.wantErr)
		})
	}
}

// TestRunValidateConfig drives --validate-config through Run, which main turns
// into the exit code: a nil error exits 0 and any error exits 1.
func TestRunValidateConfig(t *testing.T) {
	tests := []struct {
		name       string
		configText string
		args       []string
		wantErr    string
	}{
		{
			name:       "valid config",
			configText: validateConfigBase,
			args:       []string{"--pool-name=epp"},
		},
		{
			name:       "invalid config",
			configText: editBase(t, "type: max-score-picker", "type: max-score-pickr"),
			args:       []string{"--pool-name=epp"},
			wantErr:    "plugin type 'max-score-pickr' is not registered",
		},
		{
			name:       "kubernetes mode without pool-name or endpoint-selector",
			configText: validateConfigBase,
			wantErr:    "either pool-name or endpoint-selector must be set",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Run parses os.Args into the global pflag.CommandLine, so each case
			// gets its own arguments and a fresh flag set.
			origArgs, origFlags := os.Args, pflag.CommandLine
			t.Cleanup(func() { os.Args, pflag.CommandLine = origArgs, origFlags })
			os.Args = append([]string{"epp", "--validate-config", "--tracing=false", "--config-text=" + tc.configText}, tc.args...)
			pflag.CommandLine = pflag.NewFlagSet("epp", pflag.ContinueOnError)

			err := NewRunner().Run(context.Background())
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorContains(t, err, tc.wantErr)
		})
	}
}

// editBase replaces old in validateConfigBase, which must contain it exactly once.
func editBase(t *testing.T, old, replacement string) string {
	t.Helper()
	require.Equal(t, 1, strings.Count(validateConfigBase, old), "%q must occur exactly once", old)
	return strings.Replace(validateConfigBase, old, replacement, 1)
}
