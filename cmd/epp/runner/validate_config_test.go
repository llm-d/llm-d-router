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
	endpointsFile := filepath.Join(t.TempDir(), "endpoints.yaml")
	require.NoError(t, os.WriteFile(endpointsFile, []byte(`endpoints:
- name: e0
  address: 127.0.0.1
  port: "8000"
`), 0o600))

	invalidObjectivesFile := filepath.Join(t.TempDir(), "objectives.yaml")
	require.NoError(t, os.WriteFile(invalidObjectivesFile, []byte("objectives:\n- priority: 10\n"), 0o600))

	fileDiscoveryConfig := replaceOnce(t, validateConfigBase, "plugins:\n- type: approx-prefix-cache-producer",
		"plugins:\n- name: file-disc\n  type: file-discovery\n  parameters:\n    path: "+endpointsFile+
			"\n- type: approx-prefix-cache-producer") + `dataLayer:
  discovery:
    endpoints:
      pluginRef: file-disc
`

	tests := []struct {
		name           string
		configText     string
		poolName       string
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
			configText: replaceOnce(t, validateConfigBase, "type: max-score-picker", "type: max-score-pickr"),
			poolName:   "epp",
			wantErr:    "plugin type 'max-score-pickr' is not registered",
		},
		{
			name:       "profile references an undefined plugin",
			configText: replaceOnce(t, validateConfigBase, "pluginRef: decode-filter", "pluginRef: no-such-plugin"),
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
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			opts := runserver.NewOptions()
			opts.ConfigText = tc.configText
			opts.PoolName = tc.poolName
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

// replaceOnce replaces old in s, which must occur exactly once.
func replaceOnce(t *testing.T, s, old, replacement string) string {
	t.Helper()
	require.Equal(t, 1, strings.Count(s, old), "%q must occur exactly once", old)
	return strings.Replace(s, old, replacement, 1)
}
