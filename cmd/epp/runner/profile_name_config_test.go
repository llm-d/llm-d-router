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
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// TestSessionAffinityProfileNameConfigLoad verifies the production config path
// passes the configured scheduling profile names to plugin factories, including
// the implicit "default" profile when the config defines none.
func TestSessionAffinityProfileNameConfigLoad(t *testing.T) {
	const withProfiles = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: session-affinity-filter
  name: affinity
  parameters:
    profileName: %s
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: affinity
  - pluginRef: max-score-picker
`
	const withoutProfiles = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: session-affinity-filter
  name: affinity
  parameters:
    profileName: %s
`
	tests := []struct {
		name    string
		config  string
		profile string
		wantErr string
	}{
		{name: "defined profile", config: withProfiles, profile: "prefill"},
		{name: "undefined profile", config: withProfiles, profile: "prefil", wantErr: "scheduling profile 'prefil' is not defined"},
		{name: "implicit default profile", config: withoutProfiles, profile: "default"},
		{name: "undefined profile without schedulingProfiles", config: withoutProfiles, profile: "prefill", wantErr: "scheduling profile 'prefill' is not defined"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			opts := runserver.NewOptions()
			opts.ConfigText = fmt.Sprintf(tt.config, tt.profile)
			opts.PoolName = "profile-name-test-pool"

			r := NewRunner()
			rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err)

			ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
			_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
		})
	}
}
