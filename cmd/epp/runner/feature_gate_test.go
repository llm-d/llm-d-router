/*
Copyright 2025 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/requestcontrol"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// TestHAPopulateNonLeaderDatastoreFeatureGate verifies the gate is enabled by
// default and can be turned off through the featureGates config section.
func TestHAPopulateNonLeaderDatastoreFeatureGate(t *testing.T) {
	ctx := context.Background()

	t.Run("enabled by default", func(t *testing.T) {
		r := NewRunner()
		_, err := r.parseConfigurationPhaseOne(ctx, runserver.NewOptions())
		require.NoError(t, err)
		require.True(t, r.featureGates[runserver.HAPopulateNonLeaderDatastoreFeatureGate])
	})

	t.Run("disabled via config", func(t *testing.T) {
		opts := runserver.NewOptions()
		opts.ConfigText = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- haPopulateNonLeaderDatastore=false
`
		r := NewRunner()
		_, err := r.parseConfigurationPhaseOne(ctx, opts)
		require.NoError(t, err)
		require.False(t, r.featureGates[runserver.HAPopulateNonLeaderDatastoreFeatureGate])
	})
}

// TestFlowControlFeatureGateAdmissionControlWiring exercises the flowControl feature gate through
// the production config path (parseConfigurationPhaseOne -> parseConfigurationPhaseTwo ->
// initAdmissionControl) in both directions:
//   - gate on:  the FlowControlAdmissionController is wired, the loader emits a non-nil
//     FlowControlConfig, and the flow registry is exposed as the priority band control plane;
//   - gate off: the LegacyAdmissionController is wired and no flow control config is built.
//
// The "no featureGates stanza" case reads the gate's registered default from the parsed
// feature-gate map instead of hardcoding it, so this test keeps passing unchanged when the gate
// flips to enabled-by-default (#2104) and pins that the flip actually changes the default wiring.
func TestFlowControlFeatureGateAdmissionControlWiring(t *testing.T) {
	// The deprecated ENABLE_EXPERIMENTAL_FLOW_CONTROL_LAYER env var appends the gate to the config
	// during phase two; clear it so only the featureGates stanza under test drives the outcome.
	if v, ok := os.LookupEnv(enableExperimentalFlowControlLayer); ok {
		require.NoError(t, os.Unsetenv(enableExperimentalFlowControlLayer))
		t.Cleanup(func() { _ = os.Setenv(enableExperimentalFlowControlLayer, v) })
	}

	boolPtr := func(b bool) *bool { return &b }
	testCases := []struct {
		name       string
		configText string
		// wantEnabled nil means "expect whatever default the runner registered for the gate",
		// read programmatically from the feature gates parsed out of the stanza-less config.
		wantEnabled *bool
	}{
		{
			name: "no featureGates stanza follows the registered default",
			configText: `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
`,
			wantEnabled: nil,
		},
		{
			name: "flowControl gate enabled wires the flow control admission controller",
			configText: `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl
`,
			wantEnabled: boolPtr(true),
		},
		{
			name: "flowControl=false restores the legacy admission controller",
			configText: `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl=false
`,
			wantEnabled: boolPtr(false),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			opts := runserver.NewOptions()
			opts.ConfigText = tc.configText
			opts.PoolName = "test-pool"

			r := NewRunner()
			rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err)

			wantEnabled := r.featureGates[flowcontrol.FeatureGate] // Registered default.
			if tc.wantEnabled != nil {
				wantEnabled = *tc.wantEnabled
				require.Equal(t, wantEnabled, r.featureGates[flowcontrol.FeatureGate],
					"the loader should honor the explicit featureGates stanza")
			}

			ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
			eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds)
			require.NoError(t, err)

			endpointCandidates := contracts.EndpointCandidates(
				requestcontrol.NewDatastoreEndpointCandidates(ds))
			_, admissionController, controlPlane :=
				r.initAdmissionControl(ctx, opts, eppConfig, endpointCandidates)

			if wantEnabled {
				require.IsType(t, &requestcontrol.FlowControlAdmissionController{}, admissionController)
				require.NotNil(t, eppConfig.FlowControlConfig,
					"the loader should build a flow control config when the gate is on")
				require.NotNil(t, controlPlane,
					"the flow registry should be exposed as the priority band control plane")
			} else {
				require.IsType(t, &requestcontrol.LegacyAdmissionController{}, admissionController)
				require.Nil(t, eppConfig.FlowControlConfig,
					"the loader should not build a flow control config when the gate is off")
				require.Nil(t, controlPlane,
					"no priority band control plane should exist when the gate is off")
			}
		})
	}
}
