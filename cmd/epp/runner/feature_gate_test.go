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
	"testing"

	"github.com/stretchr/testify/require"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol"
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

// TestApplyDeprecatedEnvFeatureGate verifies that the deprecated
// ENABLE_EXPERIMENTAL_FLOW_CONTROL_LAYER env var enables the flowControl gate
// in both rawConfig.FeatureGates (read by InstantiateAndConfigure) and the
// featureGates map populated during parseConfigurationPhaseOne (read by
// initAdmissionControl). A regression here silently strands the EPP on the
// legacy admission controller despite the env var being set.
func TestApplyDeprecatedEnvFeatureGate(t *testing.T) {
	t.Run("env var true enables the gate in both targets", func(t *testing.T) {
		t.Setenv(enableExperimentalFlowControlLayer, "true")

		rawConfig := &configapi.EndpointPickerConfig{}
		featureGates := map[string]bool{flowcontrol.FeatureGate: false}

		applyDeprecatedEnvFeatureGate(enableExperimentalFlowControlLayer, "Flow Control layer", flowcontrol.FeatureGate, rawConfig, featureGates)

		require.Contains(t, rawConfig.FeatureGates, flowcontrol.FeatureGate)
		require.True(t, featureGates[flowcontrol.FeatureGate])
	})

	t.Run("env var unset leaves both targets untouched", func(t *testing.T) {
		rawConfig := &configapi.EndpointPickerConfig{}
		featureGates := map[string]bool{flowcontrol.FeatureGate: false}

		applyDeprecatedEnvFeatureGate(enableExperimentalFlowControlLayer, "Flow Control layer", flowcontrol.FeatureGate, rawConfig, featureGates)

		require.Empty(t, rawConfig.FeatureGates)
		require.False(t, featureGates[flowcontrol.FeatureGate])
	})
}
