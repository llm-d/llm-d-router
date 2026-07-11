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
