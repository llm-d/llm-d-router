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

package loader

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	extractormetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	igwtestutils "github.com/llm-d/llm-d-router/test/utils/igw"
)

// metricsPlugins returns an allPlugins map with mock stubs for both default metrics plugins.
// Providing them prevents ensureDataLayer from calling registerDefaultPlugin (which needs the
// global factory registry). The function still injects the DataLayer.Sources entries.
func metricsPlugins() map[string]fwkplugin.Plugin {
	return map[string]fwkplugin.Plugin{
		sourcemetrics.MetricsDataSourceType:   &mockPlugin{t: fwkplugin.TypedName{Type: sourcemetrics.MetricsDataSourceType, Name: sourcemetrics.MetricsDataSourceType}},
		extractormetrics.MetricsExtractorType: &mockPlugin{t: fwkplugin.TypedName{Type: extractormetrics.MetricsExtractorType, Name: extractormetrics.MetricsExtractorType}},
	}
}

func TestEnsureDataLayer(t *testing.T) {
	// Not parallel: shares helpers with configloader_test.go that depend on global state.

	t.Run("nil DataLayer injects metrics defaults", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{}
		handle := igwtestutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, metricsPlugins())

		require.NoError(t, err)
		require.NotNil(t, cfg.DataLayer)
		require.Len(t, cfg.DataLayer.Sources, 1)
		require.Equal(t, sourcemetrics.MetricsDataSourceType, cfg.DataLayer.Sources[0].PluginRef)
		require.Len(t, cfg.DataLayer.Sources[0].Extractors, 1)
		require.Equal(t, extractormetrics.MetricsExtractorType, cfg.DataLayer.Sources[0].Extractors[0].PluginRef)
	})

	t.Run("empty DataLayer {} injects metrics defaults (regression: was no-op)", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{
			DataLayer: &configapi.DataLayerConfig{},
		}
		handle := igwtestutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, metricsPlugins())

		require.NoError(t, err)
		require.Len(t, cfg.DataLayer.Sources, 1)
		require.Equal(t, sourcemetrics.MetricsDataSourceType, cfg.DataLayer.Sources[0].PluginRef)
	})

	t.Run("non-metrics source gets metrics injected too (additive)", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{
			DataLayer: &configapi.DataLayerConfig{
				Sources: []configapi.DataLayerSource{
					{PluginRef: "k8s-notification-source"},
				},
			},
		}
		handle := igwtestutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, metricsPlugins())

		require.NoError(t, err)
		require.Len(t, cfg.DataLayer.Sources, 2)
		refs := []string{cfg.DataLayer.Sources[0].PluginRef, cfg.DataLayer.Sources[1].PluginRef}
		require.Contains(t, refs, "k8s-notification-source")
		require.Contains(t, refs, sourcemetrics.MetricsDataSourceType)
	})

	t.Run("existing metrics-data-source is not double-injected", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{
			DataLayer: &configapi.DataLayerConfig{
				Sources: []configapi.DataLayerSource{
					{PluginRef: sourcemetrics.MetricsDataSourceType},
				},
			},
		}
		handle := igwtestutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, metricsPlugins())

		require.NoError(t, err)
		require.Len(t, cfg.DataLayer.Sources, 1, "no duplicate metrics source")
	})

	t.Run("injectDefaults: false suppresses injection", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{
			DataLayer: &configapi.DataLayerConfig{
				InjectDefaults: ptr.To(false),
			},
		}
		handle := igwtestutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, metricsPlugins())

		require.NoError(t, err)
		require.Empty(t, cfg.DataLayer.Sources)
	})

	t.Run("enableLegacyMetrics feature gate suppresses injection", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{
			FeatureGates: configapi.FeatureGates{datalayer.EnableLegacyMetricsFeatureGate},
		}
		handle := igwtestutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, metricsPlugins())

		require.NoError(t, err)
		require.Nil(t, cfg.DataLayer)
	})
}
