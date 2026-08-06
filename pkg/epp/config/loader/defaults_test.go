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
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	extractormetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	sourcemodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/models"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// dataLayerDefaultPlugins returns an allPlugins map with mock stubs for every default data layer
// plugin. Providing them prevents ensureDataLayer from calling registerDefaultPlugin (which needs
// the global factory registry). The function still injects the DataLayer.Sources entries.
func dataLayerDefaultPlugins() map[string]fwkplugin.Plugin {
	plugins := map[string]fwkplugin.Plugin{}
	for _, name := range []string{
		sourcemetrics.MetricsDataSourceType,
		extractormetrics.MetricsExtractorType,
		sourcemodels.ModelsDataSourceType,
		attrmodels.ModelsExtractorType,
	} {
		plugins[name] = &mockPlugin{t: fwkplugin.TypedName{Type: name, Name: name}}
	}
	return plugins
}

func TestEnsureDataLayer(t *testing.T) {
	// Not parallel: shares helpers with configloader_test.go that depend on global state.

	// sourceRefs returns the PluginRef of every configured source, in order.
	sourceRefs := func(cfg *configapi.EndpointPickerConfig) []string {
		refs := make([]string, 0, len(cfg.DataLayer.Sources))
		for _, source := range cfg.DataLayer.Sources {
			refs = append(refs, source.PluginRef)
		}
		return refs
	}

	t.Run("nil DataLayer injects metrics and models defaults", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins())

		require.NoError(t, err)
		require.NotNil(t, cfg.DataLayer)
		require.Len(t, cfg.DataLayer.Sources, 2)
		require.Equal(t, sourcemetrics.MetricsDataSourceType, cfg.DataLayer.Sources[0].PluginRef)
		require.Len(t, cfg.DataLayer.Sources[0].Extractors, 1)
		require.Equal(t, extractormetrics.MetricsExtractorType, cfg.DataLayer.Sources[0].Extractors[0].PluginRef)
		require.Equal(t, sourcemodels.ModelsDataSourceType, cfg.DataLayer.Sources[1].PluginRef)
		require.Len(t, cfg.DataLayer.Sources[1].Extractors, 1)
		require.Equal(t, attrmodels.ModelsExtractorType, cfg.DataLayer.Sources[1].Extractors[0].PluginRef)
	})

	t.Run("empty DataLayer {} injects defaults (regression: was no-op)", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{
			DataLayer: &configapi.DataLayerConfig{},
		}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins())

		require.NoError(t, err)
		require.Equal(t, []string{sourcemetrics.MetricsDataSourceType, sourcemodels.ModelsDataSourceType}, sourceRefs(cfg))
	})

	t.Run("unrelated source gets defaults injected too (additive)", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{
			DataLayer: &configapi.DataLayerConfig{
				Sources: []configapi.DataLayerSource{
					{PluginRef: "k8s-notification-source"},
				},
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins())

		require.NoError(t, err)
		require.Equal(t, []string{
			"k8s-notification-source",
			sourcemetrics.MetricsDataSourceType,
			sourcemodels.ModelsDataSourceType,
		}, sourceRefs(cfg))
	})

	t.Run("existing metrics-data-source is not double-injected", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{
			DataLayer: &configapi.DataLayerConfig{
				Sources: []configapi.DataLayerSource{
					{PluginRef: sourcemetrics.MetricsDataSourceType},
				},
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins())

		require.NoError(t, err)
		require.Equal(t, []string{sourcemetrics.MetricsDataSourceType, sourcemodels.ModelsDataSourceType}, sourceRefs(cfg),
			"metrics not duplicated, models still injected")
	})

	t.Run("existing models-data-source is not double-injected", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{
			DataLayer: &configapi.DataLayerConfig{
				Sources: []configapi.DataLayerSource{
					{PluginRef: sourcemodels.ModelsDataSourceType},
				},
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins())

		require.NoError(t, err)
		require.Equal(t, []string{sourcemodels.ModelsDataSourceType, sourcemetrics.MetricsDataSourceType}, sourceRefs(cfg),
			"models not duplicated, metrics still injected")
	})

	t.Run("injectDefaults: false suppresses injection", func(t *testing.T) {
		cfg := &configapi.EndpointPickerConfig{
			DataLayer: &configapi.DataLayerConfig{
				InjectDefaults: ptr.To(false),
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins())

		require.NoError(t, err)
		require.Empty(t, cfg.DataLayer.Sources)
	})

}
