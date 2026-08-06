/*
Copyright 2025 The llm-d Authors.

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

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	extractormetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	sourcemodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/models"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// mockFilterDetector implements both SaturationDetector and Filter, like the real utilization-detector.
type mockFilterDetector struct{ mockPlugin }

var (
	_ fwksched.Filter = &mockFilterDetector{}
)

func (m *mockFilterDetector) Saturation(_ context.Context, _ []fwkdl.Endpoint) float64 { return 0 }
func (m *mockFilterDetector) Filter(_ context.Context, _ *fwksched.InferenceRequest, eps []fwksched.Endpoint) []fwksched.Endpoint {
	return eps
}

// dataLayerDefaultPlugins returns an allPlugins map with mock stubs for every default data layer
// plugin. Providing them prevents ensureDataLayer from calling registerDefaultPlugin (which needs
// the global factory registry). The function still injects the DataLayer.Sources entries.
func dataLayerDefaultPlugins(handle fwkplugin.Handle) map[string]fwkplugin.Plugin {
	for _, name := range []string{
		sourcemetrics.MetricsDataSourceType,
		extractormetrics.MetricsExtractorType,
		sourcemodels.ModelsDataSourceType,
		attrmodels.ModelsExtractorType,
	} {
		handle.AddPlugin(name, &mockPlugin{t: fwkplugin.TypedName{Type: name, Name: name}})
	}
	return handle.GetAllPluginsWithNames()
}

func TestEnsureDataLayer(t *testing.T) {
	// Not parallel: shares helpers with configloader_test.go that depend on global state.

	// sourceRefs returns the PluginRef of every configured source, in order.
	sourceRefs := func(cfg *configapiv1.EndpointPickerConfig) []string {
		refs := make([]string, 0, len(cfg.DataLayer.Sources))
		for _, source := range cfg.DataLayer.Sources {
			refs = append(refs, source.PluginRef)
		}
		return refs
	}

	t.Run("nil DataLayer injects metrics and models defaults", func(t *testing.T) {
		cfg := &configapiv1.EndpointPickerConfig{}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins(handle))

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
		cfg := &configapiv1.EndpointPickerConfig{
			DataLayer: &configapiv1.DataLayerConfig{},
		}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins(handle))

		require.NoError(t, err)
		require.Equal(t, []string{sourcemetrics.MetricsDataSourceType, sourcemodels.ModelsDataSourceType}, sourceRefs(cfg))
	})

	t.Run("unrelated source gets defaults injected too (additive)", func(t *testing.T) {
		cfg := &configapiv1.EndpointPickerConfig{
			DataLayer: &configapiv1.DataLayerConfig{
				Sources: []configapiv1.DataLayerSource{
					{PluginRef: "k8s-notification-source"},
				},
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins(handle))

		require.NoError(t, err)
		require.Equal(t, []string{
			"k8s-notification-source",
			sourcemetrics.MetricsDataSourceType,
			sourcemodels.ModelsDataSourceType,
		}, sourceRefs(cfg))
	})

	t.Run("source of another type does not suppress injection", func(t *testing.T) {
		cfg := &configapiv1.EndpointPickerConfig{
			DataLayer: &configapiv1.DataLayerConfig{
				Sources: []configapiv1.DataLayerSource{
					{PluginRef: "dcgmSource"},
				},
			},
		}
		handle := testutils.NewTestHandle(context.Background())
		allPlugins := dataLayerDefaultPlugins(handle)
		handle.AddPlugin("dcgmSource", &mockPlugin{t: fwkplugin.TypedName{Type: "dcgm-data-source", Name: "dcgmSource"}})

		err := ensureDataLayer(cfg, handle, allPlugins)

		require.NoError(t, err)
		require.Equal(t, []string{
			"dcgmSource",
			sourcemetrics.MetricsDataSourceType,
			sourcemodels.ModelsDataSourceType,
		}, sourceRefs(cfg))
	})

	t.Run("existing metrics-data-source is not double-injected", func(t *testing.T) {
		cfg := &configapiv1.EndpointPickerConfig{
			DataLayer: &configapiv1.DataLayerConfig{
				Sources: []configapiv1.DataLayerSource{
					{PluginRef: sourcemetrics.MetricsDataSourceType},
				},
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins(handle))

		require.NoError(t, err)
		require.Equal(t, []string{sourcemetrics.MetricsDataSourceType, sourcemodels.ModelsDataSourceType}, sourceRefs(cfg),
			"metrics not duplicated, models still injected")
	})

	t.Run("existing models-data-source is not double-injected", func(t *testing.T) {
		cfg := &configapiv1.EndpointPickerConfig{
			DataLayer: &configapiv1.DataLayerConfig{
				Sources: []configapiv1.DataLayerSource{
					{PluginRef: sourcemodels.ModelsDataSourceType},
				},
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins(handle))

		require.NoError(t, err)
		require.Equal(t, []string{sourcemodels.ModelsDataSourceType, sourcemetrics.MetricsDataSourceType}, sourceRefs(cfg),
			"models not duplicated, metrics still injected")
	})

	t.Run("metrics source under a custom instance name is not double-injected", func(t *testing.T) {
		cfg := &configapiv1.EndpointPickerConfig{
			DataLayer: &configapiv1.DataLayerConfig{
				Sources: []configapiv1.DataLayerSource{
					{
						PluginRef:  "metricsSource",
						Extractors: []configapiv1.DataLayerExtractor{{PluginRef: "customMetricsExtractor"}},
					},
				},
			},
		}
		handle := testutils.NewTestHandle(context.Background())
		handle.AddPlugin("metricsSource", &mockPlugin{t: fwkplugin.TypedName{Type: sourcemetrics.MetricsDataSourceType, Name: "metricsSource"}})
		handle.AddPlugin("customMetricsExtractor", &mockPlugin{t: fwkplugin.TypedName{Type: extractormetrics.MetricsExtractorType, Name: "customMetricsExtractor"}})
		allPlugins := dataLayerDefaultPlugins(handle)

		err := ensureDataLayer(cfg, handle, allPlugins)

		require.NoError(t, err)
		require.Len(t, cfg.DataLayer.Sources, 2, "no duplicate metrics source, models still injected")
		require.Equal(t, "metricsSource", cfg.DataLayer.Sources[0].PluginRef)
		require.Len(t, cfg.DataLayer.Sources[0].Extractors, 1, "no duplicate metrics extractor")
		require.Equal(t, "customMetricsExtractor", cfg.DataLayer.Sources[0].Extractors[0].PluginRef)
		require.Equal(t, sourcemodels.ModelsDataSourceType, cfg.DataLayer.Sources[1].PluginRef)
	})

	t.Run("injectDefaults: false suppresses injection", func(t *testing.T) {
		cfg := &configapiv1.EndpointPickerConfig{
			DataLayer: &configapiv1.DataLayerConfig{
				InjectDefaults: ptr.To(false),
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		err := ensureDataLayer(cfg, handle, dataLayerDefaultPlugins(handle))

		require.NoError(t, err)
		require.Empty(t, cfg.DataLayer.Sources)
	})

}

func TestEnsureSaturationDetector_InjectsFilter(t *testing.T) {
	t.Run("detector implementing Filter is injected into profiles", func(t *testing.T) {
		w := 2.0
		cfg := &configapiv1.EndpointPickerConfig{
			SchedulingProfiles: []configapiv1.SchedulingProfile{
				{Name: "default", Plugins: []configapiv1.SchedulingPlugin{{PluginRef: "scorer", Weight: &w}}},
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		detectorName := "my-detector"
		detector := &mockFilterDetector{mockPlugin{t: fwkplugin.TypedName{Type: detectorName, Name: detectorName}}}
		handle.AddPlugin(detectorName, detector)

		allPlugins := handle.GetAllPluginsWithNames()
		cfg.FlowControl = &configapiv1.FlowControlConfig{
			SaturationDetector: &configapiv1.SaturationDetectorConfig{PluginRef: detectorName},
		}

		err := ensureSaturationDetector(cfg, handle, allPlugins)

		require.NoError(t, err)
		require.Len(t, cfg.SchedulingProfiles[0].Plugins, 2)
		require.Equal(t, detectorName, cfg.SchedulingProfiles[0].Plugins[1].PluginRef)
	})

	t.Run("detector not implementing Filter is not injected", func(t *testing.T) {
		w := 2.0
		cfg := &configapiv1.EndpointPickerConfig{
			SchedulingProfiles: []configapiv1.SchedulingProfile{
				{Name: "default", Plugins: []configapiv1.SchedulingPlugin{{PluginRef: "scorer", Weight: &w}}},
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		detectorName := "plain-detector"
		detector := &mockSaturationDetector{mockPlugin{t: fwkplugin.TypedName{Type: detectorName, Name: detectorName}}}
		handle.AddPlugin(detectorName, detector)

		allPlugins := handle.GetAllPluginsWithNames()
		cfg.FlowControl = &configapiv1.FlowControlConfig{
			SaturationDetector: &configapiv1.SaturationDetectorConfig{PluginRef: detectorName},
		}

		err := ensureSaturationDetector(cfg, handle, allPlugins)

		require.NoError(t, err)
		require.Len(t, cfg.SchedulingProfiles[0].Plugins, 1, "non-filter detector should not be injected")
	})

	t.Run("detector already in profile is not duplicated", func(t *testing.T) {
		detectorName := "my-detector"
		cfg := &configapiv1.EndpointPickerConfig{
			SchedulingProfiles: []configapiv1.SchedulingProfile{
				{Name: "default", Plugins: []configapiv1.SchedulingPlugin{
					{PluginRef: detectorName},
					{PluginRef: "picker"},
				}},
			},
		}
		handle := testutils.NewTestHandle(context.Background())

		detector := &mockFilterDetector{mockPlugin{t: fwkplugin.TypedName{Type: detectorName, Name: detectorName}}}
		handle.AddPlugin(detectorName, detector)

		allPlugins := handle.GetAllPluginsWithNames()
		cfg.FlowControl = &configapiv1.FlowControlConfig{
			SaturationDetector: &configapiv1.SaturationDetectorConfig{PluginRef: detectorName},
		}

		err := ensureSaturationDetector(cfg, handle, allPlugins)

		require.NoError(t, err)
		require.Len(t, cfg.SchedulingProfiles[0].Plugins, 2, "already present, no duplicate")
	})
}
