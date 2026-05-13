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
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	configapi "github.com/llm-d/llm-d-inference-scheduler/apix/config/v1alpha1"
	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	discoveryfile "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/discovery/file"
)

func init() {
	fwkplugin.Register(discoveryfile.PluginType, discoveryfile.Factory)
}

func makeConfig(pluginName, pluginType string, params json.RawMessage, discoveryRef string) *configapi.EndpointPickerConfig {
	return &configapi.EndpointPickerConfig{
		Plugins: []configapi.PluginSpec{
			{Name: pluginName, Type: pluginType, Parameters: params},
		},
		DataLayer: &configapi.DataLayerConfig{
			Discovery: &configapi.DiscoveryConfig{PluginRef: discoveryRef},
		},
	}
}

func TestResolveDiscovery_FileDiscovery(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "endpoints-*.yaml")
	require.NoError(t, err)
	_, _ = f.WriteString("endpoints: []\n")
	require.NoError(t, f.Close())

	params, _ := json.Marshal(map[string]any{"path": f.Name()})
	cfg := makeConfig("my-disc", discoveryfile.PluginType, params, "my-disc")

	r := &Runner{}
	disc, err := r.resolveDiscovery(cfg)
	require.NoError(t, err)
	assert.IsType(t, &discoveryfile.FileDiscovery{}, disc)
	assert.Equal(t, discoveryfile.PluginType, disc.TypedName().Type)
	assert.Equal(t, "my-disc", disc.TypedName().Name)
}

func TestResolveDiscovery_PluginRefNotFound(t *testing.T) {
	cfg := &configapi.EndpointPickerConfig{
		Plugins: []configapi.PluginSpec{
			{Name: "other", Type: discoveryfile.PluginType},
		},
		DataLayer: &configapi.DataLayerConfig{
			Discovery: &configapi.DiscoveryConfig{PluginRef: "nonexistent"},
		},
	}
	r := &Runner{}
	_, err := r.resolveDiscovery(cfg)
	assert.ErrorContains(t, err, "no plugin found with name")
}

func TestResolveDiscovery_UnknownType(t *testing.T) {
	cfg := makeConfig("my-disc", "unknown-type", nil, "my-disc")
	r := &Runner{}
	_, err := r.resolveDiscovery(cfg)
	assert.ErrorContains(t, err, "unknown plugin type")
}

func TestResolveDiscovery_NotEndpointDiscovery(t *testing.T) {
	const notDiscType = "not-a-discovery"
	fwkplugin.Register(notDiscType, func(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return &notDiscoveryPlugin{}, nil
	})
	cfg := makeConfig("not-disc", notDiscType, nil, "not-disc")
	r := &Runner{}
	_, err := r.resolveDiscovery(cfg)
	assert.ErrorContains(t, err, "does not implement EndpointDiscovery")
}

type notDiscoveryPlugin struct{}

func (p *notDiscoveryPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "not-a-discovery", Name: "not-a-discovery"}
}

var _ fwkplugin.Plugin = (*notDiscoveryPlugin)(nil)
