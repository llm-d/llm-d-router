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
	"bytes"
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	discoveryfile "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/discovery/file"
)

func newHandleWithPlugin(t *testing.T, name string, p fwkplugin.Plugin) fwkplugin.Handle {
	t.Helper()
	h := fwkplugin.NewEppHandle(context.Background(), nil)
	h.AddPlugin(name, p)
	return h
}

func discoveryConfigRef(ref string) *configapi.EndpointPickerConfig {
	return &configapi.EndpointPickerConfig{
		DataLayer: &configapi.DataLayerConfig{
			Discovery: &configapi.DiscoveryConfig{PluginRef: ref},
		},
	}
}

func TestResolveDiscovery_FileDiscovery(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "endpoints-*.yaml")
	require.NoError(t, err)
	_, _ = f.WriteString("endpoints: []\n")
	require.NoError(t, f.Close())

	params, _ := json.Marshal(map[string]any{"path": f.Name()})
	p, err := discoveryfile.Factory("my-disc", json.NewDecoder(bytes.NewReader(params)), nil)
	require.NoError(t, err)

	r := &Runner{PluginHandle: newHandleWithPlugin(t, "my-disc", p)}
	disc, err := r.resolveDiscovery(discoveryConfigRef("my-disc"))
	require.NoError(t, err)
	assert.IsType(t, &discoveryfile.FileDiscovery{}, disc)
	assert.Equal(t, discoveryfile.PluginType, disc.TypedName().Type)
	assert.Equal(t, "my-disc", disc.TypedName().Name)
}

func TestResolveDiscovery_PluginRefNotFound(t *testing.T) {
	r := &Runner{PluginHandle: fwkplugin.NewEppHandle(context.Background(), nil)}
	_, err := r.resolveDiscovery(discoveryConfigRef("nonexistent"))
	assert.ErrorContains(t, err, "nonexistent")
}

func TestResolveDiscovery_NotEndpointDiscovery(t *testing.T) {
	r := &Runner{PluginHandle: newHandleWithPlugin(t, "not-disc", &notDiscoveryPlugin{})}
	_, err := r.resolveDiscovery(discoveryConfigRef("not-disc"))
	assert.ErrorContains(t, err, "not-disc")
	assert.ErrorContains(t, err, "EndpointDiscovery")
}

type notDiscoveryPlugin struct{}

func (p *notDiscoveryPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "not-a-discovery", Name: "not-a-discovery"}
}

var _ fwkplugin.Plugin = (*notDiscoveryPlugin)(nil)
