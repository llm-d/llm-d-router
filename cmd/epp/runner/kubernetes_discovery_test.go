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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	discoveryendpointslice "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/discovery/endpointslice"
	discoveryfile "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/discovery/file"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

func discoveryConfig(ref string, plugins ...configapiv1.PluginSpec) *configapiv1.EndpointPickerConfig {
	return &configapiv1.EndpointPickerConfig{
		Plugins: plugins,
		DataLayer: &configapiv1.DataLayerConfig{
			Discovery: &configapiv1.DiscoveryConfig{
				Endpoints: &configapiv1.EndpointDiscoveryConfig{PluginRef: ref},
			},
		},
	}
}

func TestKubernetesDiscoveryPlugin(t *testing.T) {
	NewRunner().registerInTreePlugins()

	tests := []struct {
		name string
		cfg  *configapiv1.EndpointPickerConfig
		want bool
	}{
		{name: "nil config", cfg: nil, want: false},
		{name: "no discovery", cfg: &configapiv1.EndpointPickerConfig{}, want: false},
		{
			name: "endpointslice by name",
			cfg:  discoveryConfig("slices", configapiv1.PluginSpec{Name: "slices", Type: discoveryendpointslice.PluginType}),
			want: true,
		},
		{
			name: "endpointslice by type when unnamed",
			cfg:  discoveryConfig(discoveryendpointslice.PluginType, configapiv1.PluginSpec{Type: discoveryendpointslice.PluginType}),
			want: true,
		},
		{
			name: "file discovery",
			cfg:  discoveryConfig("files", configapiv1.PluginSpec{Name: "files", Type: discoveryfile.PluginType}),
			want: false,
		},
		{
			name: "dangling ref",
			cfg:  discoveryConfig("missing", configapiv1.PluginSpec{Name: "slices", Type: discoveryendpointslice.PluginType}),
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, kubernetesDiscoveryPlugin(tt.cfg))
		})
	}
}

func TestValidateKubernetesDiscoveryPluginFlags(t *testing.T) {
	opts := runserver.NewOptions()
	require.NoError(t, validateKubernetesDiscoveryPluginFlags(opts), "pool-name is optional")

	opts.PoolName = "pool"
	require.NoError(t, validateKubernetesDiscoveryPluginFlags(opts))

	selector, err := labels.Parse("app=vllm")
	require.NoError(t, err)
	opts.EndpointSelector = selector
	require.ErrorContains(t, validateKubernetesDiscoveryPluginFlags(opts), "endpoint-selector cannot be set")

	opts.EndpointSelector = nil
	opts.ObjectivesFile = "/etc/epp/objectives.yaml"
	require.ErrorContains(t, validateKubernetesDiscoveryPluginFlags(opts), "objectives-file requires file discovery")
}

func TestDiscoveryPoolGKNN(t *testing.T) {
	t.Setenv("NAMESPACE", "")
	opts := runserver.NewOptions()
	opts.PoolNamespace = "models"
	gknn := discoveryPoolGKNN(opts)
	require.Equal(t, "models", gknn.Namespace)
	require.Equal(t, defaultDiscoveryPoolName, gknn.Name)
	require.Equal(t, schema.GroupKind{}, gknn.GroupKind)

	opts.PoolName = "glm"
	require.Equal(t, "glm", discoveryPoolGKNN(opts).Name)
}

func TestNotificationCacheOverride(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	unstructuredOf := func(gvk schema.GroupVersionKind) *unstructured.Unstructured {
		u := &unstructured.Unstructured{}
		u.SetGroupVersionKind(gvk)
		return u
	}
	sliceGVK := discoveryv1.SchemeGroupVersion.WithKind("EndpointSlice")
	podGVK := corev1.SchemeGroupVersion.WithKind("Pod")
	sel := labels.SelectorFromSet(labels.Set{discoveryv1.LabelServiceName: "model-server"})
	managerPodScope := cache.ByObject{Namespaces: map[string]cache.Config{"pool": {}}}

	opts := &ctrl.Options{
		Scheme: scheme,
		Cache:  cache.Options{ByObject: map[client.Object]cache.ByObject{&corev1.Pod{}: managerPodScope}},
	}
	notificationCacheOverride(map[client.Object]cache.ByObject{
		unstructuredOf(sliceGVK): {Namespaces: map[string]cache.Config{"models": {LabelSelector: sel}}},
		unstructuredOf(podGVK):   {Label: sel},
	})(opts)

	byGVK := map[schema.GroupVersionKind][]cache.ByObject{}
	for obj, scope := range opts.Cache.ByObject {
		gvk := obj.GetObjectKind().GroupVersionKind()
		if _, typed := obj.(*corev1.Pod); typed {
			gvk = podGVK
		}
		byGVK[gvk] = append(byGVK[gvk], scope)
	}
	require.Len(t, byGVK[sliceGVK], 1, "the slice scope is added")
	require.Equal(t, sel.String(), byGVK[sliceGVK][0].Namespaces["models"].LabelSelector.String())
	require.Equal(t, []cache.ByObject{managerPodScope}, byGVK[podGVK], "the manager's Pod scope is kept")

	empty := &ctrl.Options{}
	notificationCacheOverride(nil)(empty)
	require.Nil(t, empty.Cache.ByObject)
}

// readyDiscovery is an EndpointDiscovery whose Ready channel the test closes.
type readyDiscovery struct {
	ready chan struct{}
}

func (d *readyDiscovery) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "test-discovery", Name: "test-discovery"}
}
func (d *readyDiscovery) Start(context.Context, fwkdl.DiscoveryNotifier) error { return nil }
func (d *readyDiscovery) Ready() <-chan struct{}                               { return d.ready }

func TestDiscoveryReadiness(t *testing.T) {
	d := &readyDiscovery{ready: make(chan struct{})}
	checker := discoveryReadiness{d}
	require.ErrorContains(t, checker.CheckReady(), "test-discovery")
	close(d.ready)
	require.NoError(t, checker.CheckReady())
}

// TestSetupRejectsKubernetesDiscoveryWithoutNotificationSource fails startup
// when a plugin registered with RegisterKubernetesDiscovery registers no
// notification source, instead of never becoming ready.
func TestSetupRejectsKubernetesDiscoveryWithoutNotificationSource(t *testing.T) {
	const pluginType = "test-k8s-discovery-without-source"
	fwkplugin.RegisterKubernetesDiscovery(pluginType, fwkplugin.StabilityAlpha,
		func(string, *json.Decoder, fwkplugin.Handle) (fwkplugin.Plugin, error) {
			return &readyDiscovery{ready: make(chan struct{})}, nil
		})
	t.Cleanup(func() {
		delete(fwkplugin.Registry, pluginType)
		delete(fwkplugin.RegistryMetadata, pluginType)
	})

	opts := runserver.NewOptions()
	opts.PoolNamespace = "models"
	opts.ConfigText = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - name: disc
    type: ` + pluginType + `
  - name: random-picker
    type: random-picker
  - name: single-profile-handler
    type: single-profile-handler
  - name: metrics-source
    type: metrics-data-source
  - name: metrics-extractor
    type: core-metrics-extractor
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: random-picker
dataLayer:
  injectDefaults: false
  sources:
    - pluginRef: metrics-source
      extractors:
        - pluginRef: metrics-extractor
  discovery:
    endpoints:
      pluginRef: disc
`
	r := NewRunner()
	// The check fails setup before the manager is created, so the rest config
	// is never dialed.
	_, _, err := r.setup(t.Context(), t.Context(), &rest.Config{Host: "https://127.0.0.1:1"}, opts, nil)
	require.ErrorContains(t, err, "registers no notification source")
}
