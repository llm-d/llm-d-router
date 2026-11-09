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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/discovery/k8speer"
)

// recordingManager records runnables so tests can run setupPeerDiscovery
// without a cluster.
type recordingManager struct {
	ctrl.Manager
	added []manager.Runnable
}

func (m *recordingManager) Add(r manager.Runnable) error {
	m.added = append(m.added, r)
	return nil
}

type fakePeerDiscovery struct {
	notDiscoveryPlugin
}

func (p *fakePeerDiscovery) Start(context.Context, fwkdl.PeerNotifier) error { return nil }

func (p *fakePeerDiscovery) Ready() <-chan struct{} { return nil }

var _ fwkdl.PeerDiscovery = (*fakePeerDiscovery)(nil)

func peerConfigRef(ref string) *configapiv1.EndpointPickerConfig {
	return &configapiv1.EndpointPickerConfig{
		DataLayer: &configapiv1.DataLayerConfig{
			Discovery: &configapiv1.DiscoveryConfig{
				Peers: &configapiv1.PeerDiscoveryConfig{PluginRef: ref},
			},
		},
	}
}

func newK8sPeerPlugin(t *testing.T, namespace string) fwkplugin.Plugin {
	t.Helper()
	t.Setenv("NAMESPACE", namespace)
	params, err := json.Marshal(map[string]string{
		"selector":  "app=epp",
		"port":      "9002",
		"namespace": namespace,
	})
	require.NoError(t, err)
	p, err := k8speer.Factory("peer-disc", fwkplugin.StrictDecoder(params), nil)
	require.NoError(t, err)
	return p
}

func TestSetupPeerDiscovery_K8sPeerInPoolNamespace(t *testing.T) {
	r := &Runner{PluginHandle: newHandleWithPlugin(t, "peer-disc", newK8sPeerPlugin(t, "pool-ns"))}
	mgr := &recordingManager{}

	require.NoError(t, r.setupPeerDiscovery(mgr, peerConfigRef("peer-disc"), "pool-ns"))
	assert.Len(t, mgr.added, 1)
}

func TestSetupPeerDiscovery_K8sPeerOutsidePoolNamespace(t *testing.T) {
	r := &Runner{PluginHandle: newHandleWithPlugin(t, "peer-disc", newK8sPeerPlugin(t, "epp-ns"))}
	mgr := &recordingManager{}

	err := r.setupPeerDiscovery(mgr, peerConfigRef("peer-disc"), "pool-ns")
	require.Error(t, err)
	assert.ErrorContains(t, err, "peer-disc")
	assert.ErrorContains(t, err, "epp-ns")
	assert.ErrorContains(t, err, "pool-ns")
	assert.Empty(t, mgr.added)
}

func TestSetupPeerDiscovery_OtherPluginIgnoresPoolNamespace(t *testing.T) {
	r := &Runner{PluginHandle: newHandleWithPlugin(t, "peer-disc", &fakePeerDiscovery{})}
	mgr := &recordingManager{}

	require.NoError(t, r.setupPeerDiscovery(mgr, peerConfigRef("peer-disc"), "pool-ns"))
	assert.Len(t, mgr.added, 1)
}
