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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/discovery/k8speer"
)

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
	mgr := &addOnlyManager{}

	require.NoError(t, r.setupPeerDiscovery(mgr, peersConfigRef("peer-disc"), "pool-ns"))
	assert.Len(t, mgr.added, 1)
}

func TestSetupPeerDiscovery_K8sPeerOutsidePoolNamespace(t *testing.T) {
	r := &Runner{PluginHandle: newHandleWithPlugin(t, "peer-disc", newK8sPeerPlugin(t, "epp-ns"))}
	mgr := &addOnlyManager{}

	err := r.setupPeerDiscovery(mgr, peersConfigRef("peer-disc"), "pool-ns")
	require.Error(t, err)
	assert.ErrorContains(t, err, "peer-disc")
	assert.ErrorContains(t, err, "epp-ns")
	assert.ErrorContains(t, err, "pool-ns")
	assert.Empty(t, mgr.added)
}

func TestSetupPeerDiscovery_OtherPluginIgnoresPoolNamespace(t *testing.T) {
	r := &Runner{PluginHandle: newHandleWithPlugin(t, "peer-disc", newStubPeerDiscovery())}
	mgr := &addOnlyManager{}

	require.NoError(t, r.setupPeerDiscovery(mgr, peersConfigRef("peer-disc"), "pool-ns"))
	assert.Len(t, mgr.added, 1)
}
