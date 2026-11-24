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
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// unregisterAgentIdentityMetrics removes the plugin's collectors from the
// process-wide registry the runner uses, so the next test can enable metrics
// again (a second registration is rejected by design).
func unregisterAgentIdentityMetrics(t *testing.T) {
	t.Helper()
	t.Cleanup(func() {
		ctrlmetrics.Registry.Unregister(prometheus.NewCounterVec(prometheus.CounterOpts{Name: "llm_d_epp_agent_identity_requests_total"}, []string{"harness", "session_header"}))
		ctrlmetrics.Registry.Unregister(prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_d_epp_agent_identity_active_sessions"}, []string{"harness"}))
	})
}

// TestAgentIdentityMetricsConfigInstantiates loads agent-identity in shadow
// mode through the production config path and checks that it declares no
// produced attribute and that its counter is on the registry the EPP serves.
func TestAgentIdentityMetricsConfigInstantiates(t *testing.T) {
	unregisterAgentIdentityMetrics(t)
	configText := `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: agent-identity
  name: agent-identity
  parameters:
    publishIdentity: false
    metrics: true
`

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	opts := runserver.NewOptions()
	opts.ConfigText = configText
	opts.PoolName = "agent-identity-config-pool"
	opts.AllowExperimentalPlugins = true // agent-identity is Alpha

	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)

	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
	require.NoError(t, err)

	p := r.PluginHandle.Plugin("agent-identity")
	require.NotNil(t, p)
	producer, ok := p.(fwkplugin.ProducerPlugin)
	require.True(t, ok)
	require.Empty(t, producer.Produces(), "publishIdentity: false must declare no produced attribute")

	rh, ok := p.(fwkrh.RequestHeaderProcessor)
	require.True(t, ok)
	req := &fwksched.InferenceRequest{
		RequestID: "agent-identity-config-test",
		Headers:   map[string]string{agentidentity.ClaudeCodeSessionHeader: "session-1"},
	}
	require.NoError(t, rh.RequestHeader(ctx, req))

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	present := map[string]bool{}
	for _, mf := range families {
		present[mf.GetName()] = true
	}
	require.True(t, present["llm_d_epp_agent_identity_requests_total"], "requests counter not on the served registry")

	// The gauge has no series until its first refresh, so check registration
	// directly: registering the same name again must collide.
	err = ctrlmetrics.Registry.Register(prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "llm_d_epp_agent_identity_active_sessions",
		Help: "probe",
	}, []string{"harness"}))
	require.ErrorContains(t, err, "previously registered", "active_sessions gauge not on the served registry")
}

// TestAgentIdentityPublishIdentityFalseFailsConsumerStartup proves the
// Produces() decision for publishIdentity: false fails loudly: a config whose
// session-state-producer requires the agent-identity attribute, with no other
// producer of it, is rejected while the configuration loads rather than
// silently giving the consumer nothing to read.
func TestAgentIdentityPublishIdentityFalseFailsConsumerStartup(t *testing.T) {
	unregisterAgentIdentityMetrics(t)
	configText := `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: agent-identity
  name: agent-identity
  parameters:
    publishIdentity: false
    metrics: true
- type: session-state-producer
  name: session-state-producer
`

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	opts := runserver.NewOptions()
	opts.ConfigText = configText
	opts.PoolName = "agent-identity-consumer-pool"
	opts.AllowExperimentalPlugins = true

	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)

	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
	require.Error(t, err, "a consumer requiring agent-identity must fail startup when the only producer is silent")
	require.Contains(t, err.Error(), "no default producer")
	require.Contains(t, err.Error(), agentidentity.AgentIdentityKey.String())
	require.Contains(t, err.Error(), sessionstate.SessionStateProducerType)
}

// TestAgentIdentityDefaultsConfigInstantiates proves a defaults config — no
// parameters — still instantiates and keeps declaring the agent-identity
// attribute.
func TestAgentIdentityDefaultsConfigInstantiates(t *testing.T) {
	configText := `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: agent-identity
  name: agent-identity
`

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	opts := runserver.NewOptions()
	opts.ConfigText = configText
	opts.PoolName = "agent-identity-defaults-pool"
	opts.AllowExperimentalPlugins = true

	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)

	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
	require.NoError(t, err)

	p := r.PluginHandle.Plugin("agent-identity")
	require.NotNil(t, p)
	producer, ok := p.(fwkplugin.ProducerPlugin)
	require.True(t, ok)
	require.Contains(t, producer.Produces(), agentidentity.AgentIdentityKey,
		"the defaults config must keep producing the agent-identity attribute")
}
