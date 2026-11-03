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

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	srckvevents "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/kvevents"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
	"github.com/stretchr/testify/require"
)

type sessionRequestProducer struct{ name string }

func (p *sessionRequestProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "test-session-cache-request", Name: p.name}
}
func (p *sessionRequestProducer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName(p.name): attrsession.SessionCacheRequest{}}
}
func (*sessionRequestProducer) Produce(context.Context, *scheduling.InferenceRequest, []scheduling.Endpoint) error {
	return nil
}

func TestSessionPrefixCacheConfiguration(t *testing.T) {
	plugin.Register("test-session-cache-request", plugin.StabilityAlpha, func(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
		return &sessionRequestProducer{name: name}, nil
	})
	for name, tc := range map[string]struct {
		sourceRef  string
		configured bool
		wantErr    string
	}{
		"missing-session-producer":        {sourceRef: "engine-events", wantErr: "external-sessions"},
		"missing-source-ref":              {configured: true, wantErr: "kvEventsSourceRef is required"},
		"attribute-only-session-producer": {sourceRef: "engine-events", configured: true},
	} {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			opts := runserver.NewOptions()
			opts.AllowExperimentalPlugins = true
			opts.ConfigText = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: session-prefix-cache-producer
  name: session-cache
  parameters:
    sessionCacheRequestProducerName: external-sessions
    cacheNamespace: model-v1
    kvEventsSourceRef: ` + tc.sourceRef + `
- type: kv-events-source
  name: engine-events
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: session-cache
`
			if tc.configured {
				opts.ConfigText += "- type: test-session-cache-request\n  name: external-sessions\n"
			}
			r := NewRunner()
			raw, err := r.parseConfigurationPhaseOne(ctx, opts)
			if err == nil {
				ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
				_, err = r.parseConfigurationPhaseTwo(ctx, raw, ds, opts.RefreshMetricsInterval)
			}
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			source, ok := r.PluginHandle.Plugin("engine-events").(*srckvevents.Source)
			require.True(t, ok, "plugin engine-events must be a %s", srckvevents.PluginType)
			state, err := source.DumpState()
			require.NoError(t, err)
			require.Contains(t, string(state), `"consumers":["session-cache/session-prefix-cache-producer"]`)
			for _, p := range r.PluginHandle.GetAllPlugins() {
				require.NotContains(t, []string{"token-producer", "precise-prefix-cache-producer", "approx-prefix-cache-producer"}, p.TypedName().Type,
					"session lookup must not instantiate a token-based cache path")
			}
		})
	}
}
