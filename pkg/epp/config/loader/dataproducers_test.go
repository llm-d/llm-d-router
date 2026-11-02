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

package loader

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

type mockDataProducer struct{ mockPlugin }

func (m *mockDataProducer) Produces() map[fwkplugin.DataKey]any { return nil }

func (m *mockDataProducer) Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return nil
}

func TestLoadRawConfigDataProducers(t *testing.T) {
	cfg, _, err := LoadRawConfig([]byte(`
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- name: tokenizer
  type: test-plugin
schedulingProfiles:
- name: default
  plugins: []
requestHandler:
  dataProducers:
  - pluginRef: tokenizer
  - pluginRef: other
`), logging.NewTestLogger())
	require.NoError(t, err)
	require.NotNil(t, cfg.RequestHandler)
	assert.Equal(t, []configapiv1.DataProducerConfig{{PluginRef: "tokenizer"}, {PluginRef: "other"}}, cfg.RequestHandler.DataProducers)
	assert.Contains(t, cfg.RequestHandler.String(), "DataProducers: [{PluginRef: tokenizer}, {PluginRef: other}]")
}

func TestValidateDataProducers(t *testing.T) {
	plugins := []configapiv1.PluginSpec{{Name: "tokenizer", Type: "token-producer"}, {Name: "prefix", Type: "approx-prefix-cache-producer"}}
	tests := []struct {
		name    string
		entries []configapiv1.DataProducerConfig
		wantErr string
	}{
		{name: "no entries"},
		{
			name:    "listed producers",
			entries: []configapiv1.DataProducerConfig{{PluginRef: "tokenizer"}, {PluginRef: "prefix"}},
		},
		{
			name:    "undefined plugin",
			entries: []configapiv1.DataProducerConfig{{PluginRef: "missing"}},
			wantErr: "data producer references undefined plugin 'missing'",
		},
		{
			name:    "duplicate entry",
			entries: []configapiv1.DataProducerConfig{{PluginRef: "tokenizer"}, {PluginRef: "tokenizer"}},
			wantErr: "data producer 'tokenizer' is listed more than once",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &configapiv1.EndpointPickerConfig{
				Plugins:        plugins,
				RequestHandler: &configapiv1.RequestHandlerConfig{DataProducers: tt.entries},
			}
			err := validateDataProducers(cfg)
			if tt.wantErr == "" {
				assert.NoError(t, err)
				return
			}
			assert.ErrorContains(t, err, tt.wantErr)
		})
	}
}

func TestBuildDataProducerPhases(t *testing.T) {
	handle := testutils.NewTestHandle(context.Background())
	for _, name := range []string{"tokenizer", "prefix", "unlisted"} {
		handle.AddPlugin(name, &mockDataProducer{mockPlugin{t: fwkplugin.TypedName{Name: name, Type: "producer"}}})
	}
	handle.AddPlugin("scorer", &mockScorer{mockPlugin{t: fwkplugin.TypedName{Name: "scorer", Type: testScorerType}}})

	phases, err := buildDataProducerPhases([]configapiv1.DataProducerConfig{{PluginRef: "tokenizer"}, {PluginRef: "prefix"}}, handle)
	require.NoError(t, err)
	assert.Equal(t, map[string]fwkrc.DataProducerPhase{
		"tokenizer": fwkrc.DataProducerPhaseRequestHandling,
		"prefix":    fwkrc.DataProducerPhaseRequestHandling,
	}, phases)

	_, err = buildDataProducerPhases([]configapiv1.DataProducerConfig{{PluginRef: "scorer"}}, handle)
	assert.ErrorContains(t, err, `the plugin "scorer" is not a data producer plugin`)

	_, err = buildDataProducerPhases([]configapiv1.DataProducerConfig{{PluginRef: "missing"}}, handle)
	assert.ErrorContains(t, err, `the configured data producer "missing" is not loaded`)
}
