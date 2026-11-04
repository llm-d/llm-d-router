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
	"fmt"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/inflightload"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
	"github.com/stretchr/testify/require"
)

func TestMultiGuardProductionConfig(t *testing.T) {
	const multi = `{"guards":[{"role":"prefill","guardPluginName":"p"},{"role":"decode","guardPluginName":"d"}]}`
	for _, tc := range []struct{ name, overflow, decodeType, decodeParams, wantError string }{
		{"screener before guards", multi, "concurrency-detector", `{"maxConcurrency":8,"failOpen":false}`, ""},
		{"token guard", multi, "concurrency-detector", `{"concurrencyMode":"tokens","maxTokenConcurrency":100,"failOpen":false}`, ""},
		{"null mixed fields", `{"guards":null,"guardPluginName":"p"}`, "concurrency-detector", `{}`, "must not be null"},
		{"case mixed fields", `{"Guards":[{"role":"decode","guardPluginName":"d"}],"Role":""}`, "concurrency-detector", `{}`, "cannot mix"},
		{"legacy", `{"guardPluginName":"p"}`, "concurrency-detector", `{"maxConcurrency":8,"failOpen":false}`, ""},
		{"missing guard", strings.ReplaceAll(multi, `"d"`, `"missing"`), "concurrency-detector", `{}`, "missing"},
		{"invalid guard interface", multi, "random-picker", `{}`, "not a scheduling filter"},
		{"unknown plugin type", multi, "missing-type", `{}`, "missing-type"},
		{"fail open", multi, "concurrency-detector", `{"maxConcurrency":8,"failOpen":true}`, "fails open"},
		{"mixed fields", `{"guardPluginName":"p","guards":[{"role":"decode","guardPluginName":"d"}]}`, "concurrency-detector", `{}`, "cannot mix"},
		{"invalid guard parameter type", multi, "concurrency-detector", `{"maxConcurrency":"eight"}`, "unmarshal"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			opts := runserver.NewOptions()
			opts.AllowExperimentalPlugins = true
			opts.PoolName = "multi-guard-test"
			opts.ConfigText = fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- name: screen
  type: disaggregatedset-rollout-screener
  parameters:
    scope: {labelSelector: "app=test"}
    revisionGating:
      mode: sum
      requiredRoles: [prefill, decode]
      overflow: %s
- name: p
  type: concurrency-detector
  parameters: {maxConcurrency: 8, failOpen: false}
- name: d
  type: %s
  parameters: %s
- type: random-picker
- type: prefill-filter
- type: decode-filter
- type: disagg-profile-handler
  parameters:
    profiles: {prefill: prefill, decode: decode}
    deciders: {prefill: always-disagg-pd-decider}
- type: always-disagg-pd-decider
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: p
  - pluginRef: random-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: d
  - pluginRef: random-picker
`, tc.overflow, tc.decodeType, tc.decodeParams)
			r := NewRunner()
			raw, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err)
			ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
			cfg, err := r.parseConfigurationPhaseTwo(ctx, raw, ds, opts.RefreshMetricsInterval)
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
				t.Log(err)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, cfg)
			screen := r.PluginHandle.Plugin("screen").(fwkplugin.ConsumerPlugin)
			_, consumesLoad := screen.Consumes().Required[attrconcurrency.InFlightLoadDataKey]
			require.True(t, consumesLoad)
			phases := datalayer.ResolveDataProducerPhases(r.PluginHandle.GetAllPlugins(), cfg.DataProducerPhases)
			require.Equal(t, fwkrh.DataProducerPhaseRequestHandling, phases[inflightload.InFlightLoadProducerType])
			require.NoError(t, fwkplugin.ValidatePluginStability(r.PluginHandle, opts.AllowExperimentalPlugins))
		})
	}
}
