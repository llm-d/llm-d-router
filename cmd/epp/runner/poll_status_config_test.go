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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// TestPollStatusDeclaredForConfiguredSources verifies the production config
// path declares the PollStatus of the polling sources in the data layer
// configuration, here the default metrics data source.
func TestPollStatusDeclaredForConfiguredSources(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	opts := runserver.NewOptions()
	opts.ConfigText = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
`
	opts.PoolName = "poll-status-test-pool"

	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)

	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
	require.NoError(t, err)

	want := datalayer.NewPollStatusProducer(nil).TypedName().Name
	producer, ok := r.PluginHandle.Plugin(want).(fwkplugin.ProducerPlugin)
	require.True(t, ok, "the poll status producer is registered on the plugin handle")
	assert.Equal(t, map[fwkplugin.DataKey]any{fwkdl.PollStatusDataKey(sourcemetrics.MetricsDataSourceType): &fwkdl.PollStatus{}},
		producer.Produces())
}
