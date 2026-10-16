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

package approximateprefix

import (
	"context"
	"testing"

	"github.com/go-logr/logr"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	datagraph "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixmetrics"
)

type admissionCountingIndexer struct {
	indexerInterface
	queries    int
	afterMatch func()
}

func (i *admissionCountingIndexer) MatchLongestPrefix(hashes []blockHash, candidates []ServerID) []int {
	i.queries++
	matched := i.indexerInterface.MatchLongestPrefix(hashes, candidates)
	if i.afterMatch != nil {
		i.afterMatch()
	}
	return matched
}

func TestAdmissionPredictionUsesPreparedMatchesAndProduceCandidates(t *testing.T) {
	disableMinBlockSizeClamp(t)
	name := t.Name()
	const blockSize = 2
	p, err := newDataProducer(t.Context(), name, config{BlockSizeTokens: blockSize}, testHandle())
	require.NoError(t, err)
	indexer := &admissionCountingIndexer{indexerInterface: p.indexerInst}
	p.indexerInst = indexer
	endpoints := []fwksched.Endpoint{namedEndpoint("a"), namedEndpoint("b"), namedEndpoint("c"), namedEndpoint("d")}
	request := &fwksched.InferenceRequest{RequestID: "queued-predictions", Body: tokenizedBody([]uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})}
	hashes := prefixhash.GetBlockHashes(t.Context(), request, blockSize, unlimitedPrefixBlocks)
	metrics := []string{predictedCachedTokensMetric, bestPredictedMetric, bestAvailableMetric, promptTokensMetric}
	before := make(map[string]*dto.Histogram, len(metrics))
	for _, metric := range metrics {
		before[metric] = admissionHistogram(t, metric, name)
	}
	datagraph.RegisterScopeSpecs([]plugin.Plugin{p})
	for _, matches := range [][]int{{2, 3, 5, 7}, {1, 2, 4, 6}} {
		for i, endpoint := range endpoints {
			id := ServerID(endpoint.GetMetadata().ID)
			indexer.RemovePod(id)
			indexer.Add(hashes[0][:matches[i]], server{ServerID: id, NumOfGPUBlocks: 10})
		}
		scoped, scopedEndpoints, violations := datagraph.ScopeInvocation(logr.Discard(), requestcontrol.AdmissionDataProducerExtensionPoint, p, request, endpoints)
		require.NoError(t, p.PrepareForAdmission(t.Context(), scoped, scopedEndpoints))
		require.NoError(t, violations.Write())
		_, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestID, plugin.StateKey(name))
		require.Error(t, err)
		for _, metric := range metrics {
			require.Equal(t, before[metric].GetSampleCount(), admissionHistogram(t, metric, name).GetSampleCount())
		}
	}
	require.Equal(t, 2, indexer.queries)
	prepared, ok := fwksched.ReadRequestAttribute[*admissionPrefix](request, p.admissionKey())
	require.True(t, ok)
	snapshot := prepared.Clone()
	for _, endpoint := range endpoints {
		indexer.RemovePod(ServerID(endpoint.GetMetadata().ID))
	}

	// The producer's candidate boundary excludes d; the picker later excludes c.
	produced := make([]fwksched.Endpoint, 3)
	for i := range produced {
		produced[i] = fwksched.NewEndpoint(endpoints[i].GetMetadata(), nil, nil)
	}
	scoped, scopedEndpoints, violations := datagraph.ScopeInvocation(logr.Discard(), requestcontrol.DataProducerExtensionPoint, p, request, produced)
	require.NoError(t, p.Produce(t.Context(), scoped, scopedEndpoints))
	require.NoError(t, violations.Write())
	require.Equal(t, 2, indexer.queries, "Produce must use the admitted match snapshot")
	require.Equal(t, snapshot, prepared, "publishing must not mutate the admission snapshot")
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestID, plugin.StateKey(name))
	require.NoError(t, err)
	require.Equal(t, 4*blockSize, state.BestAvailableCachedTokens)
	for i, blocks := range []int{1, 2, 4} {
		info, ok := produced[i].Get(p.dk)
		require.True(t, ok)
		require.Equal(t, blocks, info.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks())
	}
	for _, metric := range metrics {
		require.Equal(t, before[metric].GetSampleCount(), admissionHistogram(t, metric, name).GetSampleCount())
	}
	require.NoError(t, p.PreRequest(t.Context(), request, resultWith(produced[0], produced[0], produced[1])))
	p.wg.Wait()
	want := []float64{blockSize, 2 * blockSize, 4 * blockSize, 16}
	for i, metric := range metrics {
		after := admissionHistogram(t, metric, name)
		require.Equal(t, before[metric].GetSampleCount()+1, after.GetSampleCount())
		require.Equal(t, before[metric].GetSampleSum()+want[i], after.GetSampleSum())
	}
	_, err = plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestID, plugin.StateKey(name))
	require.Error(t, err)
}

func TestAdmissionPreparationCancellationPublishesNoState(t *testing.T) {
	disableMinBlockSizeClamp(t)
	p, err := newDataProducer(t.Context(), "cancelled-admission", config{BlockSizeTokens: 1}, testHandle())
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	p.indexerInst = &admissionCountingIndexer{indexerInterface: p.indexerInst, afterMatch: cancel}
	request := &fwksched.InferenceRequest{RequestID: "cancelled", Body: tokenizedBody([]uint32{1, 2})}
	endpoints := []fwksched.Endpoint{namedEndpoint("candidate")}
	require.ErrorIs(t, p.PrepareForAdmission(ctx, request, endpoints), context.Canceled)
	_, ok := request.GetAttribute(p.admissionKey())
	require.False(t, ok)
	_, ok = endpoints[0].Get(p.dk)
	require.False(t, ok)
	_, err = plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestID, plugin.StateKey(p.typedName.Name))
	require.Error(t, err)
}

func admissionHistogram(t *testing.T, metricName, pluginName string) *dto.Histogram {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != metricName {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			if labels["plugin_name"] == pluginName && labels["endpoint_role"] == prefixmetrics.RoleDecode {
				return metric.GetHistogram()
			}
		}
	}
	return nil
}

func TestAdmissionPreparationDefersStateAndReusesMatchingSnapshot(t *testing.T) {
	disableMinBlockSizeClamp(t)
	p, err := newDataProducer(t.Context(), "prepared-prefix", config{BlockSizeTokens: 1}, testHandle())
	require.NoError(t, err)
	indexer := &admissionCountingIndexer{indexerInterface: p.indexerInst}
	p.indexerInst = indexer
	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod"}}, nil, nil)
	endpoints := []fwksched.Endpoint{endpoint}
	request := &fwksched.InferenceRequest{RequestID: "queued", Body: tokenizedBody([]uint32{1, 2})}
	hashes := prefixhash.GetBlockHashes(t.Context(), request, 1, unlimitedPrefixBlocks)
	indexer.Add(hashes[0], server{ServerID: ServerID(endpoint.GetMetadata().ID), NumOfGPUBlocks: 10})

	require.NoError(t, p.PrepareForAdmission(t.Context(), request, endpoints))
	_, err = plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestID, plugin.StateKey(p.typedName.Name))
	require.Error(t, err)
	info, _ := endpoint.Get(p.dk)
	require.Equal(t, 2, info.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks())

	indexer.RemovePod(ServerID(endpoint.GetMetadata().ID))
	require.NoError(t, p.PrepareForAdmission(t.Context(), request, endpoints))
	info, _ = endpoint.Get(p.dk)
	require.Zero(t, info.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks())
	_, err = plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestID, plugin.StateKey(p.typedName.Name))
	require.Error(t, err)

	indexer.Add(hashes[0], server{ServerID: ServerID(endpoint.GetMetadata().ID), NumOfGPUBlocks: 10})
	queries := indexer.queries
	datagraph.RegisterScopeSpecs([]plugin.Plugin{p})
	scoped, violations := datagraph.ScopeRequest(logr.Discard(), requestcontrol.DataProducerExtensionPoint, p, request)
	require.NoError(t, p.Produce(t.Context(), scoped, endpoints))
	require.NoError(t, violations.Write())
	require.Equal(t, queries, indexer.queries)
	info, _ = endpoint.Get(p.dk)
	require.Zero(t, info.(*attrprefix.PrefixCacheMatchInfo).MatchBlocks())
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.pluginState, request.RequestID, plugin.StateKey(p.typedName.Name))
	require.NoError(t, err)
	require.Zero(t, state.PrefixCacheServers[ServerID(endpoint.GetMetadata().ID)])

	prepared, ok := fwksched.ReadRequestAttribute[*admissionPrefix](request, p.admissionKey())
	require.True(t, ok)
	cloned := prepared.Clone().(*admissionPrefix)
	prepared.state.PerPromptHashes[0][0]++
	prepared.state.PrefixCacheServers[ServerID(endpoint.GetMetadata().ID)] = 42
	require.NotEqual(t, prepared.state.PerPromptHashes, cloned.state.PerPromptHashes)
	require.Zero(t, cloned.state.PrefixCacheServers[ServerID(endpoint.GetMetadata().ID)])
}
