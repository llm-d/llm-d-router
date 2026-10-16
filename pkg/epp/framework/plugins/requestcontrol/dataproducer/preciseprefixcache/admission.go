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

package preciseprefixcache

import (
	"context"
	"slices"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

var admissionPrefixKey = plugin.NewDataKey("PreparedPrecisePrefix", PluginType)

type admissionPrefix struct {
	perPromptKeys [][]kvblock.BlockHash
	results       map[fwkdl.ID]*attrprefix.PrefixCacheMatchInfo
	totalBlocks   int
	mmTracked     bool
}

func (a *admissionPrefix) Clone() fwkdl.Cloneable {
	if a == nil {
		return nil
	}
	cloned := *a
	cloned.perPromptKeys = make([][]kvblock.BlockHash, len(a.perPromptKeys))
	for i, keys := range a.perPromptKeys {
		cloned.perPromptKeys[i] = slices.Clone(keys)
	}
	cloned.results = make(map[fwkdl.ID]*attrprefix.PrefixCacheMatchInfo, len(a.results))
	for id, info := range a.results {
		cloned.results[id] = info.Clone().(*attrprefix.PrefixCacheMatchInfo)
	}
	return &cloned
}

func (p *Producer) admissionKey() plugin.DataKey {
	return admissionPrefixKey.WithNonEmptyProducerName(p.typedName.Name)
}

func (p *Producer) PrepareForAdmission(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) error {
	prepared, err := p.matchPrefix(ctx, request, endpoints)
	if err != nil {
		return err
	}
	if err := p.publishPreparedPrefix(ctx, prepared, endpoints); err != nil {
		return err
	}
	if request != nil {
		request.PutAttribute(p.admissionKey(), prepared)
	}
	return nil
}

func (p *Producer) publishPreparedPrefix(ctx context.Context, prepared *admissionPrefix, endpoints []scheduling.Endpoint) error {
	results := make([]endpointResult, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if meta := endpoint.GetMetadata(); meta != nil {
			if info, ok := prepared.results[meta.ID]; ok {
				results = append(results, endpointResult{endpoint: endpoint, info: info})
			}
		}
	}
	return p.publishEndpointResults(ctx, results)
}
