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

	"go.opentelemetry.io/otel/trace"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

var admissionPrefixKey = plugin.NewDataKey("PreparedPrecisePrefix", PluginType)

type admissionPrefix struct {
	keysByFormat map[kvblock.CacheFormat][][]kvblock.BlockHash
	results      map[fwkdl.ID]*attrprefix.PrefixCacheMatchInfo
}

func (a *admissionPrefix) Clone() fwkdl.Cloneable {
	if a == nil {
		return nil
	}
	cloned := *a
	cloned.keysByFormat = make(map[kvblock.CacheFormat][][]kvblock.BlockHash, len(a.keysByFormat))
	for format, prompts := range a.keysByFormat {
		clonedPrompts := make([][]kvblock.BlockHash, len(prompts))
		for i, keys := range prompts {
			clonedPrompts[i] = slices.Clone(keys)
		}
		cloned.keysByFormat[format] = clonedPrompts
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
	keysByFormat, results, err := p.matchPrefix(ctx, trace.SpanFromContext(ctx), request, endpoints)
	if err != nil {
		return err
	}
	if err := p.publishEndpointResults(ctx, results); err != nil {
		return err
	}
	if request != nil && len(keysByFormat) > 0 {
		prepared := &admissionPrefix{
			keysByFormat: keysByFormat,
			results:      make(map[fwkdl.ID]*attrprefix.PrefixCacheMatchInfo, len(results)),
		}
		for _, result := range results {
			prepared.results[result.endpoint.GetMetadata().ID] = result.info
		}
		request.PutAttribute(p.admissionKey(), prepared)
	}
	return nil
}

func (a *admissionPrefix) forEndpoints(endpoints []scheduling.Endpoint) []endpointResult {
	results := make([]endpointResult, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if endpoint == nil {
			continue
		}
		if meta := endpoint.GetMetadata(); meta != nil {
			if info, ok := a.results[meta.ID]; ok {
				results = append(results, endpointResult{endpoint: endpoint, info: info})
			}
		}
	}
	return results
}
