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

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

var admissionPrefixKey = plugin.NewDataKey("PreparedApproximatePrefix", ApproxPrefixCachePluginType)

type admissionPrefix struct {
	state       *SchedulingContextState
	totalBlocks int
	blockSize   int
}

func (a *admissionPrefix) Clone() fwkdl.Cloneable {
	if a == nil {
		return nil
	}
	cloned := *a
	if a.state != nil {
		cloned.state = a.state.Clone().(*SchedulingContextState)
	}
	return &cloned
}

func (p *dataProducer) admissionKey() plugin.DataKey {
	return admissionPrefixKey.WithNonEmptyProducerName(p.typedName.Name)
}

func (p *dataProducer) PrepareForAdmission(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	prepared := p.matchPrefix(ctx, request, endpoints)
	if err := ctx.Err(); err != nil {
		return err
	}
	p.publishPrefix(prepared, endpoints)
	request.PutAttribute(p.admissionKey(), prepared)
	return nil
}

func (p *dataProducer) publishPrefix(prepared *admissionPrefix, endpoints []fwksched.Endpoint) {
	for _, endpoint := range endpoints {
		match := prepared.state.PrefixCacheServers[ServerID(endpoint.GetMetadata().ID)]
		endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(match, prepared.totalBlocks, prepared.blockSize))
	}
}
