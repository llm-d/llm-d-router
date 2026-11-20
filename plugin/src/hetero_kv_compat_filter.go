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

package disagg

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// HeteroKVCompatFilterType is the type-name of the KV compatibility filter.
const HeteroKVCompatFilterType = "hetero-kv-compat-filter"

type heteroKVCompatFilterParameters struct {
	// Decider names the hetero-pd-decider instance whose compatibility rule, approved
	// pairs and exclusions the filter applies. Defaults to "hetero-pd-decider".
	Decider string `json:"decider,omitempty"`
}

var (
	_ scheduling.Filter     = &HeteroKVCompatFilter{}
	_ plugin.ConsumerPlugin = &HeteroKVCompatFilter{}
)

// HeteroKVCompatFilter belongs in the prefill scheduling profile. It keeps only prefill
// endpoints that may exchange KV with the decode endpoint already picked (published by
// disagg-profile-handler as the peer endpoint), using the same rule as hetero-pd-decider.
// A pair that cannot exchange KV is never scheduled, instead of failing at the handshake.
//
// It fails closed: when no candidate is compatible it returns none, so the prefill
// profile fails rather than splitting toward an incompatible pod. hetero-pd-decider only
// asks for a split when a compatible candidate exists, so this is a safety net.
// Without a peer endpoint (no disaggregation in progress) candidates pass unchanged.
type HeteroKVCompatFilter struct {
	typedName   plugin.TypedName
	deciderName string
	handle      plugin.Handle
	resolveOnce sync.Once
	decider     *HeteroPDDecider
}

// HeteroKVCompatFilterFactory creates the filter.
func HeteroKVCompatFilterFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	params := heteroKVCompatFilterParameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", HeteroKVCompatFilterType, err)
		}
	}
	if params.Decider == "" {
		params.Decider = HeteroPDDeciderPluginType
	}
	if name == "" {
		name = HeteroKVCompatFilterType
	}
	return &HeteroKVCompatFilter{
		typedName:   plugin.TypedName{Type: HeteroKVCompatFilterType, Name: name},
		deciderName: params.Decider,
		handle:      handle,
	}, nil
}

// TypedName returns the typed name of the filter.
func (f *HeteroKVCompatFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// Consumes declares the peer endpoint published by disagg-profile-handler. It is
// optional: requests that are not disaggregated carry no peer.
func (f *HeteroKVCompatFilter) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Optional: map[plugin.DataKey]any{
			PeerEndpointAttributeKey: scheduling.Endpoint(nil),
		},
	}
}

func (f *HeteroKVCompatFilter) resolve(ctx context.Context) *HeteroPDDecider {
	f.resolveOnce.Do(func() {
		if f.decider != nil || f.handle == nil {
			return
		}
		if d, ok := f.handle.Plugin(f.deciderName).(*HeteroPDDecider); ok {
			f.decider = d
			return
		}
		log.FromContext(ctx).Error(fmt.Errorf("plugin %q is not a %s", f.deciderName, HeteroPDDeciderPluginType),
			"hetero-kv-compat-filter cannot apply the compatibility rule; prefill candidates pass unfiltered")
	})
	return f.decider
}

// Filter keeps the prefill candidates compatible with the peer decode endpoint.
func (f *HeteroKVCompatFilter) Filter(ctx context.Context, request *scheduling.InferenceRequest,
	endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	if request == nil {
		return endpoints
	}
	peer, ok := scheduling.ReadRequestAttribute[scheduling.Endpoint](request, PeerEndpointAttributeKey)
	if !ok || peer == nil {
		return endpoints
	}
	decider := f.resolve(ctx)
	if decider == nil {
		return endpoints
	}
	return decider.CompatiblePrefill(peer, endpoints)
}
