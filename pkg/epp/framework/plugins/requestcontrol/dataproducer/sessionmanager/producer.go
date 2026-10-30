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

// Package sessionmanager publishes per-request session cache stamps without
// inferring continuations or publishing cache prefixes.
package sessionmanager

import (
	"context"
	"encoding/json"
	"fmt"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
)

const PluginType = "session-manager"

var (
	_ fwkplugin.ConsumerPlugin    = (*Producer)(nil)
	_ requestcontrol.DataProducer = (*Producer)(nil)
)

// Producer publishes cache requests for identities minted before flow control.
type Producer struct {
	typedName       fwkplugin.TypedName
	identityKey     fwkplugin.DataKey
	cacheRequestKey fwkplugin.DataKey
	stamps          *stampGenerator
	metrics         *managerMetrics
}

// Factory constructs a configured session-manager.
func Factory(name string, decoder *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var config Config
	if decoder != nil {
		if err := decoder.Decode(&config); err != nil {
			return nil, fmt.Errorf("decode %s parameters: %w", PluginType, err)
		}
	}
	resolved, err := config.resolve()
	if err != nil {
		return nil, fmt.Errorf("configure %s: %w", PluginType, err)
	}
	if name == "" {
		name = PluginType
	}
	stamps, err := newStampGenerator()
	if err != nil {
		return nil, fmt.Errorf("initialize request stamps: %w", err)
	}
	var registerer fwkplugin.MetricsRecorder
	if handle != nil {
		registerer = handle.Metrics()
	}
	metrics, err := newManagerMetrics(name, registerer)
	if err != nil {
		return nil, err
	}
	return &Producer{
		typedName:       fwkplugin.TypedName{Type: PluginType, Name: name},
		identityKey:     requestcontrol.SessionIdentityDataKey.WithNonEmptyProducerName(resolved.sessionIdentityProducer),
		cacheRequestKey: attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName(name),
		stamps:          stamps,
		metrics:         metrics,
	}, nil
}

func (p *Producer) TypedName() fwkplugin.TypedName { return p.typedName }

func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.cacheRequestKey: attrsession.SessionCacheRequest{},
	}
}

func (p *Producer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{p.identityKey: requestcontrol.SessionIdentity{}},
	}
}

// Produce publishes only the request stamp consumed by
// session-prefix-cache-producer. Token data is intentionally unavailable here
// because tokenization runs after admission control.
func (p *Producer) Produce(_ context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	if request == nil {
		return nil
	}
	identity, ok := fwksched.ReadRequestAttribute[requestcontrol.SessionIdentity](request, p.identityKey)
	if !ok || !identity.Valid() {
		p.metrics.requestOutcomes.WithLabelValues("missing_identity").Inc()
		return nil
	}
	stamp, err := p.stamps.next()
	if err != nil {
		return err
	}
	request.PutAttribute(p.cacheRequestKey, attrsession.SessionCacheRequest{
		SessionID:   stamp,
		FullReport:  false,
		TotalTokens: 0,
		Prefixes:    nil,
	})
	p.metrics.requestOutcomes.WithLabelValues("published").Inc()
	return nil
}
