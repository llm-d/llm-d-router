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

// Package attribute publishes discovery-declared endpoint attributes
// (EndpointMetadata.Attributes, e.g. a file-discovery entry's attributes
// section) as string attributes resolved on read. Reading through
// DynamicAttribute means discovery updates (a watched file's attributes
// edited or removed) take effect for the endpoint's remaining lifetime;
// nothing is stamped once at construction.
package attribute

import (
	"context"
	"encoding/json"
	"fmt"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

// AttributeProducerType is the plugin type for the endpoint attributes producer.
const AttributeProducerType = "attributes-producer"

type parameters struct {
	// AttributeKeys lists the metadata.Attributes keys to surface as endpoint
	// attributes. Required, non-empty.
	AttributeKeys []string `json:"attributeKeys"`
}

// Factory creates an attributes producer.
func Factory(name string, decoder *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var params parameters
	if decoder != nil {
		if err := decoder.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the %q plugin: %w", AttributeProducerType, err)
		}
	}
	if len(params.AttributeKeys) == 0 {
		return nil, fmt.Errorf("%q requires non-empty 'attributeKeys'", AttributeProducerType)
	}
	keys := make([]fwkplugin.DataKey, 0, len(params.AttributeKeys))
	seen := map[fwkplugin.DataKey]struct{}{}
	for _, k := range params.AttributeKeys {
		if k == "" {
			return nil, fmt.Errorf("%q requires non-empty entries in 'attributeKeys'", AttributeProducerType)
		}
		key := fwkplugin.NewDataKey(k, nameOrDefault(name))
		if _, dup := seen[key]; dup {
			return nil, fmt.Errorf("%q has duplicate attributeKey %q", AttributeProducerType, k)
		}
		seen[key] = struct{}{}
		keys = append(keys, key)
	}
	return &Producer{
		typedName: fwkplugin.TypedName{Type: AttributeProducerType, Name: nameOrDefault(name)},
		keys:      keys,
		byKey:     zipKeyToAttribute(keys, params.AttributeKeys),
	}, nil
}

func nameOrDefault(name string) string {
	if name == "" {
		return AttributeProducerType
	}
	return name
}

func zipKeyToAttribute(keys []fwkplugin.DataKey, attrs []string) map[fwkplugin.DataKey]string {
	m := make(map[fwkplugin.DataKey]string, len(keys))
	for i, key := range keys {
		m[key] = attrs[i]
	}
	return m
}

var (
	_ fwkplugin.ProducerPlugin = (*Producer)(nil)
	_ fwkdl.Registrant         = (*Producer)(nil)
	_ fwkdl.EndpointExtractor  = (*Producer)(nil)
)

// Producer surfaces configured EndpointMetadata.Attributes keys as dynamic
// string attributes on each endpoint.
type Producer struct {
	typedName fwkplugin.TypedName
	keys      []fwkplugin.DataKey
	byKey     map[fwkplugin.DataKey]string
}

func (p *Producer) TypedName() fwkplugin.TypedName { return p.typedName }

func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	produced := make(map[fwkplugin.DataKey]any, len(p.keys))
	for _, key := range p.keys {
		produced[key] = attrstring.Value("")
	}
	return produced
}

func (p *Producer) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      p.typedName,
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType,
		),
	})
}

// Extract installs a DynamicAttribute per configured key that resolves the
// value from the endpoint's current metadata on every read, so discovery
// updates (edited or removed attributes) propagate without re-stamping.
func (p *Producer) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Type == fwkdl.EventDelete || event.Endpoint == nil {
		return nil
	}

	endpoint := event.Endpoint
	for key, attr := range p.byKey {
		endpoint.GetAttributes().Put(key, &fwkdl.DynamicAttribute{
			Get: func() fwkdl.Cloneable {
				meta := endpoint.GetMetadata()
				if meta == nil {
					return nil
				}
				value, found := meta.Attributes[attr]
				if !found {
					return nil
				}
				return attrstring.Value(value)
			},
		})
	}
	return nil
}
