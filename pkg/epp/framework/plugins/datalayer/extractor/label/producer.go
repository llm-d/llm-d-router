/*
Copyright 2026 The Kubernetes Authors.

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

// Package label publishes endpoint metadata labels as string attributes.
package label

import (
	"context"
	"encoding/json"
	"fmt"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

const LabelProducerType = "label-producer"

var (
	_ fwkplugin.ProducerPlugin = (*Producer)(nil)
	_ fwkdl.Registrant         = (*Producer)(nil)
	_ fwkdl.EndpointExtractor  = (*Producer)(nil)
)

type parameters struct {
	Labels []labelMapping `json:"labels"`
}

type labelMapping struct {
	Label        string `json:"label"`
	AttributeKey string `json:"attributeKey"`
}

func Factory(name string, decoder *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var params parameters
	if decoder != nil {
		if err := decoder.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the %q plugin: %w", LabelProducerType, err)
		}
	}
	producer, err := NewProducer(name, params)
	if err != nil {
		return nil, err
	}
	if handle != nil {
		for _, existing := range handle.GetAllPlugins() {
			if existing.TypedName().Type == LabelProducerType {
				return nil, fmt.Errorf("multiple %s instances configured (%s, %s); only one instance is supported",
					LabelProducerType, existing.TypedName(), producer.TypedName())
			}
		}
	}
	return producer, nil
}

func NewProducer(name string, params parameters) (*Producer, error) {
	if name == "" {
		name = LabelProducerType
	}
	if len(params.Labels) == 0 {
		return nil, fmt.Errorf("%q requires non-empty 'labels'", LabelProducerType)
	}
	labels := make(map[fwkplugin.DataKey]string, len(params.Labels))
	for _, mapping := range params.Labels {
		if mapping.Label == "" || mapping.AttributeKey == "" {
			return nil, fmt.Errorf("%q requires non-empty 'label' and 'attributeKey' in each mapping", LabelProducerType)
		}
		key := fwkplugin.NewDataKey(mapping.AttributeKey, name)
		if _, exists := labels[key]; exists {
			return nil, fmt.Errorf("%q has duplicate attributeKey %q", LabelProducerType, mapping.AttributeKey)
		}
		labels[key] = mapping.Label
	}

	return &Producer{
		typedName: fwkplugin.TypedName{Type: LabelProducerType, Name: name},
		labels:    labels,
	}, nil
}

type Producer struct {
	typedName fwkplugin.TypedName
	labels    map[fwkplugin.DataKey]string
}

func (p *Producer) TypedName() fwkplugin.TypedName { return p.typedName }

func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	produced := make(map[fwkplugin.DataKey]any, len(p.labels))
	for key := range p.labels {
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

func (p *Producer) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Type == fwkdl.EventDelete || event.Endpoint == nil {
		return nil
	}

	endpoint := event.Endpoint
	// Resolve labels on read to reflect metadata updates.
	for key, label := range p.labels {
		endpoint.GetAttributes().Put(key, &fwkdl.DynamicAttribute{
			Get: func() fwkdl.Cloneable {
				meta := endpoint.GetMetadata()
				if meta == nil {
					return nil
				}
				value, found := meta.Labels[label]
				if !found {
					return nil
				}
				return attrstring.Value(value)
			},
		})
	}
	return nil
}
