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

// Package label provides an endpoint extractor that copies a Pod label into
// an endpoint attribute.
package label

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

const (
	// LabelProducerType identifies the label producer plugin.
	LabelProducerType = "label-producer"

	valueTypeString = "string"
	valueTypeNumber = "number"
)

var (
	_ fwkplugin.Plugin        = (*Producer)(nil)
	_ fwkdl.Registrant        = (*Producer)(nil)
	_ fwkdl.EndpointExtractor = (*Producer)(nil)
)

type parameters struct {
	// Label is the Pod label to copy.
	Label string `json:"label"`
	// AttributeKey is the endpoint attribute key to populate.
	AttributeKey string `json:"attributeKey"`
	// ValueType controls whether the value is stored as a string or number.
	ValueType string `json:"valueType"`
}

// Factory creates a label producer.
func Factory(name string, decoder *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var params parameters
	if decoder != nil {
		if err := decoder.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", LabelProducerType, err)
		}
	}
	return NewProducer(name, params)
}

// NewProducer validates parameters and creates a label producer.
func NewProducer(name string, params parameters) (*Producer, error) {
	if name == "" {
		name = LabelProducerType
	}
	if params.Label == "" {
		return nil, fmt.Errorf("'%s' requires a non-empty 'label'", LabelProducerType)
	}
	if params.AttributeKey == "" {
		return nil, fmt.Errorf("'%s' requires a non-empty 'attributeKey'", LabelProducerType)
	}
	if params.ValueType == "" {
		params.ValueType = valueTypeString
	}
	if params.ValueType != valueTypeString && params.ValueType != valueTypeNumber {
		return nil, fmt.Errorf("'%s' valueType must be %q or %q, got %q", LabelProducerType, valueTypeString, valueTypeNumber, params.ValueType)
	}

	return &Producer{
		typedName:    fwkplugin.TypedName{Type: LabelProducerType, Name: name},
		label:        params.Label,
		attributeKey: params.AttributeKey,
		valueType:    params.ValueType,
	}, nil
}

// Producer copies a Pod label into an endpoint attribute.
type Producer struct {
	typedName    fwkplugin.TypedName
	label        string
	attributeKey string
	valueType    string
}

// TypedName returns the type and name of the producer.
func (p *Producer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// RegisterDependencies wires the producer to endpoint lifecycle events.
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

// Extract copies the configured label into the endpoint attribute.
func (p *Producer) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Type == fwkdl.EventDelete || event.Endpoint == nil {
		return nil
	}

	meta := event.Endpoint.GetMetadata()
	if meta == nil {
		return nil
	}
	value := meta.Labels[p.label]

	if p.valueType == valueTypeString {
		event.Endpoint.GetAttributes().Put(p.attributeKey, attrstring.Value(value))
		return nil
	}

	if value == "" {
		event.Endpoint.GetAttributes().Put(p.attributeKey, attrmetrics.ScalarMetricValue(0))
		return nil
	}
	number, err := strconv.ParseFloat(value, 64)
	if err != nil {
		event.Endpoint.GetAttributes().Put(p.attributeKey, attrmetrics.ScalarMetricValue(0))
		return fmt.Errorf("label %q on endpoint %s has non-numeric value %q: %w", p.label, meta.NamespacedName, value, err)
	}
	event.Endpoint.GetAttributes().Put(p.attributeKey, attrmetrics.ScalarMetricValue(number))
	return nil
}
