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

package label

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
)

const testLabel = "nvidia.com/gpu.product"

type captureRegistrar struct {
	registration fwkdl.PendingRegistration
}

func (r *captureRegistrar) Register(reg fwkdl.PendingRegistration) error {
	r.registration = reg
	return nil
}

func newEndpoint(labels map[string]string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Labels: labels}, nil)
}

func TestFactory(t *testing.T) {
	producer, err := Factory("gpu-product", fwkplugin.StrictDecoder(json.RawMessage(`{
		"label": "nvidia.com/gpu.product",
		"attributeKey": "gpu.product",
		"valueType": "string"
	}`)), nil)

	require.NoError(t, err)
	assert.Equal(t, LabelProducerType, producer.TypedName().Type)
	assert.Equal(t, "gpu-product", producer.TypedName().Name)
}

func TestFactory_Validation(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		{name: "missing label", parameters: `{"attributeKey":"gpu.product"}`, wantErr: "label"},
		{name: "missing attribute key", parameters: `{"label":"gpu"}`, wantErr: "attributeKey"},
		{name: "invalid value type", parameters: `{"label":"gpu","attributeKey":"gpu.product","valueType":"bool"}`, wantErr: "valueType"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			producer, err := Factory("test", fwkplugin.StrictDecoder(json.RawMessage(test.parameters)), nil)
			require.Error(t, err)
			assert.Contains(t, err.Error(), test.wantErr)
			assert.Nil(t, producer)
		})
	}
}

func TestExtract_StringValue(t *testing.T) {
	producer, err := NewProducer("gpu-product", parameters{
		Label:        testLabel,
		AttributeKey: "gpu.product",
		ValueType:    valueTypeString,
	})
	require.NoError(t, err)

	endpoint := newEndpoint(map[string]string{testLabel: "H100"})
	require.NoError(t, producer.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: endpoint,
	}))

	value, ok := attrstring.ReadValue(endpoint.GetAttributes(), "gpu.product")
	require.True(t, ok)
	assert.Equal(t, attrstring.Value("H100"), value)
}

func TestExtract_NumberValue(t *testing.T) {
	producer, err := NewProducer("gpu-score", parameters{
		Label:        "gpu.score",
		AttributeKey: "gpu.score",
		ValueType:    valueTypeNumber,
	})
	require.NoError(t, err)

	endpoint := newEndpoint(map[string]string{"gpu.score": "2.5"})
	require.NoError(t, producer.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: endpoint,
	}))

	value, ok := attrmetrics.ReadScalarMetricValue(endpoint.GetAttributes(), "gpu.score")
	require.True(t, ok)
	assert.Equal(t, attrmetrics.ScalarMetricValue(2.5), value)
}

func TestExtract_MissingLabelWritesFallbackValue(t *testing.T) {
	producer, err := NewProducer("gpu-product", parameters{
		Label:        testLabel,
		AttributeKey: "gpu.product",
		ValueType:    valueTypeString,
	})
	require.NoError(t, err)

	endpoint := newEndpoint(nil)
	require.NoError(t, producer.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: endpoint,
	}))

	value, ok := attrstring.ReadValue(endpoint.GetAttributes(), "gpu.product")
	require.True(t, ok)
	assert.Empty(t, value)
}

func TestRegisterDependencies(t *testing.T) {
	producer, err := NewProducer("gpu-product", parameters{
		Label:        testLabel,
		AttributeKey: "gpu.product",
	})
	require.NoError(t, err)

	var registrar captureRegistrar
	require.NoError(t, producer.RegisterDependencies(&registrar))
	assert.Equal(t, "endpoint-notification-source", registrar.registration.SourceType)
	assert.Same(t, producer, registrar.registration.Extractor)
}
