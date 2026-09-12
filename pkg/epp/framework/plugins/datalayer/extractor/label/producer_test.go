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

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	dlruntime "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
)

const (
	testProducerName = "endpoint-labels"
	testStringLabel  = "nvidia.com/gpu.product"
	testRegionLabel  = "topology.kubernetes.io/region"
)

func newEndpoint(labels map[string]string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:     types.NamespacedName{Namespace: "default", Name: "endpoint-0"},
		Labels: labels,
	}, nil)
}

func dataKey(key string) fwkplugin.DataKey {
	return fwkplugin.NewDataKey(key, testProducerName)
}

func testParameters() parameters {
	return parameters{Labels: []labelMapping{
		{Label: testStringLabel, AttributeKey: "gpu.product"},
		{Label: testRegionLabel, AttributeKey: "region"},
	}}
}

func mustProducer(t *testing.T) *Producer {
	t.Helper()
	producer, err := NewProducer(testProducerName, testParameters())
	require.NoError(t, err)
	return producer
}

func extractEndpoint(producer *Producer, endpoint fwkdl.Endpoint) error {
	return producer.Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint,
	})
}

func TestFactory(t *testing.T) {
	producer, err := Factory(testProducerName, fwkplugin.StrictDecoder(json.RawMessage(`{
		"labels": [{"label":"nvidia.com/gpu.product","attributeKey":"gpu.product"},
			{"label":"topology.kubernetes.io/region","attributeKey":"region"}]
	}`)), nil)

	require.NoError(t, err)
	assert.Equal(t, LabelProducerType, producer.TypedName().Type)
	assert.Equal(t, testProducerName, producer.TypedName().Name)
}

func TestFactoryValidation(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		{name: "missing mappings", parameters: `{}`, wantErr: "labels"},
		{name: "empty mappings", parameters: `{"labels":[]}`, wantErr: "labels"},
		{name: "missing label", parameters: `{"labels":[{"attributeKey":"gpu.product"}]}`, wantErr: "label"},
		{name: "missing attribute key", parameters: `{"labels":[{"label":"gpu"}]}`, wantErr: "attributeKey"},
		{name: "duplicate output", parameters: `{"labels":[{"label":"gpu","attributeKey":"a"},{"label":"region","attributeKey":"a"}]}`, wantErr: "duplicate attributeKey"},
		{name: "unknown field", parameters: `{"labels":[{"label":"gpu","attributeKey":"a","typo":true}]}`, wantErr: "unknown field"},
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

func TestFactoryRejectsMultipleInstances(t *testing.T) {
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	existing, err := NewProducer("existing", testParameters())
	require.NoError(t, err)
	handle.AddPlugin(existing.TypedName().Name, existing)

	producer, err := Factory("duplicate", fwkplugin.StrictDecoder(json.RawMessage(`{
		"labels": [{"label":"nvidia.com/gpu.product","attributeKey":"gpu.product"}]
	}`)), handle)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "multiple label-producer instances")
	assert.Contains(t, err.Error(), existing.TypedName().String())
	assert.Nil(t, producer)
}

func TestProduces(t *testing.T) {
	producer := mustProducer(t)
	produced := producer.Produces()
	require.Len(t, produced, 2)
	for _, key := range []string{"gpu.product", "region"} {
		assert.IsType(t, attrstring.Value(""), produced[dataKey(key)])
	}
}

func TestExtractFollowsLabelUpdates(t *testing.T) {
	endpoint := newEndpoint(nil)
	require.NoError(t, extractEndpoint(mustProducer(t), endpoint))
	for _, labels := range []map[string]string{
		nil,
		{testStringLabel: "H100", testRegionLabel: "region-1"},
		{testStringLabel: "A100", testRegionLabel: "region-1"},
		{testStringLabel: "", testRegionLabel: "region-2"},
		{testRegionLabel: "region-1"},
		{testStringLabel: "L40S"},
	} {
		endpoint.UpdateMetadata(&fwkdl.EndpointMetadata{Labels: labels})
		for key, label := range map[string]string{"gpu.product": testStringLabel, "region": testRegionLabel} {
			value, found := attrstring.ReadValue(endpoint.GetAttributes(), dataKey(key))
			want, present := labels[label]
			assert.Equal(t, present, found, "label %s in %v", label, labels)
			assert.Equal(t, attrstring.Value(want), value)
		}
	}
}

func TestExtractDeleteDoesNothing(t *testing.T) {
	endpoint := newEndpoint(map[string]string{testStringLabel: "H100"})
	require.NoError(t, mustProducer(t).Extract(context.Background(), fwkdl.EndpointEvent{
		Type: fwkdl.EventDelete, Endpoint: endpoint,
	}))
	assert.Empty(t, endpoint.GetAttributes().Keys())
}

func TestRuntimePublishesLabel(t *testing.T) {
	producer := mustProducer(t)
	runtime := dlruntime.NewRuntime(0)
	require.NoError(t, producer.RegisterDependencies(runtime))
	require.NoError(t, runtime.Configure(nil, logr.Discard()))

	endpoint := runtime.NewEndpoint(context.Background(), &fwkdl.EndpointMetadata{
		ID:     types.NamespacedName{Namespace: "default", Name: "endpoint-0"},
		Labels: map[string]string{testStringLabel: "H100", testRegionLabel: "region-1"},
	})
	require.NotNil(t, endpoint)
	for key, want := range map[string]string{"gpu.product": "H100", "region": "region-1"} {
		value, found := attrstring.ReadValue(endpoint.GetAttributes(), dataKey(key))
		require.True(t, found)
		assert.Equal(t, attrstring.Value(want), value)
	}
}
