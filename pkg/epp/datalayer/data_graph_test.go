/*
Copyright 2025 The Kubernetes Authors.

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

package datalayer

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const mockProducedDataKey = "mockProducedData"

type mockDataProducerP struct {
	name     string
	produces map[string]any
	consumes map[string]any
}

type mockProducedDataType struct {
	value int
}

func (m *mockProducedDataType) Clone() fwkdl.Cloneable {
	return &mockProducedDataType{value: m.value}
}

func (m *mockDataProducerP) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: m.name, Type: "mock"}
}

func (m *mockDataProducerP) Produces() map[string]any {
	return m.produces
}

func (m *mockDataProducerP) Consumes() map[string]any {
	return m.consumes
}

func (m *mockDataProducerP) Produce(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	endpoints[0].Put(mockProducedDataKey, &mockProducedDataType{value: 42})
	return nil
}

// typedMockPlugin is a DataProducer whose TypedName.Type can be set explicitly,
// allowing tests to simulate a plugin whose registry type is already present.
type typedMockPlugin struct {
	typeName string
	produces map[string]any
}

func (m *typedMockPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: m.typeName, Type: m.typeName}
}

func (m *typedMockPlugin) Produces() map[string]any { return m.produces }
func (m *typedMockPlugin) Produce(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	return nil
}

type MockConsumerFairnessPolicy struct {
	fwkfcmocks.MockFairnessPolicy
	consumes map[string]any
}

func (m *MockConsumerFairnessPolicy) Consumes() map[string]any {
	return m.consumes
}

type MockSchedulingPlugin struct {
	fwksched.Scorer
	consumes map[string]any
}

func (m *MockSchedulingPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: "MockSchedulingPlugin", Type: "mock"}
}

func (m *MockSchedulingPlugin) Consumes() map[string]any {
	return m.consumes
}

func TestValidatePluginExecutionOrder(t *testing.T) {
	// Request control plugin that produces data.
	pluginA := &mockDataProducerP{name: "A", produces: map[string]any{"keyA": nil}}
	// Flow control plugin.
	consumerFairnessPolicyPlugin := MockConsumerFairnessPolicy{consumes: map[string]any{"keyA": nil}}
	// Scheduling plugin.
	consumerSchedulingPlugin := MockSchedulingPlugin{consumes: map[string]any{"keyA": nil}}
	if _, ok := any(pluginA).(fwkrc.DataProducer); !ok {
		t.Fatalf("pluginA should implement DataProducer")
	}

	testCases := []struct {
		name        string
		plugins     []fwkplugin.Plugin
		expectedErr string
	}{
		{
			name:        "Plugins with no dependencies",
			plugins:     []fwkplugin.Plugin{pluginA},
			expectedErr: "",
		},
		{
			name:        "FC depends on a request control plugin (invalid layer execution order)",
			plugins:     []fwkplugin.Plugin{pluginA, &consumerFairnessPolicyPlugin},
			expectedErr: "invalid plugin layer execution order",
		},
		{
			name:        "Scheduling plugin depends on a request control plugin",
			plugins:     []fwkplugin.Plugin{pluginA, &consumerSchedulingPlugin},
			expectedErr: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ValidateAndOrderDataDependencies(tc.plugins)
			if tc.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectedErr)
				return
			}
			assert.NoError(t, err)
		})
	}
}

func TestDAGAndTopologicalOrder(t *testing.T) {
	pluginA := &mockDataProducerP{name: "A", produces: map[string]any{"keyA": nil}}
	pluginB := &mockDataProducerP{name: "B", consumes: map[string]any{"keyA": nil}, produces: map[string]any{"keyB": nil}}
	pluginC := &mockDataProducerP{name: "C", consumes: map[string]any{"keyB": nil}}
	pluginD := &mockDataProducerP{name: "D", consumes: map[string]any{"keyA": nil}}
	pluginE := &mockDataProducerP{name: "E"} // No dependencies

	// Cycle plugins
	pluginX := &mockDataProducerP{name: "X", produces: map[string]any{"keyX": nil}, consumes: map[string]any{"keyY": nil}}
	pluginY := &mockDataProducerP{name: "Y", produces: map[string]any{"keyY": nil}, consumes: map[string]any{"keyX": nil}}

	// Data type mismatch plugin.
	pluginZ1 := &mockDataProducerP{name: "Z1", produces: map[string]any{"keyZ": int(0)}}
	pluginZ2 := &mockDataProducerP{name: "Z2", consumes: map[string]any{"keyZ": string("")}}

	// Same type different pointers.
	pluginP1 := &mockDataProducerP{name: "P1", produces: map[string]any{"keyP": &mockProducedDataType{}}}
	pluginP2 := &mockDataProducerP{name: "P2", consumes: map[string]any{"keyP": &mockProducedDataType{}}}

	testCases := []struct {
		name        string
		plugins     []fwkrc.DataProducer
		expectedDAG map[string][]string
		expectedErr string
	}{
		{
			name:        "No plugins",
			plugins:     []fwkrc.DataProducer{},
			expectedDAG: map[string][]string{},
			expectedErr: "",
		},
		{
			name:    "Plugins with no dependencies",
			plugins: []fwkrc.DataProducer{pluginA, pluginE},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"E/mock": {},
			},
			expectedErr: "",
		},
		{
			name:    "Simple linear dependency (C -> B -> A)",
			plugins: []fwkrc.DataProducer{pluginA, pluginB, pluginC},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"B/mock": {"A/mock"},
				"C/mock": {"B/mock"},
			},
			expectedErr: "",
		},
		{
			name:    "DAG with multiple dependencies (B -> A, D -> A, E independent)",
			plugins: []fwkrc.DataProducer{pluginA, pluginB, pluginD, pluginE},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"B/mock": {"A/mock"},
				"D/mock": {"A/mock"},
				"E/mock": {},
			},
			expectedErr: "",
		},
		{
			name:        "Graph with a cycle (X -> Y, Y -> X)",
			plugins:     []fwkrc.DataProducer{pluginX, pluginY},
			expectedDAG: nil,
			expectedErr: "cycle detected",
		},
		{
			name:        "Data type mismatch between produced and consumed data",
			plugins:     []fwkrc.DataProducer{pluginZ1, pluginZ2},
			expectedDAG: nil,
			expectedErr: "data type mismatch between produced and consumed data",
		},
		{
			name:    "Same type different pointers (should succeed)",
			plugins: []fwkrc.DataProducer{pluginP1, pluginP2},
			expectedDAG: map[string][]string{
				"P1/mock": {},
				"P2/mock": {"P1/mock"},
			},
			expectedErr: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			producers := make(map[string]fwkplugin.ProducerPlugin)
			consumers := make(map[string]fwkplugin.ConsumerPlugin)
			for _, p := range tc.plugins {
				if pp, ok := p.(fwkplugin.ProducerPlugin); ok {
					producers[p.TypedName().String()] = pp
				}
				if cp, ok := p.(fwkplugin.ConsumerPlugin); ok {
					consumers[p.TypedName().String()] = cp
				}
			}
			dag, err := buildDAG(producers, consumers)
			if err != nil {
				if tc.expectedErr != "" {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), tc.expectedErr)
					return
				}
				assert.NoError(t, err)
			}
			orderedPlugins, err := topologicalSort(dag)

			if tc.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectedErr)
				return
			}
			assert.NoError(t, err)

			// Normalize the slices in the maps for consistent comparison
			normalizedDAG := make(map[string][]string)
			maps.Copy(normalizedDAG, dag)
			normalizedExpectedDAG := make(map[string][]string)
			maps.Copy(normalizedExpectedDAG, tc.expectedDAG)

			if diff := cmp.Diff(normalizedExpectedDAG, normalizedDAG); diff != "" {
				t.Errorf("dataProducerGraph() mismatch (-want +got):\n%s", diff)
			}

			assertTopologicalOrder(t, dag, orderedPlugins)
		})
	}
}

func TestCreateMissingDataProducers(t *testing.T) {
	const (
		keyA = "keyA"
		keyB = "keyB"
	)

	// A DataProducer that produces keyA.
	producerTypeA := "producer-a"
	producerAFactory := fwkplugin.FactoryFunc(func(name string, _ json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return &mockDataProducerP{name: name, produces: map[string]any{keyA: nil}}, nil
	})

	// A DataProducer that produces keyB.
	producerTypeB := "producer-b"
	producerBFactory := fwkplugin.FactoryFunc(func(name string, _ json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return &mockDataProducerP{name: name, produces: map[string]any{keyB: nil}}, nil
	})

	// A non-ProducerPlugin registry entry (e.g. a scheduling scorer).
	nonProducerType := "non-producer"
	nonProducerFactory := fwkplugin.FactoryFunc(func(name string, _ json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return &MockSchedulingPlugin{consumes: map[string]any{keyA: nil}}, nil
	})

	// A factory that always fails.
	failingType := "failing"
	failingFactory := fwkplugin.FactoryFunc(func(name string, _ json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return nil, errors.New("requires params")
	})

	handle := fwkplugin.NewEppHandle(context.Background(), func() []k8stypes.NamespacedName { return nil })

	testCases := []struct {
		name                    string
		existingPlugins         []fwkplugin.Plugin
		defaultProducerRegistry map[string]string
		factoryRegistry         map[string]fwkplugin.FactoryFunc
		wantTypes               []string // TypedName.Type of expected auto-created producers
		wantErr                 bool
	}{
		{
			name: "creates producer for missing consumed key",
			existingPlugins: []fwkplugin.Plugin{
				&MockSchedulingPlugin{consumes: map[string]any{keyA: nil}},
			},
			defaultProducerRegistry: map[string]string{keyA: producerTypeA},
			factoryRegistry:         map[string]fwkplugin.FactoryFunc{producerTypeA: producerAFactory},
			wantTypes:               []string{producerTypeA},
		},
		{
			name: "no missing keys - nothing created",
			existingPlugins: []fwkplugin.Plugin{
				&mockDataProducerP{name: "existing-a", produces: map[string]any{keyA: nil}},
				&MockSchedulingPlugin{consumes: map[string]any{keyA: nil}},
			},
			defaultProducerRegistry: map[string]string{keyA: producerTypeA},
			factoryRegistry:         map[string]fwkplugin.FactoryFunc{producerTypeA: producerAFactory},
			wantTypes:               nil,
		},
		{
			name: "producer already present by type - not duplicated",
			existingPlugins: []fwkplugin.Plugin{
				// Simulate a plugin whose type matches the registry key.
				&typedMockPlugin{typeName: producerTypeA, produces: map[string]any{keyA: nil}},
				&MockSchedulingPlugin{consumes: map[string]any{keyA: nil}},
			},
			defaultProducerRegistry: map[string]string{keyA: producerTypeA},
			factoryRegistry:         map[string]fwkplugin.FactoryFunc{producerTypeA: producerAFactory},
			wantTypes:               nil,
		},
		{
			name: "failing factory returns error",
			existingPlugins: []fwkplugin.Plugin{
				&MockSchedulingPlugin{consumes: map[string]any{keyA: nil}},
			},
			defaultProducerRegistry: map[string]string{keyA: failingType},
			factoryRegistry:         map[string]fwkplugin.FactoryFunc{failingType: failingFactory},
			wantErr:                 true,
		},
		{
			name: "non-ProducerPlugin registry entry is skipped",
			existingPlugins: []fwkplugin.Plugin{
				&MockSchedulingPlugin{consumes: map[string]any{keyA: nil}},
			},
			defaultProducerRegistry: map[string]string{keyA: nonProducerType},
			factoryRegistry:         map[string]fwkplugin.FactoryFunc{nonProducerType: nonProducerFactory},
			wantTypes:               nil,
		},
		{
			name: "only relevant producer is created among multiple registry entries",
			existingPlugins: []fwkplugin.Plugin{
				&MockSchedulingPlugin{consumes: map[string]any{keyA: nil}},
			},
			defaultProducerRegistry: map[string]string{keyA: producerTypeA, keyB: producerTypeB},
			factoryRegistry: map[string]fwkplugin.FactoryFunc{
				producerTypeA: producerAFactory,
				producerTypeB: producerBFactory,
			},
			wantTypes: []string{producerTypeA},
		},
		{
			name:                    "no consumers - nothing created",
			existingPlugins:         []fwkplugin.Plugin{},
			defaultProducerRegistry: map[string]string{keyA: producerTypeA},
			factoryRegistry:         map[string]fwkplugin.FactoryFunc{producerTypeA: producerAFactory},
			wantTypes:               nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := CreateMissingDataProducers(tc.existingPlugins, tc.defaultProducerRegistry, tc.factoryRegistry, handle)

			if tc.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)

			// The auto-created plugin is named after its registry type (the pluginType
			// passed to the factory), so we compare by name.
			gotNames := make([]string, 0, len(result))
			for _, p := range result {
				gotNames = append(gotNames, p.TypedName().Name)
			}

			assert.ElementsMatch(t, tc.wantTypes, gotNames)
		})
	}
}

func assertTopologicalOrder(t *testing.T, dag map[string][]string, ordered []string) {
	t.Helper()
	positions := make(map[string]int)
	for i, p := range ordered {
		positions[p] = i
	}

	for node, dependencies := range dag {
		for _, dep := range dependencies {
			assert.Less(t, positions[dep], positions[node], "Dependency %s should come before %s", dep, node)
		}
	}
}
