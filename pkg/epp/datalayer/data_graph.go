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
	"errors"
	"fmt"
	"reflect"
	"slices"

	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// ValidateAndOrderDataDependencies validates that the data dependencies among the given plugins are acyclic
// and returns a topologically sorted order of plugin names based on their data dependencies.
// Further, it validates that the plugins are ordered in a way that respects the layer execution order.
func ValidateAndOrderDataDependencies(plugins []plugin.Plugin) ([]string, error) {
	pluginMap := make(map[string]plugin.Plugin)
	for _, p := range plugins {
		pluginMap[p.TypedName().String()] = p
	}
	producers := make(map[string]plugin.ProducerPlugin)
	consumers := make(map[string]plugin.ConsumerPlugin)
	for name, p := range pluginMap {
		if producer, ok := p.(plugin.ProducerPlugin); ok {
			producers[name] = producer
		}
		if consumer, ok := p.(plugin.ConsumerPlugin); ok {
			consumers[name] = consumer
		}
	}
	dag, err := buildDAG(producers, consumers)
	if err != nil {
		return nil, err
	}
	// Topologically sort the DAG to determine the order of plugin execution.
	pluginNames, err := topologicalSort(dag)
	if err != nil {
		return nil, err
	}

	return pluginNames, nil
}

// CreateMissingDataProducers inspects the set of already-configured plugins,
// finds data keys that are consumed but not yet produced, and auto-instantiates
// the default DataProducer plugin for each such key using nil parameters.
// defaultProducerRegistry maps a data key to the plugin type that is its default producer.
// factoryRegistry maps a plugin type to its factory function.
// Only entries whose type is not already present in plugins are considered.
func CreateMissingDataProducers(plugins []plugin.Plugin, defaultProducerRegistry map[string]string, factoryRegistry map[string]plugin.FactoryFunc, handle plugin.Handle) ([]plugin.Plugin, error) {
	// Collect plugin types already present so we don't create duplicates.
	existingTypes := make(map[string]bool)
	for _, p := range plugins {
		existingTypes[p.TypedName().Type] = true
	}

	// Collect all keys already produced by existing plugins.
	producedKeys := make(map[string]bool)
	for _, p := range plugins {
		if producer, ok := p.(plugin.ProducerPlugin); ok {
			for key := range producer.Produces() {
				producedKeys[key] = true
			}
		}
	}

	// Build the set of keys that are consumed but not yet produced.
	missingKeys := make(map[string]bool)
	for _, p := range plugins {
		if consumer, ok := p.(plugin.ConsumerPlugin); ok {
			for key := range consumer.Consumes() {
				if !producedKeys[key] {
					missingKeys[key] = true
				}
			}
		}
	}

	if len(missingKeys) == 0 {
		return nil, nil
	}

	// For each missing key, look up its default producer type and collect unique types to instantiate.
	// A single producer type may satisfy multiple missing keys; deduplicate by type.
	neededTypes := make(map[string]string)
	for key := range missingKeys {
		pluginType, ok := defaultProducerRegistry[key]
		if !ok || existingTypes[pluginType] {
			continue
		}
		neededTypes[pluginType] = key
	}

	var plgns []plugin.Plugin
	for pluginType, registeredKey := range neededTypes {
		factory, ok := factoryRegistry[pluginType]
		if !ok {
			continue
		}
		// pass nil params as this is default instantiation.
		candidate, err := factory(pluginType, nil, handle)
		if err != nil {
			return nil, fmt.Errorf("failed to instantiate data producer %q: %w", pluginType, err)
		}
		producer, ok := candidate.(plugin.ProducerPlugin)
		if !ok || existingTypes[pluginType] {
			continue
		}

		// Validate that the instantiated producer produces the registered key.
		if _, ok := producer.Produces()[registeredKey]; !ok {
			return nil, fmt.Errorf("instantiated default data producer %q does not produce registered key %q", pluginType, registeredKey)
		}

		plgns = append(plgns, candidate)
		existingTypes[pluginType] = true
	}

	return plgns, nil
}

// Define constants for layer execution order. Lower value means earlier execution.
const (
	FlowControlLayer    = 0
	RequestControlLayer = 1
	SchedulingLayer     = 2
	DefaultLayer        = -1 // For plugins that don't fit into a known layer
)

func pluginToLayerExecutionOrder(plugin plugin.Plugin) int {
	// Flow control plugins
	if _, ok := plugin.(fwkfc.FairnessPolicy); ok {
		return FlowControlLayer
	}
	if _, ok := plugin.(fwkfc.OrderingPolicy); ok {
		return FlowControlLayer
	}

	// Request control plugins
	if _, ok := plugin.(fwkrc.DataProducer); ok {
		return RequestControlLayer
	}
	if _, ok := plugin.(fwkrc.Admitter); ok {
		return RequestControlLayer
	}
	if _, ok := plugin.(fwkrc.PreRequest); ok {
		return RequestControlLayer
	}
	if _, ok := plugin.(fwkrc.ResponseHeaderProcessor); ok {
		return RequestControlLayer
	}

	// Scheduling plugins
	if _, ok := plugin.(fwksched.ProfileHandler); ok {
		return SchedulingLayer
	}
	if _, ok := plugin.(fwksched.Filter); ok {
		return SchedulingLayer
	}
	if _, ok := plugin.(fwksched.Scorer); ok {
		return SchedulingLayer
	}
	if _, ok := plugin.(fwksched.Picker); ok {
		return SchedulingLayer
	}

	// If the plugin doesn't match any known layer, return -1.
	return DefaultLayer
}

// buildDAG builds a dependency graph among data preparation plugins based on their
// produced and consumed data keys.
func buildDAG(producers map[string]plugin.ProducerPlugin, consumers map[string]plugin.ConsumerPlugin) (map[string][]string, error) {
	dag := make(map[string][]string)
	// Create dependency graph as a DAG.
	for _, producer := range producers {
		dag[producer.TypedName().String()] = []string{}
	}
	for _, consumer := range consumers {
		dag[consumer.TypedName().String()] = []string{}
	}
	for pName, producer := range producers {
		for cName, consumer := range consumers {
			if pName == cName {
				continue
			}
			if producer.Produces() != nil && consumer.Consumes() != nil {
				for producedKey, producedData := range producer.Produces() {
					if consumedData, ok := consumer.Consumes()[producedKey]; ok {
						// Check types are same.
						if reflect.TypeOf(producedData) != reflect.TypeOf(consumedData) {
							return nil, errors.New("data type mismatch between produced and consumed data for key: " + producedKey)
						}
						if pluginToLayerExecutionOrder(producer) > pluginToLayerExecutionOrder(consumer) {
							return nil, errors.New("invalid plugin layer execution order: producer " + pName + " needs to be executed before consumer " + cName)
						}
						// Consumer depends on producer, so add an edge from consumer to producer.
						dag[cName] = append(dag[cName], pName)
						break
					}
				}
			}
		}
	}
	return dag, nil
}

// TopologicalSort performs Kahn's Algorithm on a DAG.
// It returns the sorted order or an error if a cycle is detected.
func topologicalSort(graph map[string][]string) ([]string, error) {
	// 1. Initialize in-degree map
	inDegree := make(map[string]int)

	// Ensure all nodes are present in the inDegree map, even those with no dependencies
	for u, neighbors := range graph {
		if _, ok := inDegree[u]; !ok {
			inDegree[u] = 0
		}
		for _, v := range neighbors {
			inDegree[v]++ // Increment in-degree for the destination node
		}
	}

	// 2. Initialize the queue with nodes having 0 in-degree
	var queue []string
	for node, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, node)
		}
	}

	var result []string

	// 3. Process the queue
	for len(queue) > 0 {
		// Dequeue
		u := queue[0]
		queue = queue[1:]

		result = append(result, u)

		// Decrease in-degree of neighbors
		if neighbors, ok := graph[u]; ok {
			for _, v := range neighbors {
				inDegree[v]--
				if inDegree[v] == 0 {
					queue = append(queue, v)
				}
			}
		}
	}

	// 4. Check for cycles
	// If the result size != total nodes, there is a cycle
	if len(result) != len(inDegree) {
		return nil, errors.New("cycle detected: graph is not a DAG")
	}

	// Reverse to get the correct order since edges point from consumer to producer
	slices.Reverse(result)
	return result, nil
}
