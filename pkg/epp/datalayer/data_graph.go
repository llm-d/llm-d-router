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
	"errors"
	"fmt"
	"reflect"

	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/util"
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
	pluginNames, err := util.TopologicalSort(dag)
	if err != nil {
		return nil, err
	}

	return pluginNames, nil
}

// CreateMissingDataProducers inspects the configured plugins, finds data keys
// that are consumed (Required) but not produced, and auto-instantiates the
// default DataProducer for each using nil parameters. Resolution is transitive:
// a producer created here may itself consume keys with registered defaults, so
// creation repeats until no required key is missing or no further producer can
// be added. defaultProducerRegistry maps a data key to its default producer type;
// factoryRegistry maps a plugin type to its factory function.
func CreateMissingDataProducers(ctx context.Context, defaultProducerRegistry map[string]string, factoryRegistry map[string]plugin.FactoryFunc, handle plugin.Handle) error {
	logger := log.FromContext(ctx)

	for {
		producedKeys := producedKeySet(handle)

		// Build the set of keys that are consumed but not yet produced.
		missingKeys := make(map[string]string)
		for _, p := range handle.GetAllPlugins() {
			if consumer, ok := p.(plugin.ConsumerPlugin); ok {
				for key := range consumer.Consumes().Required {
					if !producedKeys[key.String()] {
						missingKeys[key.String()] = consumer.TypedName().Name
					}
				}
			}
		}
		if len(missingKeys) == 0 {
			break
		}

		created := 0
		for key, consumerName := range missingKeys {
			defaultProducerNameOrType, ok := defaultProducerRegistry[key]
			if !ok {
				return fmt.Errorf("no default producer found for missing data key: %v, which is consumed by: %v", key, consumerName)
			}
			if handle.Plugin(defaultProducerNameOrType) != nil {
				// Already created. This can happen when a producer produces multiple data keys.
				continue
			}
			factory, ok := factoryRegistry[defaultProducerNameOrType]
			if !ok {
				return fmt.Errorf("factory not found for default producer: %v, this is required by datakey: %v, which is consumed by: %v", defaultProducerNameOrType, key, consumerName)
			}
			// pass nil params as this is default instantiation.
			plg, err := factory(defaultProducerNameOrType, nil, handle)
			if err != nil {
				return fmt.Errorf("failed to instantiate data producer %q: %w, this is required by datakey: %v, which is consumed by: %v", defaultProducerNameOrType, err, key, consumerName)
			}
			if _, ok := plg.(plugin.ProducerPlugin); !ok {
				return fmt.Errorf("auto-created default entry %q is not a ProducerPlugin, this is required by datakey: %v, which is consumed by: %v", defaultProducerNameOrType, key, consumerName)
			}
			handle.AddPlugin(plg.TypedName().Name, plg)
			logger.Info("auto-created default producer",
				"producer", plg.TypedName().String(),
				"dataKey", key,
				"consumer", consumerName)
			created++
		}
		// No progress despite missing keys (every default already present): stop
		// to avoid looping on a producer-name mismatch.
		if created == 0 {
			break
		}
	}

	// Warn about optional keys that still lack a producer — no error.
	producedKeys := producedKeySet(handle)
	for _, p := range handle.GetAllPlugins() {
		if consumer, ok := p.(plugin.ConsumerPlugin); ok {
			for key := range consumer.Consumes().Optional {
				if !producedKeys[key.String()] {
					logger.Info("Warning: optional data key has no producer, plugin will use fallback",
						"plugin", p.TypedName().Name, "dataKey", key.String())
				}
			}
		}
	}

	return nil
}

// producedKeySet returns the set of data-key strings produced by the plugins
// currently registered on the handle.
func producedKeySet(handle plugin.Handle) map[string]bool {
	producedKeys := make(map[string]bool)
	for _, p := range handle.GetAllPlugins() {
		if producer, ok := p.(plugin.ProducerPlugin); ok {
			for key := range producer.Produces() {
				producedKeys[key.String()] = true
			}
		}
	}
	return producedKeys
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
			dependencies := consumer.Consumes()
			if producer.Produces() == nil {
				continue
			}
			for producedKey, producedData := range producer.Produces() {
				consumedData, ok := dependencies.Required[producedKey]
				optional := false
				if !ok {
					consumedData, ok = dependencies.Optional[producedKey]
					optional = ok
				}
				if !ok {
					continue
				}
				// Check types are same.
				if reflect.TypeOf(producedData) != reflect.TypeOf(consumedData) {
					if optional {
						return nil, errors.New("data type mismatch between produced and optionally consumed data for key: " + producedKey.String())
					}
					return nil, errors.New("data type mismatch between produced and consumed data for key: " + producedKey.String())
				}
				if pluginToLayerExecutionOrder(producer) > pluginToLayerExecutionOrder(consumer) {
					return nil, errors.New("invalid plugin layer execution order: producer " + pName + " needs to be executed before consumer " + cName)
				}
				// Consumer depends on producer; optional dependencies only add
				// an edge when the producer is already configured.
				dag[cName] = append(dag[cName], pName)
				break
			}
		}
	}
	return dag, nil
}
