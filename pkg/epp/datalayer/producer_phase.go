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

package datalayer

import (
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
)

// ResolveDataProducerPhases assigns every DataProducer among plugins to a
// phase and returns the phases keyed by plugin name. A producer listed in
// explicit keeps that phase. Any other producer runs in
// DataProducerPhasePreScreen when a screener or a PreScreen producer consumes
// a key it produces, Required or Optional, directly or through other
// producers, and in DataProducerPhaseScheduling otherwise.
//
// An explicit DataProducerPhaseScheduling is never overridden; a screener that
// consumes such a producer's key fails ValidateAndOrderDataDependencies.
func ResolveDataProducerPhases(plugins []plugin.Plugin, explicit map[string]fwkrc.DataProducerPhase) map[string]fwkrc.DataProducerPhase {
	producersByKey := map[plugin.DataKey][]fwkrc.DataProducer{}
	var producers []fwkrc.DataProducer
	for _, p := range plugins {
		producer, ok := p.(fwkrc.DataProducer)
		if !ok {
			continue
		}
		producers = append(producers, producer)
		for key := range producer.Produces() {
			producersByKey[key] = append(producersByKey[key], producer)
		}
	}

	phases := make(map[string]fwkrc.DataProducerPhase, len(producers))
	for _, producer := range producers {
		if phase, ok := explicit[producer.TypedName().Name]; ok {
			phases[producer.TypedName().Name] = phase
		}
	}

	var pending []plugin.Plugin
	for _, p := range plugins {
		if _, ok := p.(fwkrc.Screener); ok {
			pending = append(pending, p)
		}
	}
	for _, producer := range producers {
		if phases[producer.TypedName().Name] == fwkrc.DataProducerPhasePreScreen {
			pending = append(pending, producer)
		}
	}
	for len(pending) > 0 {
		next := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		consumer, ok := next.(plugin.ConsumerPlugin)
		if !ok {
			continue
		}
		deps := consumer.Consumes()
		for _, keys := range []map[plugin.DataKey]any{deps.Required, deps.Optional} {
			for key := range keys {
				for _, producer := range producersByKey[key] {
					name := producer.TypedName().Name
					if _, assigned := phases[name]; assigned {
						continue
					}
					phases[name] = fwkrc.DataProducerPhasePreScreen
					pending = append(pending, producer)
				}
			}
		}
	}

	for _, producer := range producers {
		if _, assigned := phases[producer.TypedName().Name]; !assigned {
			phases[producer.TypedName().Name] = fwkrc.DataProducerPhaseScheduling
		}
	}
	return phases
}
