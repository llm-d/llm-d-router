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

package pipeline

import "fmt"

var registry = map[string]StepFactory{}

// Register adds a step factory to the global registry.
func Register(typeName string, factory StepFactory) {
	registry[typeName] = factory
}

// Build instantiates a step by type name and parameters.
func Build(typeName string, params map[string]any) (Step, error) {
	factory, ok := registry[typeName]
	if !ok {
		return nil, fmt.Errorf("unknown step type: %s", typeName)
	}
	return factory(params)
}
