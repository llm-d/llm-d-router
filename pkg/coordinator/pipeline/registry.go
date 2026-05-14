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
