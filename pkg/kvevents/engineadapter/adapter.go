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

// Package engineadapter provides engine-specific implementations of the
// kvevents.EngineAdapter interface. Each inference engine (e.g. vLLM, SGLang)
// has its own adapter that knows how to parse raw transport messages into
// domain events.
package engineadapter

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

const (
	// EngineTypeVLLM selects the vLLM adapter.
	EngineTypeVLLM = "vllm"
	// EngineTypeSGLang selects the SGLang adapter.
	EngineTypeSGLang = "sglang"
)

// Factory returns a new EngineAdapter.
type Factory func() kvevents.EngineAdapter

var registry = map[string]Factory{}

// Register makes an adapter available to NewAdapter under engineType, so an
// engine with its own KV-event format registers its adapter from an init
// function in its own package instead of patching this one. A later
// registration under the same engine type replaces the earlier one. Register
// is not safe for concurrent use with NewAdapter.
func Register(engineType string, factory Factory) {
	registry[engineType] = factory
}

// engineTypes returns the registered engine types, sorted.
func engineTypes() []string {
	return slices.Sorted(maps.Keys(registry))
}

// NewAdapter creates the EngineAdapter registered for engineType. An empty
// type selects vLLM.
func NewAdapter(engineType string) (kvevents.EngineAdapter, error) {
	if engineType == "" {
		engineType = EngineTypeVLLM
	}
	factory, ok := registry[engineType]
	if !ok {
		types := engineTypes()
		for i, t := range types {
			types[i] = strconv.Quote(t)
		}
		return nil, fmt.Errorf("unsupported engine type: %q (supported: %s)", engineType, strings.Join(types, ", "))
	}
	return factory(), nil
}
