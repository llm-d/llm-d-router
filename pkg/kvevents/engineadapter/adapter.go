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

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

const (
	// EngineTypeVLLM selects the vLLM adapter.
	EngineTypeVLLM = "vllm"
	// EngineTypeSGLang selects the SGLang adapter.
	EngineTypeSGLang = "sglang"
)

// NewAdapter creates an EngineAdapter for the given engine type.
// Supported types: "vllm" (default), "sglang".
func NewAdapter(engineType string) (kvevents.EngineAdapter, error) {
	switch engineType {
	case EngineTypeVLLM, "":
		return NewVLLMAdapter(), nil
	case EngineTypeSGLang:
		return NewSGLangAdapter(), nil
	default:
		return nil, fmt.Errorf("unsupported engine type: %q (supported: %q, %q)", engineType, EngineTypeVLLM, EngineTypeSGLang)
	}
}
