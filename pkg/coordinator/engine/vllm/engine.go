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

// Package vllm prepares coordinator requests for the vLLM inference protocol.
package vllm

import (
	"github.com/llm-d/llm-d-router/pkg/coordinator/engine"
)

// Engine implements vLLM request preparation and response parsing.
type Engine struct {
	useOpenAIFormat bool
	limits          engine.Limits
}

// New constructs a vLLM engine using the configured wire format and input limits.
func New(useOpenAIFormat bool, limits engine.Limits) Engine {
	return Engine{useOpenAIFormat: useOpenAIFormat, limits: limits}
}
