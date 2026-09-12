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

// Package lora holds the residency helpers shared by the LoRA-aware
// scheduling plugins.
package lora

import (
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// IsBaseModelRequest reports whether the request targets the served base
// model rather than an adapter, judged by the model name the endpoints stamp
// on their metrics.
func IsBaseModelRequest(request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) bool {
	if request == nil {
		return false
	}
	for _, endpoint := range endpoints {
		if base := endpoint.GetMetrics().BaseModel; base != "" && base == request.TargetModel {
			return true
		}
	}
	return false
}

// BusyGPUAdapters counts GPU-resident adapters with a request in flight.
func BusyGPUAdapters(m *fwkdl.Metrics) int {
	busy := 0
	for name, state := range m.LoadedModels {
		if state.Level != fwkdl.LoraLoadLevelGPU {
			continue
		}
		if _, active := m.ActiveModels[name]; active {
			busy++
		}
	}
	return busy
}

// HasIdleGPUAdapter reports whether an unpinned GPU-resident adapter has no
// request in flight, so the model server can evict it without stalling anyone.
func HasIdleGPUAdapter(m *fwkdl.Metrics) bool {
	for name, state := range m.LoadedModels {
		if state.Level != fwkdl.LoraLoadLevelGPU || state.Pinned {
			continue
		}
		if _, active := m.ActiveModels[name]; !active {
			return true
		}
	}
	return false
}
