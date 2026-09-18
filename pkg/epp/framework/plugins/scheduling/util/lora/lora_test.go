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

package lora

import (
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

var gpu = fwkdl.LoraLoadState{Level: fwkdl.LoraLoadLevelGPU}

func endpoint(name string, m *fwkdl.Metrics) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}}, m, nil)
}

func TestIsBaseModelRequest(t *testing.T) {
	eps := []fwksched.Endpoint{endpoint("a", &fwkdl.Metrics{}), endpoint("b", &fwkdl.Metrics{BaseModel: "base"})}
	assert.True(t, IsBaseModelRequest(&fwksched.InferenceRequest{TargetModel: "base"}, eps))
	assert.False(t, IsBaseModelRequest(&fwksched.InferenceRequest{TargetModel: "adapter"}, eps))
	assert.False(t, IsBaseModelRequest(&fwksched.InferenceRequest{TargetModel: ""}, eps), "an unreported base model never matches")
	assert.False(t, IsBaseModelRequest(nil, eps))
}

func TestBusyAndIdleGPUAdapters(t *testing.T) {
	m := &fwkdl.Metrics{
		LoadedModels: map[string]fwkdl.LoraLoadState{
			"busy":   gpu,
			"idle":   gpu,
			"pinned": {Level: fwkdl.LoraLoadLevelGPU, Pinned: true},
			"host":   {Level: fwkdl.LoraLoadLevelCPU},
		},
		ActiveModels: map[string]int{"busy": 1, "host": 1},
	}
	assert.Equal(t, 1, BusyGPUAdapters(m), "only GPU residents with requests in flight")
	assert.True(t, HasIdleGPUAdapter(m))

	m.ActiveModels["idle"] = 1
	assert.Equal(t, 2, BusyGPUAdapters(m))
	assert.False(t, HasIdleGPUAdapter(m), "the pinned resident is not evictable")
}
