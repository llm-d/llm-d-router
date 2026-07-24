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

package gpu

import (
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	// DCGMExtractorType is the plugin type for the DCGM metrics extractor.
	DCGMExtractorType = "dcgm-extractor"
)

// GPUUtilizationDataKey identifies per-endpoint GPU utilization published
// by the DCGM extractor and consumed by GPU-aware filters and scorers.
var GPUUtilizationDataKey = plugin.NewDataKey("GPUUtilization", DCGMExtractorType)

// GPUUtilization is a normalized GPU compute utilization value in [0.0, 1.0],
// derived from DCGM_FI_DEV_GPU_UTIL (which reports 0-100).
// For multi-GPU pods the extractor aggregates across visible devices.
type GPUUtilization float64

func (v GPUUtilization) Clone() fwkdl.Cloneable {
	return v
}

// ReadGPUUtilization retrieves GPU utilization from an endpoint's AttributeMap.
func ReadGPUUtilization(attrs fwkdl.AttributeMap, key string) (GPUUtilization, bool) {
	return fwkdl.ReadAttribute[GPUUtilization](attrs, key)
}
