/*
Copyright 2026 The Kubernetes Authors.

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

package diffusion

import (
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	diffusionloadconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/diffusionload/constants"
)

// DiffusionLoadDataKey carries the per-endpoint in-flight diffusion cost
// snapshot used by cost-aware scorers. Published per request by
// DiffusionLoadProducer.Produce onto each candidate endpoint.
var DiffusionLoadDataKey = plugin.NewDataKey("DiffusionLoadDataKey", diffusionloadconstants.DiffusionLoadProducerType)

// DiffusionLoad captures the outstanding declared diffusion work an endpoint
// has committed to, as tracked by the EPP.
type DiffusionLoad struct {
	// CostUnits is the sum of the declared cost of in-flight image generation
	// requests routed to this endpoint, in step-megapixel units
	// (inference steps x output megapixels x image count). Updated by
	// PreRequest (when an endpoint is chosen) and released when the request's
	// response stream ends.
	CostUnits int64
}

// Clone returns an independent copy of the DiffusionLoad.
func (l *DiffusionLoad) Clone() fwkdl.Cloneable {
	if l == nil {
		return nil
	}
	cp := *l
	return &cp
}
