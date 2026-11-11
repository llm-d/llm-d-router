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

package requestcontrol

import (
	"strings"

	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// sheddableAdmissionPriority is the negative band used when the priority class is
// sheddable. It matches the default negative flow-control band: any priority below
// zero is sheddable, and every over-quota request shares this one band.
const sheddableAdmissionPriority = -1

// parsedPriorityClass is the priority-class hint after header normalization.
// class is always critical or sheddable. invalid means the header was present
// but empty or unrecognized, so class was forced to critical.
type parsedPriorityClass struct {
	class   string
	invalid bool
}

// admissionPriority applies the priority class to the InferenceObjective priority.
// Sheddable replaces that priority with the negative band. Critical, missing, and
// invalid values leave it unchanged, including when the objective is already negative.
func admissionPriority(objectivePriority int, class parsedPriorityClass) int {
	if class.class == metadata.PriorityClassSheddable {
		return sheddableAdmissionPriority
	}
	return objectivePriority
}

// parsePriorityClass reads x-llm-d-inference-priority-class.
// A missing header is critical. An empty or unrecognized value is critical and invalid.
// The raw header value is not retained.
func parsePriorityClass(headers map[string]string) parsedPriorityClass {
	value, ok := metadata.GetLowerCaseHeaderValue(headers, metadata.PriorityClassKey)
	if !ok {
		return parsedPriorityClass{class: metadata.PriorityClassCritical}
	}
	switch strings.ToLower(strings.TrimSpace(value)) {
	case metadata.PriorityClassCritical:
		return parsedPriorityClass{class: metadata.PriorityClassCritical}
	case metadata.PriorityClassSheddable:
		return parsedPriorityClass{class: metadata.PriorityClassSheddable}
	default:
		return parsedPriorityClass{class: metadata.PriorityClassCritical, invalid: true}
	}
}
