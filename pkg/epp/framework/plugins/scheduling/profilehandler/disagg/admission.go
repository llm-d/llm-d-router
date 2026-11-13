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

package disagg

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

// FilterForAdmission checks the built-in P/D routes using prepared cost data.
// Profiles with custom role selection and unknown deciders bypass this check.
func (h *Handler) FilterForAdmission(ctx context.Context, request *scheduling.InferenceRequest,
	profiles map[string]scheduling.SchedulerProfile, endpoints []scheduling.Endpoint,
	filter func(string, []scheduling.Endpoint) []scheduling.Endpoint,
) (bool, map[string][]scheduling.Endpoint) {
	if !hasAdmissionRoleFilter(profiles[h.decodeProfile], bylabel.DecodeRoleType) {
		return true, nil
	}
	_, hasPrefill := profiles[h.prefillProfile]
	usePrefill := hasPrefill && (h.stageOrder == StageOrderPrefillFirst || h.pdDecider != nil)
	if usePrefill && !hasAdmissionRoleFilter(profiles[h.prefillProfile], bylabel.PrefillRoleType) {
		return true, nil
	}
	if usePrefill && h.stageOrder != StageOrderPrefillFirst {
		switch h.pdDecider.(type) {
		case *AlwaysDisaggPDDecider, *PrefixBasedPDDecider:
		default:
			return true, nil
		}
	}

	decode := bylabel.NewDecodeRole().Filter(ctx, request, endpoints)
	prefill := bylabel.NewPrefillRole().Filter(ctx, request, endpoints)
	// Missing workers remain scheduling errors, distinct from exhausted capacity.
	if len(decode) == 0 || (usePrefill && len(prefill) == 0) {
		return true, nil
	}
	decodeFit := filter(flowcontrol.SaturationStageDecode, decode)
	if len(decodeFit) == 0 {
		return false, nil
	}
	if !usePrefill {
		return true, map[string][]scheduling.Endpoint{h.decodeProfile: decodeFit}
	}

	prefillFit := filter(flowcontrol.SaturationStagePrefill, prefill)
	if len(prefillFit) > 0 {
		return true, map[string][]scheduling.Endpoint{
			h.decodeProfile:  decodeFit,
			h.prefillProfile: prefillFit,
		}
	}

	if decider, ok := h.pdDecider.(*PrefixBasedPDDecider); ok && h.stageOrder != StageOrderPrefillFirst {
		decodeOnly := make([]scheduling.Endpoint, 0, len(decodeFit))
		for _, endpoint := range decodeFit {
			// Match disaggregate's soft failure without caching a decision for a
			// decoder that the scheduler has not selected.
			needsPrefill, err := decider.computeNeedsRemotePrefill(ctx, request, endpoint)
			if err != nil || !needsPrefill {
				decodeOnly = append(decodeOnly, endpoint)
			}
		}
		if len(decodeOnly) > 0 {
			return true, map[string][]scheduling.Endpoint{h.decodeProfile: decodeOnly}
		}
	}
	return false, nil
}

func hasAdmissionRoleFilter(profile scheduling.SchedulerProfile, roleType string) bool {
	profileRoles, ok := profile.(interface{ HasRoleFilter(string) bool })
	return ok && profileRoles.HasRoleFilter(roleType)
}
