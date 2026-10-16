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

package scheduling

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/headerprofile"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
)

func TestFilterForAdmissionSingleProfile(t *testing.T) {
	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": NewSchedulerProfile()}))
	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, nil, nil)
	for _, fits := range []bool{false, true} {
		ready, admitted := scheduler.FilterForAdmission(t.Context(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{endpoint},
			func(stage string, candidates []fwksched.Endpoint) []fwksched.Endpoint {
				require.Equal(t, flowcontrol.SaturationStageDecode, stage)
				if fits {
					return candidates
				}
				return nil
			})
		require.Equal(t, fits, ready)
		require.Len(t, admitted, map[bool]int{false: 0, true: 1}[fits])
		if fits {
			require.Equal(t, []fwksched.Endpoint{endpoint}, admitted["default"])
		}
	}
}

func TestFilterForAdmissionDisaggUsesCanonicalProfiles(t *testing.T) {
	decode := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Labels: map[string]string{bylabel.RoleLabel: bylabel.RoleDecode}}, nil, nil)
	prefill := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Labels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefill}}, nil, nil)
	handler := disagg.NewDisaggProfileHandler("d", "p", "", nil, nil).WithStageOrder(disagg.StageOrderPrefillFirst)
	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(handler, map[string]fwksched.SchedulerProfile{
		"d": NewSchedulerProfile().WithFilters(bylabel.NewDecodeRole()),
		"p": NewSchedulerProfile().WithFilters(bylabel.NewPrefillRole()),
	}))
	ready, _ := scheduler.FilterForAdmission(t.Context(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{decode, prefill},
		func(stage string, candidates []fwksched.Endpoint) []fwksched.Endpoint {
			if stage == flowcontrol.SaturationStagePrefill {
				return nil
			}
			return candidates
		})
	require.False(t, ready)
}

func TestFilterForAdmissionUnknownHandlerBypasses(t *testing.T) {
	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(headerprofile.NewHeaderProfileHandler("", ""),
		map[string]fwksched.SchedulerProfile{"decode": NewSchedulerProfile()}))
	endpoints := []fwksched.Endpoint{fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, nil, nil)}
	ready, admitted := scheduler.FilterForAdmission(t.Context(), &fwksched.InferenceRequest{}, endpoints,
		func(string, []fwksched.Endpoint) []fwksched.Endpoint {
			t.Fatal("unexpected capacity evaluation")
			return nil
		})
	require.True(t, ready)
	require.Nil(t, admitted)
}

type admissionRecordingProfile struct {
	calls      int
	candidates []fwksched.Endpoint
}

func (p *admissionRecordingProfile) Run(_ context.Context, _ *fwksched.InferenceRequest, candidates []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	p.calls++
	p.candidates = candidates
	return &fwksched.ProfileRunResult{TargetEndpoints: candidates}, nil
}

func TestScheduleWithAdmissionPreservesStageCandidates(t *testing.T) {
	shared := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Labels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefillDecode}}, nil, nil)
	prefill := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Labels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefill}}, nil, nil)
	decodeProfile, prefillProfile := &admissionRecordingProfile{}, &admissionRecordingProfile{}
	handler := disagg.NewDisaggProfileHandler("d", "p", "", nil, nil).WithStageOrder(disagg.StageOrderPrefillFirst)
	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(handler, map[string]fwksched.SchedulerProfile{
		"d": decodeProfile,
		"p": prefillProfile,
	}))
	endpoints := []fwksched.Endpoint{shared, prefill}
	result, err := scheduler.ScheduleWithAdmission(t.Context(), &fwksched.InferenceRequest{}, endpoints,
		map[string][]fwksched.Endpoint{"d": {shared}, "p": {prefill}})
	require.NoError(t, err)
	require.Equal(t, []fwksched.Endpoint{shared}, result.ProfileResults["d"].TargetEndpoints)
	require.Equal(t, []fwksched.Endpoint{prefill}, result.ProfileResults["p"].TargetEndpoints)
	require.Equal(t, 1, decodeProfile.calls)
	require.Equal(t, 1, prefillProfile.calls)
	require.Equal(t, []fwksched.Endpoint{shared, prefill}, endpoints)
}

func TestScheduleWithAdmissionUnmaskedProfileUsesAllCandidates(t *testing.T) {
	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, nil, nil)
	profile := &admissionRecordingProfile{}
	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": profile}))
	endpoints := []fwksched.Endpoint{endpoint}
	_, err := scheduler.ScheduleWithAdmission(t.Context(), &fwksched.InferenceRequest{}, endpoints,
		map[string][]fwksched.Endpoint{"another-profile": nil})
	require.NoError(t, err)
	require.Equal(t, endpoints, profile.candidates)
	require.Equal(t, 1, profile.calls)
}

func TestScheduleWithAdmissionEmptyMaskKeepsNoCandidates(t *testing.T) {
	scheduler := NewSchedulerWithConfig(NewSchedulerConfig(single.NewSingleProfileHandler(),
		map[string]fwksched.SchedulerProfile{"default": NewSchedulerProfile()}))
	endpoints := []fwksched.Endpoint{fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, nil, nil)}
	_, err := scheduler.ScheduleWithAdmission(t.Context(), &fwksched.InferenceRequest{}, endpoints,
		map[string][]fwksched.Endpoint{"default": nil})
	require.ErrorContains(t, err, "no endpoints available for the given request")
}
