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
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

type admissionProfile struct{ role string }

func (p admissionProfile) HasRoleFilter(role string) bool { return p.role == role }

func (admissionProfile) Run(context.Context, *scheduling.InferenceRequest, []scheduling.Endpoint) (*scheduling.ProfileRunResult, error) {
	panic("admission must not run scheduler profiles")
}

func admissionProfiles() map[string]scheduling.SchedulerProfile {
	return map[string]scheduling.SchedulerProfile{
		"decode":  admissionProfile{bylabel.DecodeRoleType},
		"prefill": admissionProfile{bylabel.PrefillRoleType},
	}
}

func admissionEndpoint(role string, cached int) scheduling.Endpoint {
	endpoint := makeTestEndpoint(cached)
	endpoint.GetMetadata().Labels = map[string]string{bylabel.RoleLabel: role}
	return endpoint
}

func noPrefillCapacity(stage string, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	if stage == flowcontrol.SaturationStagePrefill {
		return nil
	}
	return endpoints
}

func TestAdmissionRequiredPrefillWaits(t *testing.T) {
	endpoints := []scheduling.Endpoint{admissionEndpoint(bylabel.RoleDecode, 0), admissionEndpoint(bylabel.RolePrefill, 0)}
	for _, prefillFirst := range []bool{false, true} {
		t.Run(map[bool]string{false: "always_decider", true: "prefill_first"}[prefillFirst], func(t *testing.T) {
			handler := NewDisaggProfileHandler("decode", "prefill", "", newAlwaysDisaggPDDecider(), nil)
			if prefillFirst {
				handler.WithStageOrder(StageOrderPrefillFirst)
			}
			ready, _ := handler.FilterForAdmission(t.Context(), makeRequestWithTokens(10), admissionProfiles(), endpoints, noPrefillCapacity)
			require.False(t, ready)
			ready, admitted := handler.FilterForAdmission(t.Context(), makeRequestWithTokens(10), admissionProfiles(), endpoints,
				func(_ string, candidates []scheduling.Endpoint) []scheduling.Endpoint { return candidates })
			require.True(t, ready)
			require.Equal(t, map[string][]scheduling.Endpoint{"decode": {endpoints[0]}, "prefill": {endpoints[1]}}, admitted)
		})
	}
}

func TestAdmissionOptionalPrefillDoesNotWait(t *testing.T) {
	endpoints := []scheduling.Endpoint{admissionEndpoint(bylabel.RoleDecode, 0), admissionEndpoint(bylabel.RolePrefill, 0)}
	handler := NewDisaggProfileHandler("decode", "prefill", "", nil, nil)
	ready, admitted := handler.FilterForAdmission(t.Context(), makeRequestWithTokens(10), admissionProfiles(), endpoints, noPrefillCapacity)
	require.True(t, ready)
	require.Equal(t, map[string][]scheduling.Endpoint{"decode": {endpoints[0]}}, admitted)
}

func TestAdmissionPrefixFallbackDoesNotMemoize(t *testing.T) {
	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 5})
	require.NoError(t, err)
	handler := NewDisaggProfileHandler("decode", "prefill", "", decider, nil)
	uncached := admissionEndpoint(bylabel.RoleDecode, 0)
	cached := admissionEndpoint(bylabel.RoleDecode, 10)
	prefill := admissionEndpoint(bylabel.RolePrefill, 0)
	encode := admissionEndpoint(bylabel.RoleEncode, 0)
	request := makeRequestWithTokens(10)
	endpoints := []scheduling.Endpoint{uncached, cached, prefill, encode}
	for range 2 {
		ready, admitted := handler.FilterForAdmission(t.Context(), request, admissionProfiles(), endpoints, noPrefillCapacity)
		require.True(t, ready)
		require.Equal(t, map[string][]scheduling.Endpoint{"decode": {cached}}, admitted)
		require.Equal(t, []scheduling.Endpoint{uncached, cached, prefill, encode}, endpoints)
		_, memoized := request.GetAttribute(remotePrefillDecisionAttributeKey)
		require.False(t, memoized)
		_, declined := request.GetAttribute(prefillDeclinedAttributeKey)
		require.False(t, declined)
	}
	// The real scheduling decision remains free to inspect its selected decoder.
	require.True(t, decider.disaggregate(t.Context(), request, uncached))
}

func TestAdmissionPrefixNeedsPrefillWaits(t *testing.T) {
	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 5})
	require.NoError(t, err)
	handler := NewDisaggProfileHandler("decode", "prefill", "", decider, nil)
	endpoints := []scheduling.Endpoint{admissionEndpoint(bylabel.RoleDecode, 0), admissionEndpoint(bylabel.RolePrefill, 0)}
	ready, _ := handler.FilterForAdmission(t.Context(), makeRequestWithTokens(10), admissionProfiles(), endpoints, noPrefillCapacity)
	require.False(t, ready)
}

func TestAdmissionSharedEndpointUsesSeparateStageLimits(t *testing.T) {
	shared := admissionEndpoint(bylabel.RolePrefillDecode, 0)
	prefill := admissionEndpoint(bylabel.RolePrefill, 0)
	handler := NewDisaggProfileHandler("decode", "prefill", "", newAlwaysDisaggPDDecider(), nil)
	filter := func(stage string, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
		if stage == flowcontrol.SaturationStagePrefill {
			return []scheduling.Endpoint{prefill}
		}
		return endpoints
	}
	ready, admitted := handler.FilterForAdmission(t.Context(), makeRequestWithTokens(10), admissionProfiles(), []scheduling.Endpoint{shared, prefill}, filter)
	require.True(t, ready)
	require.Equal(t, map[string][]scheduling.Endpoint{"decode": {shared}, "prefill": {prefill}}, admitted)

	decode := admissionEndpoint(bylabel.RoleDecode, 0)
	ready, admitted = handler.FilterForAdmission(t.Context(), makeRequestWithTokens(10), admissionProfiles(), []scheduling.Endpoint{shared, prefill, decode}, filter)
	require.True(t, ready)
	require.Equal(t, map[string][]scheduling.Endpoint{"decode": {shared, decode}, "prefill": {prefill}}, admitted)
}

func TestAdmissionSharedDecoderCanSkipPrefill(t *testing.T) {
	decider, err := NewPrefixBasedPDDecider(PrefixBasedPDDeciderConfig{NonCachedTokens: 5})
	require.NoError(t, err)
	handler := NewDisaggProfileHandler("decode", "prefill", "", decider, nil)
	shared := admissionEndpoint(bylabel.RolePrefillDecode, 10)
	ready, admitted := handler.FilterForAdmission(t.Context(), makeRequestWithTokens(10), admissionProfiles(), []scheduling.Endpoint{shared}, noPrefillCapacity)
	require.True(t, ready)
	require.Equal(t, map[string][]scheduling.Endpoint{"decode": {shared}}, admitted)
}

func TestAdmissionMissingStageAndUnknownConfigurationBypass(t *testing.T) {
	decode := admissionEndpoint(bylabel.RoleDecode, 0)
	prefill := admissionEndpoint(bylabel.RolePrefill, 0)
	for _, tc := range []struct {
		name      string
		handler   *Handler
		profiles  map[string]scheduling.SchedulerProfile
		endpoints []scheduling.Endpoint
	}{
		{"no_prefill_workers", NewDisaggProfileHandler("decode", "prefill", "", newAlwaysDisaggPDDecider(), nil), admissionProfiles(), []scheduling.Endpoint{decode}},
		{"no_decode_workers", NewDisaggProfileHandler("decode", "prefill", "", newAlwaysDisaggPDDecider(), nil), admissionProfiles(), []scheduling.Endpoint{prefill}},
		{"unknown_decider", NewDisaggProfileHandler("decode", "prefill", "", &mockPDDecider{}, nil), admissionProfiles(), []scheduling.Endpoint{decode, prefill}},
		{"custom_role_profile", NewDisaggProfileHandler("decode", "prefill", "", newAlwaysDisaggPDDecider(), nil), map[string]scheduling.SchedulerProfile{"decode": admissionProfile{}}, []scheduling.Endpoint{decode, prefill}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ready, admitted := tc.handler.FilterForAdmission(t.Context(), makeRequestWithTokens(10), tc.profiles, tc.endpoints,
				func(string, []scheduling.Endpoint) []scheduling.Endpoint {
					t.Fatal("unexpected capacity evaluation")
					return nil
				})
			require.True(t, ready)
			require.Nil(t, admitted)
		})
	}
}
