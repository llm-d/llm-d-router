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

package inflightload

import (
	"fmt"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

func TestInFlightLoadProducer_PrefillReleaseIndependentOfProfileName(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct{ name, profileName, role, prefillProfile string }{
		{"default_profile", fwksched.DefaultPrefillProfile, bylabel.RolePrefill, fwksched.DefaultPrefillProfile},
		{"custom_profile", "prompt-stage", bylabel.RolePrefill, ""},
		{"encode_prefill", "prompt-stage", bylabel.RoleEncodePrefill, ""},
		{"unlabeled_default_profile", fwksched.DefaultPrefillProfile, "", fwksched.DefaultPrefillProfile},
		{"explicit_custom_profile", "prompt-stage", bylabel.RolePrefill, "prompt-stage"},
		{"explicit_combined_role", "prompt-stage", bylabel.RolePrefillDecode, "prompt-stage"},
		{"explicit_unlabeled_profile", "prompt-stage", "", "prompt-stage"},
	} {
		for _, includeOutput := range []bool{false, true} {
			for _, signal := range []string{signalFirstByte, signalKVEvent} {
				t.Run(fmt.Sprintf("%s_output_%t_%s", tc.name, includeOutput, signal), func(t *testing.T) {
					var producer *InFlightLoadProducer
					if signal == signalKVEvent {
						producer = newKVEventsProducer(t, fmt.Sprintf("profile-name-%s-%t", tc.name, includeOutput))
					} else {
						producer = newTestProducer(t)
					}
					producer.addEstimatedOutputTokens = includeOutput

					ctx := t.Context()
					prefill := newEventEndpoint("profile-prefill", tc.role, prefillAddress)
					decode := newEventEndpoint("profile-decode", bylabel.RoleDecode, decodeAddress)
					prefillID := prefill.GetMetadata().ID.String()
					decodeID := decode.GetMetadata().ID.String()
					prompt := seq(0, 8000)
					req := newStampedRequest(t.Name(), prompt)
					res := &fwksched.SchedulingResult{
						PrimaryProfileName: "decode",
						PrefillProfileName: tc.prefillProfile,
						ProfileResults: map[string]*fwksched.ProfileRunResult{
							tc.profileName: {TargetEndpoints: []fwksched.Endpoint{prefill}},
							"decode":       {TargetEndpoints: []fwksched.Endpoint{decode}},
						},
					}
					req.SchedulingResult = res
					require.NoError(t, producer.PreRequest(ctx, req, res))
					require.Equal(t, int64(1), producer.GetRequests(prefillID))
					initialPrefillTokens := int64(len(prompt))
					if includeOutput && !endpointHasPrefillOnlyRole(prefill) {
						initialPrefillTokens += UnknownOutputTokens
					}
					require.Equal(t, initialPrefillTokens, producer.GetTokens(prefillID))
					require.Equal(t, int64(1), producer.GetRequests(decodeID))
					initialDecodeTokens := int64(len(prompt))
					if includeOutput {
						initialDecodeTokens = UnknownOutputTokens
					}
					require.Equal(t, initialDecodeTokens, producer.GetTokens(decodeID))

					if signal == signalKVEvent {
						assert.Equal(t, float64(1), pendingWatch(producer.typedName.Name))
						fullPrompt := prompt[:len(prompt)/blockSize*blockSize]
						require.NoError(t, producer.ProcessEvents(ctx,
							kvevents.EventSource{Endpoint: prefillSource}, storeBatch(req.RequestID, fullPrompt)))
						assert.Equal(t, int64(1), producer.GetRequests(prefillID), "prefill must stay charged until decode receives the KV blocks")
						assert.Equal(t, initialPrefillTokens, producer.GetTokens(prefillID))
						assert.Equal(t, int64(1), producer.GetRequests(decodeID))
						assert.Equal(t, initialDecodeTokens, producer.GetTokens(decodeID))
						assert.Equal(t, float64(1), pendingWatch(producer.typedName.Name), "prefill stores must not consume the decode watch")
						require.NoError(t, producer.ProcessEvents(ctx,
							kvevents.EventSource{Endpoint: decodeSource}, storeBatch(req.RequestID, fullPrompt)))
						assert.Zero(t, pendingWatch(producer.typedName.Name))
						require.NoError(t, producer.ProcessEvents(ctx,
							kvevents.EventSource{Endpoint: decodeSource}, storeBatch(req.RequestID, fullPrompt)))
						assert.Equal(t, float64(1), completions(producer.typedName.Name, signalKVEvent), "duplicate events must complete the request once")
					} else {
						producer.ResponseBody(ctx, req, &fwkrh.Response{StartOfStream: true}, nil)
					}

					assert.Equal(t, int64(0), producer.GetRequests(prefillID), "completed prefill must release its request contribution")
					assert.Equal(t, int64(0), producer.GetTokens(prefillID), "completed prefill must release its token contribution")
					assert.Equal(t, int64(1), producer.GetRequests(decodeID), "decode must remain in flight")
					remainingDecodeTokens := int64(0)
					if includeOutput {
						remainingDecodeTokens = UnknownOutputTokens
					}
					assert.Equal(t, remainingDecodeTokens, producer.GetTokens(decodeID))
					producer.ResponseBody(ctx, req, &fwkrh.Response{StartOfStream: true}, nil)
					assert.Zero(t, producer.GetRequests(prefillID))
					assert.Zero(t, producer.GetTokens(prefillID))
					assert.Equal(t, int64(1), producer.GetRequests(decodeID))
					assert.Equal(t, remainingDecodeTokens, producer.GetTokens(decodeID))

					producer.ResponseBody(ctx, req, &fwkrh.Response{EndOfStream: true}, nil)
					assert.Zero(t, producer.GetRequests(prefillID))
					assert.Zero(t, producer.GetTokens(prefillID))
					assert.Zero(t, producer.GetRequests(decodeID))
					assert.Zero(t, producer.GetTokens(decodeID))
				})
			}
		}
	}
}

func TestInFlightLoadProducer_SingleTargetPrefillCompletion(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name        string
		profileName string
		role        string
		releaseAll  bool
	}{
		{"unlabeled_default_profile", fwksched.DefaultPrefillProfile, "", true},
		{"custom_profile", "prompt-stage", bylabel.RolePrefill, true},
		{"encode_prefill", "prompt-stage", bylabel.RoleEncodePrefill, true},
		{"combined_role", "default", bylabel.RolePrefillDecode, false},
		{"unlabeled_serving_engine", "default", "", false},
	} {
		for _, includeOutput := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s_output_%t", tc.name, includeOutput), func(t *testing.T) {
				producer := newKVEventsProducer(t, fmt.Sprintf("single-profile-%s-%s-%t", tc.profileName, tc.role, includeOutput))
				producer.addEstimatedOutputTokens = includeOutput
				ctx := t.Context()
				endpoint := newEventEndpoint("single-profile", tc.role, prefillAddress)
				endpointID := endpoint.GetMetadata().ID.String()
				prompt := seq(0, 200)
				req := newStampedRequest(t.Name(), prompt)
				res := &fwksched.SchedulingResult{
					PrimaryProfileName: tc.profileName,
					ProfileResults: map[string]*fwksched.ProfileRunResult{
						tc.profileName: {TargetEndpoints: []fwksched.Endpoint{endpoint}},
					},
				}
				if tc.profileName == fwksched.DefaultPrefillProfile {
					res.PrefillProfileName = tc.profileName
				}
				req.SchedulingResult = res
				require.NoError(t, producer.PreRequest(ctx, req, res))
				require.Equal(t, int64(1), producer.GetRequests(endpointID))
				require.NoError(t, producer.ProcessEvents(ctx,
					kvevents.EventSource{Endpoint: prefillSource}, storeBatch(req.RequestID, prompt[:192])))
				if tc.releaseAll {
					require.Zero(t, producer.GetRequests(endpointID))
					require.Zero(t, producer.GetTokens(endpointID))
				} else {
					require.Equal(t, int64(1), producer.GetRequests(endpointID), "a serving engine still has decode work")
					remainingTokens := int64(0)
					if includeOutput {
						remainingTokens = int64(len(prompt)) + UnknownOutputTokens
					}
					require.Equal(t, remainingTokens, producer.GetTokens(endpointID))
				}
				require.Zero(t, pendingWatch(producer.typedName.Name), "the event must complete the prefill watch")
				producer.ResponseBody(ctx, req, &fwkrh.Response{StartOfStream: true}, nil)
				producer.ResponseBody(ctx, req, &fwkrh.Response{EndOfStream: true}, nil)
				require.Zero(t, producer.GetRequests(endpointID))
				require.Zero(t, producer.GetTokens(endpointID))
			})
		}
	}
}

func TestInFlightLoadProducer_PrefillEventTargets(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name             string
		prefillAddresses []string
		decodeAddresses  []string
		wantSources      []string
	}{
		{
			name:             "decode_targets_take_precedence",
			prefillAddresses: []string{prefillAddress, "10.0.0.3"},
			decodeAddresses:  []string{decodeAddress, "10.0.0.4"},
			wantSources:      []string{decodeSource, "10.0.0.4:8000"},
		},
		{
			name:             "all_prefill_targets_are_fallbacks",
			prefillAddresses: []string{prefillAddress, "10.0.0.3"},
			wantSources:      []string{prefillSource, "10.0.0.3:8000"},
		},
		{
			name:             "decode_without_address_uses_prefill_fallbacks",
			prefillAddresses: []string{prefillAddress, "10.0.0.3"},
			decodeAddresses:  []string{""},
			wantSources:      []string{prefillSource, "10.0.0.3:8000"},
		},
		{
			name:             "no_usable_address",
			prefillAddresses: []string{""},
			decodeAddresses:  []string{""},
		},
		{
			name:            "decode_only_targets",
			decodeAddresses: []string{decodeAddress, "10.0.0.4"},
			wantSources:     []string{decodeSource, "10.0.0.4:8000"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			producer := newKVEventsProducer(t, "event-targets-"+tc.name)
			ctx := t.Context()
			res := &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{}}
			for i, address := range tc.prefillAddresses {
				name := fmt.Sprintf("prompt-stage-%d", i)
				res.ProfileResults[name] = &fwksched.ProfileRunResult{
					TargetEndpoints: []fwksched.Endpoint{newEventEndpoint(name, bylabel.RolePrefill, address)},
				}
			}
			for i, address := range tc.decodeAddresses {
				name := fmt.Sprintf("decode-stage-%d", i)
				res.ProfileResults[name] = &fwksched.ProfileRunResult{
					TargetEndpoints: []fwksched.Endpoint{newEventEndpoint(name, bylabel.RoleDecode, address)},
				}
			}
			for _, source := range []string{prefillSource, decodeSource, "10.0.0.3:8000", "10.0.0.4:8000"} {
				req := newStampedRequest(t.Name()+source, seq(0, blockSize))
				req.SchedulingResult = res
				require.NoError(t, producer.PreRequest(ctx, req, res))
				before := completions(producer.typedName.Name, signalKVEvent)
				require.NoError(t, producer.ProcessEvents(ctx,
					kvevents.EventSource{Endpoint: source}, storeBatch(req.RequestID, seq(0, blockSize))))
				completed := completions(producer.typedName.Name, signalKVEvent) - before
				if slices.Contains(tc.wantSources, source) {
					require.Equal(t, float64(1), completed, "source %s must complete prefill", source)
				} else {
					require.Zero(t, completed, "source %s must not complete prefill", source)
				}
				producer.ResponseBody(ctx, req, &fwkrh.Response{EndOfStream: true}, nil)
				require.Zero(t, pendingWatch(producer.typedName.Name))
			}
		})
	}
}

func TestInFlightLoadProducer_ExplicitPrefillProfileOverridesRole(t *testing.T) {
	t.Parallel()
	producer := newKVEventsProducer(t, "explicit-prefill-overrides-role")
	prefill := newEventEndpoint("explicit-prefill", bylabel.RolePrefill, prefillAddress)
	primary := newEventEndpoint("primary", bylabel.RolePrefill, decodeAddress)
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "primary",
		PrefillProfileName: "prompt-stage",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prompt-stage": {TargetEndpoints: []fwksched.Endpoint{prefill}},
			"primary":      {TargetEndpoints: []fwksched.Endpoint{primary}},
		},
	}
	prompt := seq(0, blockSize)
	req := newStampedRequest(t.Name(), prompt)
	req.SchedulingResult = res
	require.NoError(t, producer.PreRequest(t.Context(), req, res))
	require.NoError(t, producer.ProcessEvents(t.Context(),
		kvevents.EventSource{Endpoint: prefillSource}, storeBatch(req.RequestID, prompt)))
	require.Equal(t, int64(1), producer.GetRequests(prefill.GetMetadata().ID.String()))
	require.Equal(t, float64(1), pendingWatch(producer.typedName.Name))
	require.NoError(t, producer.ProcessEvents(t.Context(),
		kvevents.EventSource{Endpoint: decodeSource}, storeBatch(req.RequestID, prompt)))
	require.Zero(t, producer.GetRequests(prefill.GetMetadata().ID.String()))
	require.Equal(t, int64(1), producer.GetRequests(primary.GetMetadata().ID.String()))
	require.Zero(t, pendingWatch(producer.typedName.Name))
	producer.ResponseBody(t.Context(), req, &fwkrh.Response{EndOfStream: true}, nil)
	require.Zero(t, producer.GetRequests(primary.GetMetadata().ID.String()))
}
