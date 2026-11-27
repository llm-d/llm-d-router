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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/headerprofile"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

func TestInFlightLoadProducer_DispatchedStageAccounting(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name           string
		prefillRole    string
		primaryRole    string
		remotePrefill  bool
		sharedEndpoint bool
		cachedBlocks   int
	}{
		{name: "mixed_prefiller", prefillRole: bylabel.RolePrefillDecode, primaryRole: bylabel.RoleDecode, remotePrefill: true},
		{name: "mixed_decoder", prefillRole: bylabel.RolePrefill, primaryRole: bylabel.RolePrefillDecode, remotePrefill: true},
		{name: "unlabeled_targets", remotePrefill: true},
		{name: "no_prefill_result_decode_role", primaryRole: bylabel.RoleDecode},
		{name: "no_prefill_result_mixed_role", primaryRole: bylabel.RolePrefillDecode},
		{name: "same_endpoint", primaryRole: bylabel.RolePrefillDecode, remotePrefill: true, sharedEndpoint: true},
		{name: "cached_remote_prefill", remotePrefill: true, cachedBlocks: 1000},
		{name: "cached_no_prefill_result_mixed_role", primaryRole: bylabel.RolePrefillDecode, cachedBlocks: 1000},
	} {
		for _, includeOutput := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s_output_%t", tc.name, includeOutput), func(t *testing.T) {
				producer := newTestProducer(t)
				producer.addEstimatedOutputTokens = includeOutput
				primary := newEventEndpoint("stage-primary", tc.primaryRole, decodeAddress)
				prefill := newEventEndpoint("stage-prefill", tc.prefillRole, prefillAddress)
				if tc.sharedEndpoint {
					prefill = primary
				}
				inputTokens := int64(8000 - tc.cachedBlocks*4)
				if tc.cachedBlocks > 0 {
					cacheTarget := primary
					if tc.remotePrefill {
						cacheTarget = prefill
					}
					cacheTarget.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(tc.cachedBlocks, 2000, 4))
				}
				profiles := map[string]*fwksched.ProfileRunResult{
					"answer-stage": {TargetEndpoints: []fwksched.Endpoint{primary}},
				}
				if tc.remotePrefill {
					profiles["prompt-stage"] = &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{prefill}}
				}
				req := newStampedRequest(t.Name(), seq(0, 8000))
				handler := disagg.NewDisaggProfileHandler("answer-stage", "prompt-stage", "", nil, nil)
				res, err := handler.ProcessResults(t.Context(), req, profiles)
				require.NoError(t, err)
				req.SchedulingResult = res
				require.Equal(t, "prompt-stage", res.PrefillProfileName)
				require.NoError(t, producer.PreRequest(t.Context(), req, res))
				primaryID := primary.GetMetadata().ID.String()
				primaryTokens := int64(8000)
				if !tc.remotePrefill {
					primaryTokens = inputTokens
				}
				if includeOutput {
					primaryTokens = UnknownOutputTokens
					if !tc.remotePrefill && tc.primaryRole != bylabel.RoleDecode {
						primaryTokens += inputTokens
					}
				}
				primaryRequests := int64(1)
				if tc.sharedEndpoint {
					primaryTokens += inputTokens
					primaryRequests++
				}
				assert.Equal(t, primaryTokens, producer.GetTokens(primaryID))
				assert.Equal(t, primaryRequests, producer.GetRequests(primaryID))
				if tc.remotePrefill && !tc.sharedEndpoint {
					assert.Equal(t, inputTokens, producer.GetTokens(prefill.GetMetadata().ID.String()))
				}
				producer.ResponseBody(t.Context(), req, &fwkrh.Response{StartOfStream: true}, nil)
				remainingTokens := int64(0)
				if includeOutput {
					remainingTokens = UnknownOutputTokens
					if !tc.remotePrefill && tc.primaryRole != bylabel.RoleDecode {
						remainingTokens += inputTokens
					}
				}
				assert.Equal(t, remainingTokens, producer.GetTokens(primaryID))
				assert.Equal(t, int64(1), producer.GetRequests(primaryID))
				producer.ResponseBody(t.Context(), req, &fwkrh.Response{StartOfStream: true}, nil)
				assert.Equal(t, remainingTokens, producer.GetTokens(primaryID))
				producer.ResponseBody(t.Context(), req, &fwkrh.Response{EndOfStream: true}, nil)
				assert.Zero(t, producer.GetTokens(primaryID))
				assert.Zero(t, producer.GetRequests(primaryID))
				assert.Zero(t, producer.GetTokens(prefill.GetMetadata().ID.String()))
				assert.Zero(t, producer.GetRequests(prefill.GetMetadata().ID.String()))
			})
		}
	}
}

func TestInFlightLoadProducer_PrefillOnlyStage(t *testing.T) {
	t.Parallel()
	for _, includeOutput := range []bool{false, true} {
		t.Run(fmt.Sprintf("output_%t", includeOutput), func(t *testing.T) {
			producer := newKVEventsProducer(t, fmt.Sprintf("stage-prefill-only-%t", includeOutput))
			producer.addEstimatedOutputTokens = includeOutput
			prefill := newEventEndpoint("standalone-prefill", bylabel.RolePrefillDecode, prefillAddress)
			req := newStampedRequest(t.Name(), seq(0, blockSize))
			handler := headerprofile.NewHeaderProfileHandler("", "prompt-stage").WithPrefillProfileName("prompt-stage")
			res, err := handler.ProcessResults(t.Context(), req, map[string]*fwksched.ProfileRunResult{
				"prompt-stage": {TargetEndpoints: []fwksched.Endpoint{prefill}},
			})
			require.NoError(t, err)
			req.SchedulingResult = res
			require.NoError(t, producer.PreRequest(t.Context(), req, res))
			id := prefill.GetMetadata().ID.String()
			assert.Equal(t, int64(blockSize), producer.GetTokens(id))
			assert.Equal(t, int64(1), producer.GetRequests(id))
			require.NoError(t, producer.ProcessEvents(t.Context(),
				kvevents.EventSource{Endpoint: prefillSource}, storeBatch(req.RequestID, seq(0, blockSize))))
			assert.Zero(t, producer.GetTokens(id))
			assert.Zero(t, producer.GetRequests(id))
			producer.ResponseBody(t.Context(), req, &fwkrh.Response{StartOfStream: true, EndOfStream: true}, nil)
			assert.Zero(t, producer.GetTokens(id))
			assert.Zero(t, producer.GetRequests(id))
		})
	}
}

func TestInFlightLoadProducer_DispatchedLegacyRoles(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name           string
		role           string
		tokens         int64
		prefillProfile string
	}{
		{name: "prefill", role: bylabel.RolePrefill, tokens: 8000},
		{name: "decode", role: bylabel.RoleDecode, tokens: UnknownOutputTokens},
		{name: "combined", role: bylabel.RolePrefillDecode, tokens: 8000 + UnknownOutputTokens},
		{name: "coordinator_decode", role: bylabel.RoleDecode, tokens: UnknownOutputTokens, prefillProfile: "prompt-stage"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			producer := newTestProducer(t)
			endpoint := newEventEndpoint("legacy-"+tc.name, tc.role, decodeAddress)
			req := newStampedRequest(t.Name(), seq(0, 8000))
			handler := headerprofile.NewHeaderProfileHandler("", "custom-stage").WithPrefillProfileName(tc.prefillProfile)
			res, err := handler.ProcessResults(t.Context(), req, map[string]*fwksched.ProfileRunResult{
				"custom-stage": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
			})
			require.NoError(t, err)
			req.SchedulingResult = res
			require.Equal(t, tc.prefillProfile, res.PrefillProfileName)
			require.NoError(t, producer.PreRequest(t.Context(), req, res))
			id := endpoint.GetMetadata().ID.String()
			assert.Equal(t, tc.tokens, producer.GetTokens(id))
			producer.ResponseBody(t.Context(), req, &fwkrh.Response{EndOfStream: true}, nil)
			assert.Zero(t, producer.GetTokens(id))
			assert.Zero(t, producer.GetRequests(id))
		})
	}
}

func TestInFlightLoadProducer_StageEventSources(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name           string
		primaryAddress string
		primaryPort    string
		primaryRank    int
		rankByHeader   bool
		includeOutput  bool
		wantKVComplete bool
	}{
		{name: "distinct_workers", primaryAddress: decodeAddress, wantKVComplete: true},
		{name: "shared_event_source", primaryAddress: prefillAddress},
		{name: "missing_primary_address"},
		{name: "different_rank_same_host", primaryAddress: prefillAddress, primaryRank: 1, rankByHeader: true, wantKVComplete: true},
		{name: "different_port_same_host", primaryAddress: prefillAddress, primaryPort: "8001", wantKVComplete: true},
		{name: "distinct_workers_output", primaryAddress: decodeAddress, includeOutput: true, wantKVComplete: true},
		{name: "shared_event_source_output", primaryAddress: prefillAddress, includeOutput: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			producer := newKVEventsProducer(t, "stage-events-"+tc.name)
			producer.addEstimatedOutputTokens = tc.includeOutput
			primary := newEventEndpoint("event-primary", bylabel.RolePrefillDecode, tc.primaryAddress)
			if tc.primaryPort != "" {
				primary.GetMetadata().Port = tc.primaryPort
			}
			primary.GetMetadata().RankByHeader = tc.rankByHeader
			primary.GetMetadata().RankIndex = tc.primaryRank
			prefill := newEventEndpoint("event-prefill", bylabel.RolePrefillDecode, prefillAddress)
			encoder := newEventEndpoint("event-encoder", bylabel.RoleEncode, "10.0.0.3")
			req := newStampedRequest(t.Name(), seq(0, blockSize))
			handler := disagg.NewDisaggProfileHandler("answer-stage", "prompt-stage", "image-stage", nil, nil)
			res, err := handler.ProcessResults(t.Context(), req, map[string]*fwksched.ProfileRunResult{
				"answer-stage": {TargetEndpoints: []fwksched.Endpoint{primary}},
				"prompt-stage": {TargetEndpoints: []fwksched.Endpoint{prefill}},
				"image-stage":  {TargetEndpoints: []fwksched.Endpoint{encoder}},
			})
			require.NoError(t, err)
			req.SchedulingResult = res
			require.NoError(t, producer.PreRequest(t.Context(), req, res))
			prefillID := prefill.GetMetadata().ID.String()
			for _, source := range []string{prefillSource, eventEndpoint(encoder)} {
				require.NoError(t, producer.ProcessEvents(t.Context(),
					kvevents.EventSource{Endpoint: source}, storeBatch(req.RequestID, seq(0, blockSize))))
				assert.Equal(t, int64(1), producer.GetRequests(prefillID), "source %s must not complete the prefill", source)
			}
			if tc.wantKVComplete {
				require.NoError(t, producer.ProcessEvents(t.Context(),
					kvevents.EventSource{Endpoint: eventEndpoint(primary)}, storeBatch(req.RequestID, seq(0, blockSize))))
				assert.Zero(t, producer.GetRequests(prefillID))
				assert.Equal(t, int64(1), producer.GetRequests(primary.GetMetadata().ID.String()))
				assert.Equal(t, float64(1), completions(producer.typedName.Name, signalKVEvent))
				require.NoError(t, producer.ProcessEvents(t.Context(),
					kvevents.EventSource{Endpoint: eventEndpoint(primary)}, storeBatch(req.RequestID, seq(0, blockSize))))
				assert.Equal(t, float64(1), completions(producer.typedName.Name, signalKVEvent))
			}
			producer.ResponseBody(t.Context(), req, &fwkrh.Response{StartOfStream: true}, nil)
			assert.Zero(t, producer.GetRequests(prefillID))
			assert.Zero(t, producer.GetTokens(prefillID))
			remainingTokens := int64(0)
			if tc.includeOutput {
				remainingTokens = UnknownOutputTokens
			}
			assert.Equal(t, remainingTokens, producer.GetTokens(primary.GetMetadata().ID.String()))
			assert.Equal(t, int64(1), producer.GetRequests(primary.GetMetadata().ID.String()))
			producer.ResponseBody(t.Context(), req, &fwkrh.Response{EndOfStream: true}, nil)
			for _, endpoint := range []fwksched.Endpoint{prefill, primary, encoder} {
				assert.Zero(t, producer.GetRequests(endpoint.GetMetadata().ID.String()))
				assert.Zero(t, producer.GetTokens(endpoint.GetMetadata().ID.String()))
			}
		})
	}
}
