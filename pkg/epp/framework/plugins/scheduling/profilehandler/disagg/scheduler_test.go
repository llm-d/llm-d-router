/*
Copyright 2025 The llm-d Authors.

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

package disagg_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log" // Import config for thresholds

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/utilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loadaware"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

const (
	prefill = "prefill"
	decode  = "decode"

	// averageCharactersPerToken derives token counts from character-length
	// prompt fixtures in tests.
	averageCharactersPerToken = 4
)

// completionsBody builds a completions request body whose tokenized prompt carries
// len(prompt)/averageCharactersPerToken token IDs, which the decider reads as
// the input token count.
func completionsBody(prompt string) *fwkrequest.InferenceRequestBody {
	return &fwkrequest.InferenceRequestBody{
		Completions:      &fwkrequest.CompletionsRequest{Prompt: fwkrequest.Prompt{Raw: prompt}},
		TokenizedRequest: &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: make([]uint32, len(prompt)/averageCharactersPerToken)}}},
	}
}

// Tests the scheduler expected behavior.
func TestPDSchedule(t *testing.T) {
	scrapedAt := time.Now()
	endpoint1 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "endpoint1"},
			Address: "1.2.3.4",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0, UpdateTime: scrapedAt},
		fwkdl.NewAttributes(),
	)
	endpoint2 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "endpoint2"},
			Address: "5.6.7.8",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RoleDecode},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0, UpdateTime: scrapedAt},
		fwkdl.NewAttributes(),
	)
	noRoleEndpoint1 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "noRoleEndpoint1"},
			Address: "1.1.1.1",
		},
		&fwkdl.Metrics{WaitingQueueSize: 2, UpdateTime: scrapedAt},
		fwkdl.NewAttributes(),
	)

	prefillDecodeResult := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			decode: {
				TargetEndpoints: []fwksched.Endpoint{
					&fwksched.ScoredEndpoint{
						Endpoint: endpoint2,
					},
				},
			},
			prefill: {
				TargetEndpoints: []fwksched.Endpoint{
					&fwksched.ScoredEndpoint{
						Endpoint: endpoint1,
					},
				},
			},
		},

		PrimaryProfileName: decode,
	}

	decodeResult := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			decode: {
				TargetEndpoints: []fwksched.Endpoint{
					&fwksched.ScoredEndpoint{
						Endpoint: endpoint2,
					},
				},
			},
		},
		PrimaryProfileName: decode,
	}

	tests := []struct {
		name     string
		req      *fwksched.InferenceRequest
		input    []fwksched.Endpoint
		wantRes  *fwksched.SchedulingResult
		wantRes2 *fwksched.SchedulingResult // a subsequent call to check prefix cache and how it affects PD
		err      bool
	}{
		{
			name: "no candidate endpoints",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "any-model",
				Body:        completionsBody("12345678901"),
			},
			input: []fwksched.Endpoint{},
			err:   true,
		},
		{
			name: "one decode endpoint, long prompt, no prefill endpoint available",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "critical",
				Body:        completionsBody("12345678901"),
			},
			// The long, uncached prompt makes the decider pick the prefill profile,
			// but no Prefill-role endpoint is present: the request must fail rather
			// than silently complete decode-only.
			input: []fwksched.Endpoint{endpoint2},
			err:   true,
		},
		{
			name: "one prefill endpoint, long prompt",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "critical",
				Body:        completionsBody("12345678901"),
			},
			// no Decode endpoint
			input: []fwksched.Endpoint{endpoint1},
			err:   true,
		},
		{
			name: "1P1D - long prompt",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "critical",
				Body:        completionsBody("12345678906"),
			},
			// endpoint2 will be picked in the decode profile result, endpoint1 will be in the prefill profile result
			input:    []fwksched.Endpoint{endpoint1, endpoint2},
			wantRes:  prefillDecodeResult,
			wantRes2: decodeResult,
		},
		{
			name: "1P1Dshort",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "critical",
				Body:        completionsBody("12345"),
			},
			// endpoint2 will be picked because it is the decode endpoint, endpoint1 shouldn't be picked,
			// because the prompt is too short
			input:    []fwksched.Endpoint{endpoint1, endpoint2},
			wantRes:  decodeResult,
			wantRes2: decodeResult,
		},
		{
			name: "TestRolesWithNoDecode",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "critical",
				Body:        completionsBody("12345678901"),
			},
			input: []fwksched.Endpoint{endpoint1, noRoleEndpoint1},
			wantRes: &fwksched.SchedulingResult{
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					decode: {
						TargetEndpoints: []fwksched.Endpoint{
							&fwksched.ScoredEndpoint{
								Endpoint: noRoleEndpoint1,
							},
						},
					},
					prefill: {
						TargetEndpoints: []fwksched.Endpoint{
							&fwksched.ScoredEndpoint{
								Endpoint: endpoint1,
							},
						},
					},
				},
				PrimaryProfileName: decode,
			},
		},
		{
			name: "1P2D - long prompt",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "critical",
				Body:        completionsBody("1234567890123456789012345678901234567890"),
			},
			// endpoint2 will be picked in the decode profile result cause it has higher score than noRoleEndpoint1
			// endpoint1 will be in the prefill profile result
			input:    []fwksched.Endpoint{endpoint1, endpoint2, noRoleEndpoint1},
			wantRes:  prefillDecodeResult,
			wantRes2: decodeResult,
		},
	}

	ctx := context.Background()
	logger := testr.New(t)
	ctx = log.IntoContext(ctx, logger)

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			//  initialize scheduler with config
			prefixScorer, err := prefix.New(ctx, prefix.PrefixCacheScorerPluginType, "")
			assert.NoError(t, err, "Prefix plugin creation returned unexpected error")
			datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{prefixScorer})

			prefillSchedulerProfile := scheduling.NewSchedulerProfile().
				WithFilters(bylabel.NewPrefillRole()).
				WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
			err = prefillSchedulerProfile.AddPlugins(scheduling.NewWeightedScorer(prefixScorer, 50))
			assert.NoError(t, err, "SchedulerProfile AddPlugins returned unexpected error")

			decodeSchedulerProfile := scheduling.NewSchedulerProfile().
				WithFilters(bylabel.NewDecodeRole()).
				WithScorers(scheduling.NewWeightedScorer(loadaware.NewLoadAware(ctx, loadaware.QueueThresholdDefault), 1)).
				WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
			err = decodeSchedulerProfile.AddPlugins(scheduling.NewWeightedScorer(prefixScorer, 0))
			assert.NoError(t, err, "SchedulerProfile AddPlugins returned unexpected error")

			deciderPlugin, err := disagg.NewPrefixBasedPDDecider(disagg.PrefixBasedPDDeciderConfig{NonCachedTokens: 2})
			assert.NoError(t, err)

			profileHandle := disagg.NewDisaggProfileHandler(decode, prefill, "",
				deciderPlugin, nil)

			schedulerConfig := scheduling.NewSchedulerConfig(profileHandle, map[string]fwksched.SchedulerProfile{
				prefill: prefillSchedulerProfile,
				decode:  decodeSchedulerProfile,
			})
			scheduler := scheduling.NewSchedulerWithConfig(schedulerConfig)

			inputTokens := len(test.req.Body.Completions.Prompt.Raw) / averageCharactersPerToken
			for _, pod := range test.input {
				pod.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(0, inputTokens, 1))
			}
			got, err := scheduler.Schedule(ctx, test.req, test.input)

			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}
			if test.err {
				var typedErr errcommon.Error
				if !errors.As(err, &typedErr) {
					t.Fatalf("Schedule error is not an errcommon.Error: %v", err)
				}
				assert.Equal(t, errcommon.ServiceUnavailable, typedErr.Code)
				assert.Equal(t, string(errcommon.RequestDroppedReasonNoEndpoints), typedErr.Headers[errcommon.RequestDroppedReasonHeaderKey])
			}

			if diff := cmp.Diff(test.wantRes, got, cmpopts.IgnoreUnexported(fwkdl.Attributes{}), cmpopts.IgnoreFields(fwksched.ScoredEndpoint{}, "Score"),
				cmpopts.IgnoreFields(fwksched.ProfileRunResult{}, "ScoredCandidates")); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
			if test.wantRes2 != nil { // Checking the prefix match in the decode pod.
				// update number of cached tokens for the following schedule call
				for _, pod := range test.input {
					pod.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(inputTokens, inputTokens, 1))
				}

				// Fresh request for the second schedule call so per-request
				// memoization from the first call doesn't leak. Production
				// models each schedule call as its own *InferenceRequest.
				nextReq := &fwksched.InferenceRequest{
					RequestID:   uuid.NewString(),
					TargetModel: test.req.TargetModel,
					Body:        test.req.Body,
					Headers:     test.req.Headers,
				}
				got, err = scheduler.Schedule(ctx, nextReq, test.input)
				if test.err != (err != nil) {
					t.Errorf("Unexpected error in schedule call, got %v, want %v", err, test.err)
				}

				if diff := cmp.Diff(test.wantRes2, got, cmpopts.IgnoreUnexported(fwkdl.Attributes{}), cmpopts.IgnoreFields(fwksched.ScoredEndpoint{}, "Score"),
					cmpopts.IgnoreFields(fwksched.ProfileRunResult{}, "ScoredCandidates")); diff != "" {
					t.Errorf("Unexpected output in subsequent schedule call (-want +got): %v", diff)
				}
			}
		})
	}
}

func TestPDSchedule_AggregatedFallback(t *testing.T) {
	const fallback = "fallback"
	const encode = "encode"
	ctx := context.Background()

	fallbackEndpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "fallback-endpoint"},
			Address: "1.2.3.4",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RolePrefillDecode},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0},
		fwkdl.NewAttributes(),
	)
	decodeEndpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "decode-endpoint"},
			Address: "5.6.7.8",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RoleDecode},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0},
		fwkdl.NewAttributes(),
	)
	prefillOnlyEndpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "prefill-only-endpoint"},
			Address: "9.10.11.12",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0},
		fwkdl.NewAttributes(),
	)
	encodeEndpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:     k8stypes.NamespacedName{Name: "encode-endpoint"},
			Labels: map[string]string{bylabel.RoleLabel: bylabel.RoleEncode},
		},
		&fwkdl.Metrics{},
		fwkdl.NewAttributes(),
	)

	prefillFilter, err := bylabel.NewSelector("strict-prefill", &metav1.LabelSelector{
		MatchLabels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
	})
	require.NoError(t, err)
	decodeFilter, err := bylabel.NewSelector("strict-decode", &metav1.LabelSelector{
		MatchLabels: map[string]string{bylabel.RoleLabel: bylabel.RoleDecode},
	})
	require.NoError(t, err)
	fallbackFilter, err := bylabel.NewSelector("full-capability", &metav1.LabelSelector{
		MatchLabels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefillDecode},
	})
	require.NoError(t, err)

	capacityFilter, err := utilization.Factory("capacity", json.NewDecoder(strings.NewReader(
		`{"conditions":[{"metric":"running-requests","maxValue":0}]}`)), nil)
	require.NoError(t, err)

	prefillProfile := scheduling.NewSchedulerProfile().
		WithFilters(prefillFilter).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
	decodeProfile := scheduling.NewSchedulerProfile().
		WithFilters(decodeFilter, capacityFilter.(fwksched.Filter)).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
	fallbackProfile := scheduling.NewSchedulerProfile().
		WithFilters(fallbackFilter, capacityFilter.(fwksched.Filter)).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
	encodeProfile := scheduling.NewSchedulerProfile().
		WithFilters(bylabel.NewEncodeRole()).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	for _, tc := range []struct {
		name             string
		hasDecode        bool
		hasPrefill       bool
		hasFallback      bool
		encodeConfigured bool
		hasEncode        bool
		decodeBusy       bool
		fallbackBusy     bool
		prefillDeclined  bool
		cancelled        bool
		wantPrimary      string
		wantFallbackRuns int
	}{
		{name: "decode unavailable", hasPrefill: true, hasFallback: true, wantPrimary: fallback, wantFallbackRuns: 1},
		{name: "both stages available", hasDecode: true, hasPrefill: true, hasFallback: true, wantPrimary: decode},
		{name: "required prefill unavailable", hasDecode: true, hasFallback: true, wantPrimary: fallback, wantFallbackRuns: 1},
		{name: "prefill declined", hasDecode: true, hasFallback: true, prefillDeclined: true, wantPrimary: decode},
		{name: "prefill only cannot satisfy fallback", hasPrefill: true, wantFallbackRuns: 1},
		{name: "fallback unavailable", hasDecode: true, wantFallbackRuns: 1},
		{name: "decode busy", hasDecode: true, hasFallback: true, decodeBusy: true, wantPrimary: fallback, wantFallbackRuns: 1},
		{name: "both workers busy", hasDecode: true, hasFallback: true, decodeBusy: true, fallbackBusy: true, wantFallbackRuns: 1},
		{name: "request cancelled", hasFallback: true, cancelled: true},
		{name: "encode unavailable retains PD", hasDecode: true, hasPrefill: true, hasFallback: true, encodeConfigured: true, wantPrimary: decode},
		{name: "all stages available", hasDecode: true, hasPrefill: true, hasFallback: true, encodeConfigured: true, hasEncode: true, wantPrimary: decode},
		{name: "encode succeeds before prefill fails", hasDecode: true, hasFallback: true, encodeConfigured: true, hasEncode: true, wantPrimary: fallback, wantFallbackRuns: 1},
		{name: "encode succeeds but fallback unavailable", hasDecode: true, encodeConfigured: true, hasEncode: true, wantFallbackRuns: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			decider := &disagg.AlwaysDisaggPDDecider{}
			handler := disagg.NewDisaggProfileHandler(decode, prefill, "", decider, nil).WithFallbackProfile(fallback)
			if tc.encodeConfigured {
				handler = disagg.NewDisaggProfileHandler(decode, prefill, encode, decider, disagg.NewAlwaysDisaggEncodeDecider()).WithFallbackProfile(fallback)
			}
			if tc.prefillDeclined {
				handler = disagg.NewDisaggProfileHandler(decode, prefill, "", nil, nil).WithFallbackProfile(fallback)
			}
			var endpoints []fwksched.Endpoint
			if tc.hasDecode {
				endpoints = append(endpoints, decodeEndpoint)
			}
			if tc.hasPrefill {
				endpoints = append(endpoints, prefillOnlyEndpoint)
			}
			if tc.hasFallback {
				endpoints = append(endpoints, fallbackEndpoint)
			}
			if tc.hasEncode {
				endpoints = append(endpoints, encodeEndpoint)
			}
			decodeEndpoint.GetMetrics().RunningRequestsSize = 0
			fallbackEndpoint.GetMetrics().RunningRequestsSize = 0
			if tc.decodeBusy {
				decodeEndpoint.GetMetrics().RunningRequestsSize = 1
			}
			if tc.fallbackBusy {
				fallbackEndpoint.GetMetrics().RunningRequestsSize = 1
			}

			fallbackRuns := 0
			profiles := map[string]fwksched.SchedulerProfile{
				decode: decodeProfile,
				prefill: profileFunc(func(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
					_, hasPeer := fwksched.ReadRequestAttribute[fwksched.Endpoint](req, disagg.PeerEndpointAttributeKey)
					assert.True(t, hasPeer)
					return prefillProfile.Run(ctx, req, endpoints)
				}),
				fallback: profileFunc(func(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
					fallbackRuns++
					_, hasPeer := fwksched.ReadRequestAttribute[fwksched.Endpoint](req, disagg.PeerEndpointAttributeKey)
					assert.False(t, hasPeer)
					return fallbackProfile.Run(ctx, req, endpoints)
				}),
			}
			encodeRuns := 0
			if tc.encodeConfigured {
				profiles[encode] = profileFunc(func(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
					encodeRuns++
					return encodeProfile.Run(ctx, req, endpoints)
				})
			}
			scheduler := scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(handler, profiles))

			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			if tc.cancelled {
				cancel()
			}
			req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "critical", Body: completionsBody("12345678901"), Headers: map[string]string{reqcommon.RequestIDHeaderKey: "test-request"}}
			if tc.encodeConfigured {
				req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrequest.MultiModalFeature{{Modality: reqcommon.ModalityImage, Hash: "image", Length: 1}}
			}
			counter := disagg.LlmdDisaggDecisionCount.WithLabelValues(handler.TypedName().Name, handler.TypedName().Type, req.TargetModel, metricsutil.DisaggPathAggregatedFallback)
			before := testutil.ToFloat64(counter)
			got, err := scheduler.Schedule(ctx, req, endpoints)
			assert.Equal(t, tc.wantFallbackRuns, fallbackRuns)
			wantDecisions := 0.0
			if tc.wantPrimary == fallback {
				wantDecisions = 1
			}
			assert.Equal(t, before+wantDecisions, testutil.ToFloat64(counter))
			if tc.encodeConfigured {
				assert.Equal(t, 1, encodeRuns)
			}
			if tc.wantPrimary == "" {
				require.Error(t, err)
				assert.Nil(t, got)
				var typedErr errcommon.Error
				require.ErrorAs(t, err, &typedErr)
				assert.Equal(t, errcommon.ServiceUnavailable, typedErr.Code)
				assert.Equal(t, string(errcommon.RequestDroppedReasonNoEndpoints), typedErr.Headers[errcommon.RequestDroppedReasonHeaderKey])
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tc.wantPrimary, got.PrimaryProfileName)
			switch {
			case tc.wantPrimary == fallback:
				require.Len(t, got.ProfileResults, 1)
				require.Equal(t, fallbackEndpoint.GetMetadata().ID, got.PrimaryEndpoint().GetMetadata().ID)
				for _, header := range []string{routing.PrefillEndpointHeader, routing.EncoderEndpointsHeader} {
					routing.SetRoutingHeader(req.Headers, header, "stale-endpoint")
				}
				require.NoError(t, handler.PreRequest(ctx, req, got))
				require.Equal(t, map[string]string{reqcommon.RequestIDHeaderKey: "test-request"}, req.Headers)
			case tc.prefillDeclined:
				assert.NotContains(t, got.ProfileResults, prefill)
			default:
				assert.Contains(t, got.ProfileResults, prefill)
				assert.Equal(t, tc.hasEncode, got.ProfileResults[encode] != nil)
			}
		})
	}
}

type profileFunc func(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) (*fwksched.ProfileRunResult, error)

func (f profileFunc) Run(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	return f(ctx, req, endpoints)
}

func TestPDSchedule_PrefillFirst(t *testing.T) {
	ctx := context.Background()
	scrapedAt := time.Now()

	endpoint1 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "endpoint1"},
			Address: "1.2.3.4",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0, UpdateTime: scrapedAt},
		fwkdl.NewAttributes(),
	)
	endpoint2 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "endpoint2"},
			Address: "5.6.7.8",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RoleDecode},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0, UpdateTime: scrapedAt},
		fwkdl.NewAttributes(),
	)

	prefillDecodeResult := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			decode: {
				TargetEndpoints: []fwksched.Endpoint{
					&fwksched.ScoredEndpoint{
						Endpoint: endpoint2,
					},
				},
			},
			prefill: {
				TargetEndpoints: []fwksched.Endpoint{
					&fwksched.ScoredEndpoint{
						Endpoint: endpoint1,
					},
				},
			},
		},
		PrimaryProfileName: decode,
	}

	prefillSchedulerProfile := scheduling.NewSchedulerProfile().
		WithFilters(bylabel.NewPrefillRole()).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	decodeSchedulerProfile := scheduling.NewSchedulerProfile().
		WithFilters(bylabel.NewDecodeRole()).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	profileHandle := disagg.NewDisaggProfileHandler(decode, prefill, "",
		nil, nil).WithStageOrder(disagg.StageOrderPrefillFirst)

	schedulerConfig := scheduling.NewSchedulerConfig(profileHandle, map[string]fwksched.SchedulerProfile{
		prefill: prefillSchedulerProfile,
		decode:  decodeSchedulerProfile,
	})
	scheduler := scheduling.NewSchedulerWithConfig(schedulerConfig)

	req := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "critical",
		Body:        completionsBody("12345678906"),
	}
	input := []fwksched.Endpoint{endpoint1, endpoint2}

	got, err := scheduler.Schedule(ctx, req, input)
	assert.NoError(t, err)

	if diff := cmp.Diff(prefillDecodeResult, got, cmpopts.IgnoreUnexported(fwkdl.Attributes{}), cmpopts.IgnoreFields(fwksched.ScoredEndpoint{}, "Score"),
		cmpopts.IgnoreFields(fwksched.ProfileRunResult{}, "ScoredCandidates")); diff != "" {
		t.Errorf("Unexpected output (-want +got): %v", diff)
	}
}
