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

func TestFallbackHandler_ValidateWrappedProfile(t *testing.T) {
	profile := scheduling.NewSchedulerProfile()
	profiles := map[string]fwksched.SchedulerProfile{"backup": profileFunc(profile.Run)}
	handler := disagg.NewDisaggProfileHandler(decode, prefill, "", nil, nil)

	require.ErrorContains(t, handler.WithFallbackProfile("backup").ValidateSchedulingProfiles(profiles),
		`profile "backup" must implement PoolAwareSchedulerProfile`)
}

func TestPDSchedule_AggregatedFallback(t *testing.T) {
	const fallback = "fallback"
	const encode = "encode"
	roles := map[string]string{decode: bylabel.RoleDecode, prefill: bylabel.RolePrefill, fallback: bylabel.RolePrefillDecode, encode: bylabel.RoleEncode}
	capacity, err := utilization.Factory("capacity", json.NewDecoder(strings.NewReader(
		`{"conditions":[{"metric":"running-requests","maxValue":0}]}`)), nil)
	require.NoError(t, err)
	newProfile := func(filters ...fwksched.Filter) *scheduling.SchedulerProfile {
		return scheduling.NewSchedulerProfile().WithFilters(filters...).WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
	}
	selectors := make(map[string]fwksched.Filter, len(roles))
	for name, role := range roles {
		selector, err := bylabel.NewSelector(name, &metav1.LabelSelector{MatchLabels: map[string]string{bylabel.RoleLabel: role}})
		require.NoError(t, err)
		selectors[name] = selector
	}
	capacityFilter := capacity.(fwksched.Filter)
	decodeProfile := newProfile(capacityFilter, selectors[decode])
	prefillProfile := newProfile(selectors[prefill], capacityFilter)
	fallbackProfile := newProfile(selectors[fallback], capacityFilter)
	encodeProfile := newProfile(bylabel.NewEncodeRole())

	for _, tc := range []struct {
		name             string
		hasDecode        bool
		hasPrefill       bool
		hasFallback      bool
		encodeConfigured bool
		hasEncode        bool
		decodeBusy       bool
		prefillBusy      bool
		fallbackBusy     bool
		prefillDeclined  bool
		cancelled        bool
		wantPrimary      string
		wantFallbackRuns int
		opaqueDecode     bool
	}{
		{name: "decode unavailable", hasPrefill: true, hasFallback: true, wantPrimary: fallback, wantFallbackRuns: 1},
		{name: "required prefill unavailable", hasDecode: true, hasFallback: true, wantPrimary: fallback, wantFallbackRuns: 1},
		{name: "prefill declined", hasDecode: true, hasFallback: true, prefillDeclined: true, wantPrimary: decode},
		{name: "ready decode at capacity", hasDecode: true, hasFallback: true, decodeBusy: true},
		{name: "ready required prefill at capacity", hasDecode: true, hasPrefill: true, hasFallback: true, prefillBusy: true},
		{name: "opaque decode profile fails closed", hasFallback: true, opaqueDecode: true},
		{name: "fallback at capacity", hasFallback: true, fallbackBusy: true, wantFallbackRuns: 1},
		{name: "request cancelled", hasFallback: true, cancelled: true},
		{name: "encode unavailable retains PD", hasDecode: true, hasPrefill: true, hasFallback: true, encodeConfigured: true, wantPrimary: decode},
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
			endpoints := make([]fwksched.Endpoint, 0, len(roles))
			for _, worker := range []struct {
				name          string
				present, busy bool
			}{
				{decode, tc.hasDecode, tc.decodeBusy}, {prefill, tc.hasPrefill, tc.prefillBusy},
				{fallback, tc.hasFallback, tc.fallbackBusy}, {encode, tc.hasEncode, false},
			} {
				if !worker.present {
					continue
				}
				metrics := &fwkdl.Metrics{}
				if worker.busy {
					metrics.RunningRequestsSize = 1
				}
				endpoints = append(endpoints, fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
					ID: k8stypes.NamespacedName{Name: worker.name}, Labels: map[string]string{bylabel.RoleLabel: roles[worker.name]},
				}, metrics, fwkdl.NewAttributes()))
			}

			fallbackRuns := 0
			profiles := map[string]fwksched.SchedulerProfile{
				decode: decodeProfile,
				prefill: &trackedProfile{
					PoolAwareSchedulerProfile: prefillProfile,
					run: func(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
						_, hasPeer := fwksched.ReadRequestAttribute[fwksched.Endpoint](req, disagg.PeerEndpointAttributeKey)
						assert.True(t, hasPeer)
						return prefillProfile.Run(ctx, req, endpoints)
					},
				},
				fallback: &trackedProfile{
					PoolAwareSchedulerProfile: fallbackProfile,
					run: func(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
						fallbackRuns++
						_, hasPeer := fwksched.ReadRequestAttribute[fwksched.Endpoint](req, disagg.PeerEndpointAttributeKey)
						assert.False(t, hasPeer)
						return fallbackProfile.Run(ctx, req, endpoints)
					},
				},
			}
			if tc.opaqueDecode {
				profiles[decode] = profileFunc(decodeProfile.Run)
			}
			encodeRuns := 0
			if tc.encodeConfigured {
				profiles[encode] = profileFunc(func(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
					encodeRuns++
					return encodeProfile.Run(ctx, req, endpoints)
				})
			}
			scheduler := scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(handler, profiles))

			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if tc.cancelled {
				cancel()
			}
			req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "critical", Body: completionsBody("12345678901"), Headers: map[string]string{reqcommon.RequestIDHeaderKey: "test-request"}}
			if tc.encodeConfigured {
				req.Body.TokenizedRequest.Prompts[0].MultiModalFeatures = []fwkrequest.MultiModalFeature{{Modality: reqcommon.ModalityImage, Hash: "image", Length: 1}}
			}
			counter := disagg.LlmdDisaggDecisionCount.WithLabelValues(handler.TypedName().Name, handler.TypedName().Type, req.TargetModel, metricsutil.DisaggPathFallback)
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
				require.Equal(t, fallback, got.PrimaryEndpoint().GetMetadata().ID.Name)
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

type trackedProfile struct {
	fwksched.PoolAwareSchedulerProfile
	run profileFunc
}

func (p *trackedProfile) Run(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	return p.run(ctx, req, endpoints)
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

func TestPDSchedule_DecoratedHandler(t *testing.T) {
	handler := &decoratedHandler{Handler: disagg.NewDisaggProfileHandler(decode, prefill, "", nil, nil)}
	profile := scheduling.NewSchedulerProfile().WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
	scheduler := scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(handler, map[string]fwksched.SchedulerProfile{decode: profile}))
	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "decode"}}, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	result, err := scheduler.Schedule(context.Background(), &fwksched.InferenceRequest{RequestID: uuid.NewString()}, []fwksched.Endpoint{endpoint})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, 2, handler.picks)
}

type decoratedHandler struct {
	*disagg.Handler
	picks int
}

func (h *decoratedHandler) Pick(ctx context.Context, request *fwksched.InferenceRequest,
	profiles map[string]fwksched.SchedulerProfile, results map[string]*fwksched.ProfileRunResult,
) map[string]fwksched.SchedulerProfile {
	h.picks++
	return h.Handler.Pick(ctx, request, profiles, results)
}
