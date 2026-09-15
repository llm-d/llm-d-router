package disagg_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/go-logr/logr/testr"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log" // Import config for thresholds

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
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

type rejectAllFilter struct{}

func (*rejectAllFilter) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "reject-all", Name: "reject-all"}
}

func (*rejectAllFilter) Filter(_ context.Context, _ *fwksched.InferenceRequest, _ []fwksched.Endpoint) []fwksched.Endpoint {
	return nil
}

// completionsBody builds a completions request body whose tokenized prompt carries
// len(prompt)/averageCharactersPerToken token IDs, which the decider reads as
// the input token count.
func completionsBody(prompt string) *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{
		Completions:      &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}},
		TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: make([]uint32, len(prompt)/averageCharactersPerToken)}}},
	}
}

// Tests the scheduler expected behavior.
func TestPDSchedule(t *testing.T) {
	endpoint1 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "endpoint1"},
			Address: "1.2.3.4",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0},
		fwkdl.NewAttributes(),
	)
	endpoint2 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "endpoint2"},
			Address: "5.6.7.8",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RoleDecode},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0},
		fwkdl.NewAttributes(),
	)
	noRoleEndpoint1 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "noRoleEndpoint1"},
			Address: "1.1.1.1",
		},
		&fwkdl.Metrics{WaitingQueueSize: 2},
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

	prefillProfile := scheduling.NewSchedulerProfile().
		WithFilters(bylabel.NewByLabel("strict-prefill", bylabel.RoleLabel, false, bylabel.RolePrefill)).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
	decodeProfile := scheduling.NewSchedulerProfile().
		WithFilters(bylabel.NewByLabel("strict-decode", bylabel.RoleLabel, false, bylabel.RoleDecode)).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
	fallbackProfile := scheduling.NewSchedulerProfile().
		WithFilters(bylabel.NewByLabel("full-capability", bylabel.RoleLabel, false, bylabel.RolePrefillDecode)).
		WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
	handler := disagg.NewDisaggProfileHandler(decode, prefill, "", &disagg.AlwaysDisaggPDDecider{}, nil).
		WithFallbackProfile(fallback)
	scheduler := scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(handler, map[string]fwksched.SchedulerProfile{
		decode: decodeProfile, prefill: prefillProfile, fallback: fallbackProfile,
	}))
	for _, tc := range []struct {
		name      string
		endpoints []fwksched.Endpoint
		primary   string
	}{
		{"decode unavailable", []fwksched.Endpoint{fallbackEndpoint, prefillOnlyEndpoint}, fallback},
		{"decode recovered", []fwksched.Endpoint{fallbackEndpoint, prefillOnlyEndpoint, decodeEndpoint}, decode},
		{"prefill unavailable", []fwksched.Endpoint{fallbackEndpoint, decodeEndpoint}, fallback},
		{"prefill recovered", []fwksched.Endpoint{fallbackEndpoint, prefillOnlyEndpoint, decodeEndpoint}, decode},
		{"prefill-only cannot satisfy fallback selector", []fwksched.Endpoint{prefillOnlyEndpoint}, ""},
		{"fallback unavailable", []fwksched.Endpoint{decodeEndpoint}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "critical", Body: completionsBody("12345678901")}
			got, err := scheduler.Schedule(ctx, req, tc.endpoints)
			if tc.primary == "" {
				require.Error(t, err)
				assert.Nil(t, got)
				var typedErr errcommon.Error
				require.ErrorAs(t, err, &typedErr)
				assert.Equal(t, errcommon.ServiceUnavailable, typedErr.Code)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.primary, got.PrimaryProfileName)
			if tc.primary == fallback {
				require.Len(t, got.ProfileResults, 1)
				assert.Equal(t, "fallback-endpoint", got.ProfileResults[fallback].TargetEndpoints[0].GetMetadata().ID.Name)
			} else {
				assert.Contains(t, got.ProfileResults, prefill)
				assert.NotContains(t, got.ProfileResults, fallback)
			}
		})
	}

	drainedProfile := scheduling.NewSchedulerProfile().WithFilters(&rejectAllFilter{})
	drainedScheduler := scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(handler, map[string]fwksched.SchedulerProfile{
		decode: drainedProfile, prefill: prefillProfile, fallback: fallbackProfile,
	}))
	got, err := drainedScheduler.Schedule(ctx, &fwksched.InferenceRequest{}, []fwksched.Endpoint{fallbackEndpoint, decodeEndpoint})
	require.NoError(t, err)
	assert.Equal(t, fallback, got.PrimaryProfileName)
}

type profileFunc func(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) (*fwksched.ProfileRunResult, error)

func (f profileFunc) Run(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	return f(ctx, req, endpoints)
}

func TestPDSchedule_FallbackErrors(t *testing.T) {
	noEndpoints := errcommon.Error{Code: errcommon.ServiceUnavailable,
		Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonNoEndpoints)}}
	for _, stage := range []string{decode, prefill} {
		for _, fallbackName := range []string{"a-fallback", "z-fallback"} {
			for _, tc := range []struct {
				name         string
				stageErr     error
				wantFallback bool
			}{
				{"no endpoints", fmt.Errorf("wrapped: %w", noEndpoints), true},
				{"capacity rejection", errcommon.Error{Code: errcommon.ResourceExhausted}, false},
				{"unrelated unavailable", errcommon.Error{Code: errcommon.ServiceUnavailable}, false},
				{"unexpected error", errors.New("profile failed"), false},
			} {
				t.Run(stage+"/"+fallbackName+"/"+tc.name, func(t *testing.T) {
					fallbackErr := errcommon.Error{Code: errcommon.ResourceExhausted, Msg: "fallback rejected"}
					attempts := 0
					handler := disagg.NewDisaggProfileHandler(decode, prefill, "", &disagg.AlwaysDisaggPDDecider{}, nil).WithFallbackProfile(fallbackName)
					profiles := map[string]fwksched.SchedulerProfile{
						decode: profileFunc(func(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
							return &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, fwkdl.NewAttributes())}}, nil
						}),
						fallbackName: profileFunc(func(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
							attempts++
							return nil, fallbackErr
						}),
					}
					profiles[stage] = profileFunc(func(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
						return nil, tc.stageErr
					})
					scheduler := scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(handler, profiles))
					result, err := scheduler.Schedule(context.Background(), &fwksched.InferenceRequest{}, nil)
					require.Error(t, err)
					assert.Nil(t, result)
					if tc.wantFallback {
						assert.Equal(t, 1, attempts)
						var typedErr errcommon.Error
						require.ErrorAs(t, err, &typedErr)
						assert.Equal(t, fallbackErr, typedErr)
					} else {
						assert.Zero(t, attempts)
					}
				})
			}
		}
	}
}

func TestPDSchedule_PrefillFirst(t *testing.T) {
	ctx := context.Background()

	endpoint1 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "endpoint1"},
			Address: "1.2.3.4",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0},
		fwkdl.NewAttributes(),
	)
	endpoint2 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: "endpoint2"},
			Address: "5.6.7.8",
			Labels:  map[string]string{bylabel.RoleLabel: bylabel.RoleDecode},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0},
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
