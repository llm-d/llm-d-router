package disagg_test

import (
	"context"
	"testing"

	"github.com/go-logr/logr/testr"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log" // Import config for thresholds

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
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
)

// Tests the scheduler expected behavior.
func TestPDSchedule(t *testing.T) {
	endpoint1 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: "endpoint1"},
			Address:        "1.2.3.4",
			Labels:         map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0},
		fwkdl.NewAttributes(),
	)
	endpoint2 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: "endpoint2"},
			Address:        "5.6.7.8",
			Labels:         map[string]string{bylabel.RoleLabel: bylabel.RoleDecode},
		},
		&fwkdl.Metrics{WaitingQueueSize: 0},
		fwkdl.NewAttributes(),
	)
	noRoleEndpoint1 := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: "noRoleEndpoint1"},
			Address:        "1.1.1.1",
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
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "12345678901"},
					},
				},
			},
			input: []fwksched.Endpoint{},
			err:   true,
		},
		{
			name: "one decode endpoint, long prompt",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "critical",
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "12345678901"},
					},
				},
			},
			// endpoint2 will be picked because it is the only endpoint with Decode role
			input:   []fwksched.Endpoint{endpoint2},
			wantRes: decodeResult,
		},
		{
			name: "one prefill endpoint, long prompt",
			req: &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "critical",
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "12345678901"},
					},
				},
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
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "12345678906"},
					},
				},
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
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "12345"},
					},
				},
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
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "12345678901"},
					},
				},
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
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: "1234567890123456789012345678901234567890"},
					},
				},
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
			prefixScorer, err := prefix.New(ctx)
			assert.NoError(t, err, "Prefix plugin creation returned unexpected error")

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

			inputTokens := len(test.req.Body.Completions.Prompt.Raw) / disagg.AverageCharactersPerToken
			for _, pod := range test.input {
				pod.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(0, inputTokens, 1))
			}
			got, err := scheduler.Schedule(ctx, test.req, test.input)

			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			if diff := cmp.Diff(test.wantRes, got, cmpopts.IgnoreUnexported(fwkdl.Attributes{}), cmpopts.IgnoreFields(fwksched.ScoredEndpoint{}, "Score")); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
			if test.wantRes2 != nil { // Checking the prefix match in the decode pod.
				// update number of cached tokens for the following schedule call
				for _, pod := range test.input {
					pod.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(inputTokens, inputTokens, 1))
				}

				got, err = scheduler.Schedule(ctx, test.req, test.input)
				if test.err != (err != nil) {
					t.Errorf("Unexpected error in schedule call, got %v, want %v", err, test.err)
				}

				if diff := cmp.Diff(test.wantRes2, got, cmpopts.IgnoreUnexported(fwkdl.Attributes{}), cmpopts.IgnoreFields(fwksched.ScoredEndpoint{}, "Score")); diff != "" {
					t.Errorf("Unexpected output in subsequent schedule call (-want +got): %v", diff)
				}
			}
		})
	}
}
