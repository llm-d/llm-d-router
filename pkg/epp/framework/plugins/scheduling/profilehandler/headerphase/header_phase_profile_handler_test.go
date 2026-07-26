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

package headerphase

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// ingestedHeaderKey is the key request.Headers actually carries once the EPP's request
// handler stores it (always lowercased, see pkg/epp/handlers/request.go). Handlers are
// constructed with the mixed-case defaultHeaderName throughout these tests specifically
// to exercise the constructor's normalization against this lowercase key.
const ingestedHeaderKey = "epp-phase"

type fakeSchedulerProfile struct{}

func (f *fakeSchedulerProfile) Run(_ context.Context, _ *fwksched.InferenceRequest, _ []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	return &fwksched.ProfileRunResult{}, nil
}

func TestNewHeaderPhaseProfileHandler(t *testing.T) {
	handler := NewHeaderPhaseProfileHandler(defaultHeaderName, defaultProfileName)

	wantTypedName := fwkplugin.TypedName{
		Type: HeaderPhaseProfileHandlerType,
		Name: HeaderPhaseProfileHandlerType,
	}
	if diff := cmp.Diff(wantTypedName, handler.TypedName()); diff != "" {
		t.Errorf("Unexpected TypedName (-want +got): %s", diff)
	}
}

func TestNewHeaderPhaseProfileHandlerEmptyFallsBackToDefault(t *testing.T) {
	// The constructor itself must uphold the "never empty" invariant, independent of the
	// Factory: an empty headerName/defaultProfile must not produce a handler that can
	// never match any request (request.Headers[""] is always empty, and an empty
	// defaultProfile could never name a real schedulingProfiles entry).
	handler := NewHeaderPhaseProfileHandler("", "")
	if handler.headerName != ingestedHeaderKey {
		t.Errorf("Expected headerName %q, got %q", ingestedHeaderKey, handler.headerName)
	}
	if handler.defaultProfile != defaultProfileName {
		t.Errorf("Expected defaultProfile %q, got %q", defaultProfileName, handler.defaultProfile)
	}
}

func TestHeaderPhaseProfileHandlerFactory(t *testing.T) {
	tests := []struct {
		name               string
		rawParameters      string
		wantHeaderName     string
		wantDefaultProfile string
		wantErr            bool
	}{
		{
			name:               "no parameters, uses default header and default profile",
			rawParameters:      "",
			wantHeaderName:     ingestedHeaderKey,
			wantDefaultProfile: defaultProfileName,
		},
		{
			name:               "custom header name",
			rawParameters:      `{"headerName": "x-phase"}`,
			wantHeaderName:     "x-phase",
			wantDefaultProfile: defaultProfileName,
		},
		{
			name:               "custom header name with mixed case and padding is normalized",
			rawParameters:      `{"headerName": "  X-Custom-Phase  "}`,
			wantHeaderName:     "x-custom-phase",
			wantDefaultProfile: defaultProfileName,
		},
		{
			name:               "whitespace-only header name falls back to the default",
			rawParameters:      `{"headerName": "   "}`,
			wantHeaderName:     ingestedHeaderKey,
			wantDefaultProfile: defaultProfileName,
		},
		{
			name:               "custom default profile",
			rawParameters:      `{"defaultProfile": "prefill"}`,
			wantHeaderName:     ingestedHeaderKey,
			wantDefaultProfile: "prefill",
		},
		{
			name:               "whitespace-only default profile falls back to the default",
			rawParameters:      `{"defaultProfile": "   "}`,
			wantHeaderName:     ingestedHeaderKey,
			wantDefaultProfile: defaultProfileName,
		},
		{
			name:          "malformed json returns an error",
			rawParameters: `{invalid}`,
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// StrictDecoder is what the framework actually passes to factories
			// (DisallowUnknownFields), so use it here too rather than a plain decoder.
			decoder := fwkplugin.StrictDecoder(json.RawMessage(tt.rawParameters))

			plugin, err := Factory("custom-name", decoder, nil)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Factory() expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Factory() returned unexpected error: %v", err)
			}

			handler, ok := plugin.(*HeaderPhaseProfileHandler)
			if !ok {
				t.Fatalf("Expected *HeaderPhaseProfileHandler, got %T", plugin)
			}

			wantTypedName := fwkplugin.TypedName{
				Type: HeaderPhaseProfileHandlerType,
				Name: "custom-name",
			}
			if diff := cmp.Diff(wantTypedName, handler.TypedName()); diff != "" {
				t.Errorf("Unexpected TypedName (-want +got): %s", diff)
			}
			if handler.headerName != tt.wantHeaderName {
				t.Errorf("Expected headerName %q, got %q", tt.wantHeaderName, handler.headerName)
			}
			if handler.defaultProfile != tt.wantDefaultProfile {
				t.Errorf("Expected defaultProfile %q, got %q", tt.wantDefaultProfile, handler.defaultProfile)
			}
		})
	}
}

func TestHeaderPhaseNoMatchError(t *testing.T) {
	handler := NewHeaderPhaseProfileHandler(defaultHeaderName, defaultProfileName)

	tests := []struct {
		name           string
		phase          string
		wantErrContain string
	}{
		{
			name:           "empty phase reports missing header",
			phase:          "",
			wantErrContain: `missing "epp-phase" header`,
		},
		{
			name:           "non-empty phase reports the unconfigured value",
			phase:          "prefill",
			wantErrContain: `no scheduling profile configured for "epp-phase" header value "prefill"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := handler.noMatchError(tt.phase)
			if err == nil {
				t.Fatalf("noMatchError() returned nil, want an error")
			}
			if !strings.Contains(err.Error(), tt.wantErrContain) {
				t.Errorf("noMatchError() = %q, want it to contain %q", err.Error(), tt.wantErrContain)
			}
		})
	}
}

func TestHeaderPhaseDefaultProfileNotConfiguredError(t *testing.T) {
	handler := NewHeaderPhaseProfileHandler(defaultHeaderName, "prefill")

	err := handler.defaultProfileNotConfiguredError()
	if err == nil {
		t.Fatalf("defaultProfileNotConfiguredError() returned nil, want an error")
	}
	wantErrContain := `defaultProfile "prefill" is not a configured scheduling profile`
	if !strings.Contains(err.Error(), wantErrContain) {
		t.Errorf("defaultProfileNotConfiguredError() = %q, want it to contain %q", err.Error(), wantErrContain)
	}
}

func TestHeaderPhaseWithName(t *testing.T) {
	handler := NewHeaderPhaseProfileHandler(defaultHeaderName, defaultProfileName).WithName("renamed")

	if handler.TypedName().Name != "renamed" {
		t.Errorf("Expected Name to be %q, got %q", "renamed", handler.TypedName().Name)
	}
	if handler.TypedName().Type != HeaderPhaseProfileHandlerType {
		t.Errorf("Expected Type to remain %q, got %q", HeaderPhaseProfileHandlerType, handler.TypedName().Type)
	}
}

func TestHeaderPhasePick(t *testing.T) {
	encodeProfile := &fakeSchedulerProfile{}
	decodeProfile := &fakeSchedulerProfile{}
	profiles := map[string]fwksched.SchedulerProfile{
		"encode": encodeProfile,
		"decode": decodeProfile,
	}

	tests := []struct {
		name           string
		request        *fwksched.InferenceRequest
		profiles       map[string]fwksched.SchedulerProfile
		profileResults map[string]*fwksched.ProfileRunResult
		wantProfiles   map[string]fwksched.SchedulerProfile
	}{
		{
			name:           "header names a configured profile, not yet run",
			request:        &fwksched.InferenceRequest{Headers: map[string]string{ingestedHeaderKey: "encode"}},
			profiles:       profiles,
			profileResults: map[string]*fwksched.ProfileRunResult{},
			wantProfiles:   map[string]fwksched.SchedulerProfile{"encode": encodeProfile},
		},
		{
			name:     "selected profile already ran",
			request:  &fwksched.InferenceRequest{Headers: map[string]string{ingestedHeaderKey: "encode"}},
			profiles: profiles,
			profileResults: map[string]*fwksched.ProfileRunResult{
				"encode": {TargetEndpoints: nil},
			},
			wantProfiles: map[string]fwksched.SchedulerProfile{},
		},
		{
			// With more than one profile configured, a missing header falls back to
			// defaultProfile ("decode" here, the package default) rather than failing.
			name:           "missing header falls back to the default profile",
			request:        &fwksched.InferenceRequest{Headers: map[string]string{}},
			profiles:       profiles,
			profileResults: map[string]*fwksched.ProfileRunResult{},
			wantProfiles:   map[string]fwksched.SchedulerProfile{"decode": decodeProfile},
		},
		{
			name:           "header names an unconfigured profile",
			request:        &fwksched.InferenceRequest{Headers: map[string]string{ingestedHeaderKey: "prefill"}},
			profiles:       profiles,
			profileResults: map[string]*fwksched.ProfileRunResult{},
			wantProfiles:   map[string]fwksched.SchedulerProfile{},
		},
		{
			name:           "header value has surrounding whitespace, still matches",
			request:        &fwksched.InferenceRequest{Headers: map[string]string{ingestedHeaderKey: "  encode  "}},
			profiles:       profiles,
			profileResults: map[string]*fwksched.ProfileRunResult{},
			wantProfiles:   map[string]fwksched.SchedulerProfile{"encode": encodeProfile},
		},
		{
			// A whitespace-only value is treated as missing, so it also falls back to
			// the default profile, same as a header that is absent entirely.
			name:           "whitespace-only header value falls back to the default profile",
			request:        &fwksched.InferenceRequest{Headers: map[string]string{ingestedHeaderKey: "   "}},
			profiles:       profiles,
			profileResults: map[string]*fwksched.ProfileRunResult{},
			wantProfiles:   map[string]fwksched.SchedulerProfile{"decode": decodeProfile},
		},
		{
			// Unlike the header name, the header value is matched case-sensitively
			// against the configured profile names: only the name gets normalized to
			// match how the EPP's request handler stores it, since that's a fixed key
			// this plugin controls, but the value is caller-supplied and compared
			// verbatim against schedulingProfiles names.
			name:           "header value case does not match the configured profile name",
			request:        &fwksched.InferenceRequest{Headers: map[string]string{ingestedHeaderKey: "Encode"}},
			profiles:       profiles,
			profileResults: map[string]*fwksched.ProfileRunResult{},
			wantProfiles:   map[string]fwksched.SchedulerProfile{},
		},
		{
			// With exactly one configured profile, it always runs -- even with no
			// header at all -- since there is nothing else to choose. This is
			// what makes a deployment scaled down to a single stage work without
			// swapping to a different profile handler.
			name:           "single configured profile runs with no header",
			request:        &fwksched.InferenceRequest{Headers: map[string]string{}},
			profiles:       map[string]fwksched.SchedulerProfile{"decode": decodeProfile},
			profileResults: map[string]*fwksched.ProfileRunResult{},
			wantProfiles:   map[string]fwksched.SchedulerProfile{"decode": decodeProfile},
		},
		{
			// The single-profile shortcut is unconditional: even a header naming a
			// phase that isn't configured doesn't stop the one configured profile from
			// running.
			name:           "single configured profile runs even with an unrecognized header value",
			request:        &fwksched.InferenceRequest{Headers: map[string]string{ingestedHeaderKey: "prefill"}},
			profiles:       map[string]fwksched.SchedulerProfile{"decode": decodeProfile},
			profileResults: map[string]*fwksched.ProfileRunResult{},
			wantProfiles:   map[string]fwksched.SchedulerProfile{"decode": decodeProfile},
		},
	}

	handler := NewHeaderPhaseProfileHandler(defaultHeaderName, defaultProfileName)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := handler.Pick(context.Background(), tt.request, tt.profiles, tt.profileResults)
			if diff := cmp.Diff(tt.wantProfiles, got); diff != "" {
				t.Errorf("Pick() returned unexpected profiles (-want +got): %s", diff)
			}
		})
	}
}

func TestHeaderPhasePickCustomDefaultProfile(t *testing.T) {
	encodeProfile := &fakeSchedulerProfile{}
	prefillProfile := &fakeSchedulerProfile{}
	profiles := map[string]fwksched.SchedulerProfile{
		"encode":  encodeProfile,
		"prefill": prefillProfile,
	}

	tests := []struct {
		name         string
		request      *fwksched.InferenceRequest
		wantProfiles map[string]fwksched.SchedulerProfile
	}{
		{
			name:         "missing header falls back to the configured custom default",
			request:      &fwksched.InferenceRequest{Headers: map[string]string{}},
			wantProfiles: map[string]fwksched.SchedulerProfile{"prefill": prefillProfile},
		},
		{
			// The default is only a fallback for a missing header; an explicit header
			// still wins even when it names a different profile than the default.
			name:         "explicit header overrides the custom default",
			request:      &fwksched.InferenceRequest{Headers: map[string]string{ingestedHeaderKey: "encode"}},
			wantProfiles: map[string]fwksched.SchedulerProfile{"encode": encodeProfile},
		},
	}

	// defaultProfile "prefill" is configured but is not itself named "decode", proving
	// the fallback uses the configured value rather than the package default.
	handler := NewHeaderPhaseProfileHandler(defaultHeaderName, "prefill")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := handler.Pick(context.Background(), tt.request, profiles, map[string]*fwksched.ProfileRunResult{})
			if diff := cmp.Diff(tt.wantProfiles, got); diff != "" {
				t.Errorf("Pick() returned unexpected profiles (-want +got): %s", diff)
			}
		})
	}
}

func TestHeaderPhasePickDefaultProfileNotConfiguredStillErrors(t *testing.T) {
	// defaultProfile ("decode") names a profile that isn't in schedulingProfiles here;
	// the fallback itself failing must not be mistaken for success, and the reported
	// reason must still describe the real cause (the missing header), not the
	// unconfigured default it tried.
	profiles := map[string]fwksched.SchedulerProfile{
		"encode":  &fakeSchedulerProfile{},
		"prefill": &fakeSchedulerProfile{},
	}
	handler := NewHeaderPhaseProfileHandler(defaultHeaderName, defaultProfileName)
	request := &fwksched.InferenceRequest{Headers: map[string]string{}}

	got := handler.Pick(context.Background(), request, profiles, map[string]*fwksched.ProfileRunResult{})
	if len(got) != 0 {
		t.Fatalf("Pick() returned %d profiles, want 0", len(got))
	}

	_, err := handler.ProcessResults(context.Background(), request, map[string]*fwksched.ProfileRunResult{})
	if err == nil {
		t.Fatalf("ProcessResults() expected error, got nil")
	}
	if !strings.Contains(err.Error(), `missing "epp-phase" header`) {
		t.Errorf("ProcessResults() error = %q, want it to report the missing header, not the failed default", err.Error())
	}
}

func TestHeaderPhaseProcessResults(t *testing.T) {
	successResult := &fwksched.ProfileRunResult{
		TargetEndpoints: nil,
	}

	tests := []struct {
		name            string
		request         *fwksched.InferenceRequest
		profileResults  map[string]*fwksched.ProfileRunResult
		wantResult      *fwksched.SchedulingResult
		wantErr         bool
		wantErrContains string
	}{
		{
			name: "single successful profile",
			profileResults: map[string]*fwksched.ProfileRunResult{
				"encode": successResult,
			},
			wantResult: &fwksched.SchedulingResult{
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"encode": successResult,
				},
				PrimaryProfileName: "encode",
			},
		},
		{
			name:            "no profiles selected, nil request, reports missing header",
			request:         nil,
			profileResults:  map[string]*fwksched.ProfileRunResult{},
			wantErr:         true,
			wantErrContains: `missing "epp-phase" header`,
		},
		{
			name:            "no profiles selected, empty header, reports missing header",
			request:         &fwksched.InferenceRequest{Headers: map[string]string{}},
			profileResults:  map[string]*fwksched.ProfileRunResult{},
			wantErr:         true,
			wantErrContains: `missing "epp-phase" header`,
		},
		{
			name:            "no profiles selected, unconfigured header value, reports the value",
			request:         &fwksched.InferenceRequest{Headers: map[string]string{ingestedHeaderKey: "prefill"}},
			profileResults:  map[string]*fwksched.ProfileRunResult{},
			wantErr:         true,
			wantErrContains: `no scheduling profile configured for "epp-phase" header value "prefill"`,
		},
		{
			name: "multiple profiles returns error",
			profileResults: map[string]*fwksched.ProfileRunResult{
				"encode": successResult,
				"decode": successResult,
			},
			wantErr:         true,
			wantErrContains: "is intended to run a single profile per request, got 2",
		},
		{
			name: "nil result (profile execution failure) returns error",
			profileResults: map[string]*fwksched.ProfileRunResult{
				"encode": nil,
			},
			wantErr:         true,
			wantErrContains: "failed to run scheduler profile 'encode'",
		},
	}

	handler := NewHeaderPhaseProfileHandler(defaultHeaderName, defaultProfileName)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := handler.ProcessResults(context.Background(), tt.request, tt.profileResults)

			if tt.wantErr {
				if err == nil {
					t.Fatalf("ProcessResults() expected error, got nil")
				}
				if tt.wantErrContains != "" && !strings.Contains(err.Error(), tt.wantErrContains) {
					t.Errorf("ProcessResults() error = %q, want it to contain %q", err.Error(), tt.wantErrContains)
				}
				return
			}

			if err != nil {
				t.Fatalf("ProcessResults() unexpected error: %v", err)
			}

			if diff := cmp.Diff(tt.wantResult, got); diff != "" {
				t.Errorf("Unexpected SchedulingResult (-want +got): %s", diff)
			}
		})
	}
}
