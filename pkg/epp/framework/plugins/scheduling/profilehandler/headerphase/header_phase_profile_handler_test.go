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
	handler := NewHeaderPhaseProfileHandler(defaultHeaderName)

	wantTypedName := fwkplugin.TypedName{
		Type: HeaderPhaseProfileHandlerType,
		Name: HeaderPhaseProfileHandlerType,
	}
	if diff := cmp.Diff(wantTypedName, handler.TypedName()); diff != "" {
		t.Errorf("Unexpected TypedName (-want +got): %s", diff)
	}
}

func TestNewHeaderPhaseProfileHandlerEmptyFallsBackToDefault(t *testing.T) {
	// The constructor itself must uphold the "never empty" invariant, independent of
	// Factory: an empty headerName must not produce a handler that can never match any
	// request (request.Headers[""] is always empty).
	handler := NewHeaderPhaseProfileHandler("")
	if handler.headerName != ingestedHeaderKey {
		t.Errorf("Expected headerName %q, got %q", ingestedHeaderKey, handler.headerName)
	}
}

func TestHeaderPhaseProfileHandlerFactory(t *testing.T) {
	tests := []struct {
		name           string
		rawParameters  string
		wantHeaderName string
		wantErr        bool
	}{
		{
			name:           "no parameters, uses default header, normalized to lowercase",
			rawParameters:  "",
			wantHeaderName: ingestedHeaderKey,
		},
		{
			name:           "custom header name",
			rawParameters:  `{"headerName": "x-phase"}`,
			wantHeaderName: "x-phase",
		},
		{
			name:           "custom header name with mixed case and padding is normalized",
			rawParameters:  `{"headerName": "  X-Custom-Phase  "}`,
			wantHeaderName: "x-custom-phase",
		},
		{
			name:           "whitespace-only header name falls back to the default",
			rawParameters:  `{"headerName": "   "}`,
			wantHeaderName: ingestedHeaderKey,
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
		})
	}
}

func TestHeaderPhaseNoMatchError(t *testing.T) {
	handler := NewHeaderPhaseProfileHandler(defaultHeaderName)

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

func TestHeaderPhaseWithName(t *testing.T) {
	handler := NewHeaderPhaseProfileHandler(defaultHeaderName).WithName("renamed")

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
			name:           "missing header",
			request:        &fwksched.InferenceRequest{Headers: map[string]string{}},
			profiles:       profiles,
			profileResults: map[string]*fwksched.ProfileRunResult{},
			wantProfiles:   map[string]fwksched.SchedulerProfile{},
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
			name:           "whitespace-only header value is treated as missing",
			request:        &fwksched.InferenceRequest{Headers: map[string]string{ingestedHeaderKey: "   "}},
			profiles:       profiles,
			profileResults: map[string]*fwksched.ProfileRunResult{},
			wantProfiles:   map[string]fwksched.SchedulerProfile{},
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
	}

	handler := NewHeaderPhaseProfileHandler(defaultHeaderName)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := handler.Pick(context.Background(), tt.request, tt.profiles, tt.profileResults)
			if len(got) != len(tt.wantProfiles) {
				t.Errorf("Pick() returned %d profiles, want %d", len(got), len(tt.wantProfiles))
			}
			for name := range tt.wantProfiles {
				if _, ok := got[name]; !ok {
					t.Errorf("Pick() missing expected profile %q", name)
				}
			}
		})
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

	handler := NewHeaderPhaseProfileHandler(defaultHeaderName)
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
