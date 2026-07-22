/*
Copyright 2026 The Kubernetes Authors.

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
	"fmt"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// HeaderPhaseProfileHandlerType is the type of the HeaderPhaseProfileHandler.
	HeaderPhaseProfileHandlerType = "header-phase-profile-handler"

	// defaultHeaderName is the request header read when parameters.HeaderName is empty.
	defaultHeaderName = "EPP-Phase"
)

// compile-time type assertion
var _ fwksched.ProfileHandler = &HeaderPhaseProfileHandler{}

// parameters configures the HeaderPhaseProfileHandler.
type parameters struct {
	// HeaderName is the request header whose value names the scheduling profile to run.
	// Defaults to "EPP-Phase" when empty.
	HeaderName string `json:"headerName"`
}

// Factory defines the factory function for HeaderPhaseProfileHandler.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' profile handler - %w", HeaderPhaseProfileHandlerType, err)
		}
	}

	headerName := params.HeaderName
	if headerName == "" {
		headerName = defaultHeaderName
	}

	return NewHeaderPhaseProfileHandler(headerName).WithName(name), nil
}

// NewHeaderPhaseProfileHandler initializes a new HeaderPhaseProfileHandler and returns its pointer.
func NewHeaderPhaseProfileHandler(headerName string) *HeaderPhaseProfileHandler {
	return &HeaderPhaseProfileHandler{
		typedName:  fwkplugin.TypedName{Type: HeaderPhaseProfileHandlerType, Name: HeaderPhaseProfileHandlerType},
		headerName: headerName,
	}
}

// HeaderPhaseProfileHandler runs exactly one scheduling profile per request: the one
// named by the value of a request header. This lets a single EPP instance serve several
// phases of a disaggregated pipeline (e.g. encode, prefill, decode) whose caller already
// knows, out of band, which phase each request is for - unlike the disagg profile
// handler, which decides which profiles to run via decider plugins.
type HeaderPhaseProfileHandler struct {
	typedName  fwkplugin.TypedName
	headerName string
}

// TypedName returns the type and name tuple of this plugin instance.
func (h *HeaderPhaseProfileHandler) TypedName() fwkplugin.TypedName {
	return h.typedName
}

// WithName sets the name of the profile handler.
func (h *HeaderPhaseProfileHandler) WithName(name string) *HeaderPhaseProfileHandler {
	h.typedName.Name = name
	return h
}

// Pick selects the single SchedulingProfile named by the request's phase header. It
// returns an empty map once that profile has run, or when the header is missing or
// names a profile that isn't configured (ProcessResults then reports the error).
func (h *HeaderPhaseProfileHandler) Pick(_ context.Context, request *fwksched.InferenceRequest, profiles map[string]fwksched.SchedulerProfile,
	profileResults map[string]*fwksched.ProfileRunResult) map[string]fwksched.SchedulerProfile {
	// ponytail: single profile per request; extend to loop over an ordered phase
	// list parsed from the header for non-deferred multi-profile scheduling (#2135).
	if len(profileResults) > 0 { // the selected profile has already run
		return map[string]fwksched.SchedulerProfile{}
	}

	phase := request.Headers[h.headerName]
	profile, ok := profiles[phase]
	if !ok {
		return map[string]fwksched.SchedulerProfile{}
	}

	return map[string]fwksched.SchedulerProfile{phase: profile}
}

// ProcessResults handles the outcome of the single profile run selected by Pick.
// It specifies in the SchedulingResult the key of the primary profile that should be
// used to get the request's selected destination.
func (h *HeaderPhaseProfileHandler) ProcessResults(_ context.Context, request *fwksched.InferenceRequest,
	profileResults map[string]*fwksched.ProfileRunResult) (*fwksched.SchedulingResult, error) {
	// ponytail: single profile per request; extend by re-parsing the header's
	// ordered phase list to pick a primary among multiple results for
	// non-deferred multi-profile scheduling (#2135).
	switch len(profileResults) {
	case 0:
		var phase string
		if request != nil {
			phase = request.Headers[h.headerName]
		}
		if phase == "" {
			return nil, fmt.Errorf("header-phase profile handler: missing %q header", h.headerName)
		}
		return nil, fmt.Errorf("header-phase profile handler: no scheduling profile configured for %q header value %q", h.headerName, phase)
	case 1:
		// exactly one profile ran, handled below
	default:
		return nil, fmt.Errorf("header-phase profile handler is intended to run a single profile per request, got %d", len(profileResults))
	}

	var profileName string
	for name := range profileResults {
		profileName = name
	}

	if profileResults[profileName] == nil { // there was an error while running the profile
		return nil, fmt.Errorf("failed to run scheduler profile '%s'", profileName)
	}

	return &fwksched.SchedulingResult{
		ProfileResults:     profileResults,
		PrimaryProfileName: profileName,
	}, nil
}
