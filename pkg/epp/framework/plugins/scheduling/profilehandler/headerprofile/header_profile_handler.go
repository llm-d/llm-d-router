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

package headerprofile

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// HeaderProfileHandlerType is the type of the HeaderProfileHandler.
	HeaderProfileHandlerType = "header-profile-handler"

	// defaultHeaderName is the request header read when parameters.HeaderName is empty.
	// Kept mixed-case for the README and for tests that use it as a mixed-case
	// constructor input; defaultHeaderNameLower is the form actually used as a header
	// key.
	defaultHeaderName = "EPP-Profile"

	// defaultProfileName is the scheduling profile run when parameters.DefaultProfile is
	// empty.
	defaultProfileName = "decode"
)

// defaultHeaderNameLower is defaultHeaderName normalized once at init, the same way
// NewHeaderProfileHandler normalizes any configured headerName.
var defaultHeaderNameLower = strings.ToLower(defaultHeaderName)

// compile-time type assertion
var _ fwksched.ProfileHandler = &HeaderProfileHandler{}

// parameters configures the HeaderProfileHandler.
type parameters struct {
	// HeaderName is the request header whose value names the scheduling profile to run.
	// Defaults to defaultHeaderName when empty.
	HeaderName string `json:"headerName"`
	// DefaultProfile is the scheduling profile to run when the header is missing or
	// blank. Defaults to defaultProfileName when empty. Useful for requests that never
	// carry the header at all - pass-through calls (e.g. /models) or a deployment that
	// doesn't disaggregate.
	DefaultProfile string `json:"defaultProfile"`
}

// HeaderProfileHandlerFactory defines the factory function for HeaderProfileHandler.
func HeaderProfileHandlerFactory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' profile handler - %w", HeaderProfileHandlerType, err)
		}
	}

	return NewHeaderProfileHandler(params.HeaderName, params.DefaultProfile).WithName(name), nil
}

// NewHeaderProfileHandler trims and defaults both arguments, additionally
// lowercasing headerName to match how the EPP stores ingested header keys
// (pkg/epp/handlers/request.go); defaultProfile stays case-sensitive since it names a
// schedulingProfiles entry.
func NewHeaderProfileHandler(headerName, defaultProfile string) *HeaderProfileHandler {
	headerName = strings.ToLower(strings.TrimSpace(headerName))
	if headerName == "" {
		headerName = defaultHeaderNameLower
	}

	defaultProfile = strings.TrimSpace(defaultProfile)
	if defaultProfile == "" {
		defaultProfile = defaultProfileName
	}

	return &HeaderProfileHandler{
		typedName:      fwkplugin.TypedName{Type: HeaderProfileHandlerType, Name: HeaderProfileHandlerType},
		headerName:     headerName,
		defaultProfile: defaultProfile,
	}
}

// HeaderProfileHandler runs exactly one scheduling profile per request, named by a
// request header, falling back to the sole configured profile or to defaultProfile when
// there's nothing to choose or the header is missing/blank; an unrecognized header value
// is still an error.
type HeaderProfileHandler struct {
	typedName      fwkplugin.TypedName
	headerName     string
	defaultProfile string
}

// TypedName returns the type and name tuple of this plugin instance.
func (h *HeaderProfileHandler) TypedName() fwkplugin.TypedName {
	return h.typedName
}

// WithName sets the name of the profile handler.
func (h *HeaderProfileHandler) WithName(name string) *HeaderProfileHandler {
	h.typedName.Name = name
	return h
}

// profileHeader returns the trimmed value of the profile-selecting header, or "" when
// request is nil or the header is absent or blank. Trimming avoids surprising lookup
// failures when the header carries incidental leading/trailing whitespace.
func (h *HeaderProfileHandler) profileHeader(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	return strings.TrimSpace(request.Headers[h.headerName])
}

// noMatchError explains why no configured scheduling profile matches profileName, the
// already-trimmed value of the profile-selecting header.
func (h *HeaderProfileHandler) noMatchError(profileName string) error {
	if profileName == "" {
		return fmt.Errorf("header profile handler: missing %q header", h.headerName)
	}
	return fmt.Errorf("header profile handler: no scheduling profile configured for %q header value %q", h.headerName, profileName)
}

// defaultProfileNotConfiguredError reports defaultProfile itself missing from
// schedulingProfiles - a config bug, distinct from noMatchError's per-request issue.
func (h *HeaderProfileHandler) defaultProfileNotConfiguredError() error {
	return fmt.Errorf("header profile handler: defaultProfile %q is not a configured scheduling profile", h.defaultProfile)
}

// Pick implements fwksched.ProfileHandler.Pick; see README.md for behavior.
func (h *HeaderProfileHandler) Pick(ctx context.Context, request *fwksched.InferenceRequest, profiles map[string]fwksched.SchedulerProfile,
	profileResults map[string]*fwksched.ProfileRunResult) map[string]fwksched.SchedulerProfile {
	if len(profileResults) > 0 { // the selected profile has already run
		return map[string]fwksched.SchedulerProfile{}
	}

	// With exactly one configured profile there is nothing to choose: always run
	// it, so a deployment scaled down to a single stage works without swapping profile
	// handlers or requiring every caller to send the header.
	if len(profiles) == 1 {
		for name, profile := range profiles {
			return map[string]fwksched.SchedulerProfile{name: profile}
		}
	}

	profileName := h.profileHeader(request)
	resolvedProfileName := profileName
	if resolvedProfileName == "" {
		resolvedProfileName = h.defaultProfile
	}

	profile, ok := profiles[resolvedProfileName]
	if !ok {
		err := h.noMatchError(profileName)
		if profileName == "" {
			err = h.defaultProfileNotConfiguredError()
		}
		registeredProfiles := slices.Sorted(maps.Keys(profiles))
		// Logged here, not returned from ProcessResults (skipped when Pick returns empty),
		// at Info, not Error, since a bad or missing header is a client issue an operator
		// still needs to see by default. registeredProfiles is included to help spot typos
		// or config/header misalignment.
		// See README.md for why the client only gets a generic 429 instead of this reason.
		log.FromContext(ctx).Info("no scheduling profile selected for request",
			"error", err, "resolvedProfileName", resolvedProfileName, "registeredProfiles", registeredProfiles)
		return map[string]fwksched.SchedulerProfile{}
	}

	return map[string]fwksched.SchedulerProfile{resolvedProfileName: profile}
}

// ProcessResults handles the outcome of the single profile run selected by Pick.
// It specifies in the SchedulingResult the key of the primary profile that should be
// used to get the request's selected destination.
func (h *HeaderProfileHandler) ProcessResults(_ context.Context, request *fwksched.InferenceRequest,
	profileResults map[string]*fwksched.ProfileRunResult) (*fwksched.SchedulingResult, error) {
	switch len(profileResults) {
	case 0:
		return nil, h.noMatchError(h.profileHeader(request))
	case 1:
		// exactly one profile ran, handled below
	default:
		return nil, fmt.Errorf("header profile handler is intended to run a single profile per request, got %d", len(profileResults))
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
