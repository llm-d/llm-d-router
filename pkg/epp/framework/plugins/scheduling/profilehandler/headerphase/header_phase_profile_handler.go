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
	"fmt"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// HeaderPhaseProfileHandlerType is the type of the HeaderPhaseProfileHandler.
	HeaderPhaseProfileHandlerType = "header-phase-profile-handler"

	// defaultHeaderName is the request header read when parameters.HeaderName is empty.
	defaultHeaderName = "EPP-Phase"

	// defaultProfileName is the scheduling profile run when parameters.DefaultProfile is
	// empty.
	defaultProfileName = "decode"
)

// compile-time type assertion
var _ fwksched.ProfileHandler = &HeaderPhaseProfileHandler{}

// parameters configures the HeaderPhaseProfileHandler.
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

// Factory defines the factory function for HeaderPhaseProfileHandler.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' profile handler - %w", HeaderPhaseProfileHandlerType, err)
		}
	}

	return NewHeaderPhaseProfileHandler(params.HeaderName, params.DefaultProfile).WithName(name), nil
}

// NewHeaderPhaseProfileHandler initializes a new HeaderPhaseProfileHandler and returns
// its pointer.
//
// headerName is lowercased and trimmed, falling back to defaultHeaderName when that
// leaves it empty: the EPP's request handler lowercases every incoming header name at
// ingestion (pkg/epp/handlers/request.go), so the configured name must be normalized the
// same way to match, and an empty header key would never match any request.
//
// defaultProfile is trimmed, falling back to defaultProfileName when that leaves it
// empty. Unlike headerName, it is not case-normalized: it names a schedulingProfiles
// entry, which (like the header value itself) is matched case-sensitively.
func NewHeaderPhaseProfileHandler(headerName, defaultProfile string) *HeaderPhaseProfileHandler {
	headerName = strings.ToLower(strings.TrimSpace(headerName))
	if headerName == "" {
		headerName = strings.ToLower(defaultHeaderName)
	}

	defaultProfile = strings.TrimSpace(defaultProfile)
	if defaultProfile == "" {
		defaultProfile = defaultProfileName
	}

	return &HeaderPhaseProfileHandler{
		typedName:      fwkplugin.TypedName{Type: HeaderPhaseProfileHandlerType, Name: HeaderPhaseProfileHandlerType},
		headerName:     headerName,
		defaultProfile: defaultProfile,
	}
}

// HeaderPhaseProfileHandler runs exactly one scheduling profile per request: the one
// named by the value of a request header. This lets a single EPP instance serve several
// phases of a disaggregated pipeline (e.g. encode, prefill, decode) whose caller already
// knows, out of band, which phase each request is for - unlike the disagg profile
// handler, which decides which profiles to run via decider plugins.
//
// Two fallbacks keep single-stage and header-less traffic working without a different
// profile handler: with exactly one configured profile there is nothing to disaggregate,
// so that profile always runs regardless of the header (or its absence); with more than
// one configured profile, a request whose header is missing or blank runs defaultProfile
// instead of failing. A header naming a profile that isn't configured is still an error -
// only the header's absence triggers the default, not an unrecognized value.
type HeaderPhaseProfileHandler struct {
	typedName      fwkplugin.TypedName
	headerName     string
	defaultProfile string
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

// phaseHeader returns the trimmed value of the phase header, or "" when request is
// nil or the header is absent or blank. Trimming avoids surprising lookup failures
// when the header carries incidental leading/trailing whitespace.
func (h *HeaderPhaseProfileHandler) phaseHeader(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	return strings.TrimSpace(request.Headers[h.headerName])
}

// noMatchError explains why no configured scheduling profile matches phase, the
// already-trimmed value of the phase header.
func (h *HeaderPhaseProfileHandler) noMatchError(phase string) error {
	if phase == "" {
		return fmt.Errorf("header-phase profile handler: missing %q header", h.headerName)
	}
	return fmt.Errorf("header-phase profile handler: no scheduling profile configured for %q header value %q", h.headerName, phase)
}

// defaultProfileNotConfiguredError explains that defaultProfile itself isn't a
// configured scheduling profile. Distinct from noMatchError: this is a configuration
// bug that fails every request whose phase header is missing or blank, not an ordinary
// client-side unrecognized header value.
func (h *HeaderPhaseProfileHandler) defaultProfileNotConfiguredError() error {
	return fmt.Errorf("header-phase profile handler: defaultProfile %q is not a configured scheduling profile", h.defaultProfile)
}

// Pick selects the single SchedulingProfile to run: the only configured profile when
// there is just one, otherwise the one named by the request's phase header, falling
// back to defaultProfile when the header is missing or blank. It returns an empty map
// once that profile has run, or when no profile could be resolved. In the latter case
// the scheduler's run loop (pkg/epp/scheduling.Scheduler.Schedule) stops without ever
// calling ProcessResults, so the specific reason is logged here rather than returned
// from ProcessResults, where it would be unreachable. The client never sees that
// reason: it only gets the scheduler's generic "failed to run any scheduler profile"
// error, which pkg/epp/requestcontrol/director.go maps to a 429 ResourceExhausted
// response - misleading, since a malformed or missing header is a client error, not a
// capacity problem. Surfacing the real reason to the client needs a
// scheduler/ProfileHandler contract change and is out of scope here; the log is a
// diagnostic aid for operators, not an equivalent substitute for what the caller
// receives.
func (h *HeaderPhaseProfileHandler) Pick(ctx context.Context, request *fwksched.InferenceRequest, profiles map[string]fwksched.SchedulerProfile,
	profileResults map[string]*fwksched.ProfileRunResult) map[string]fwksched.SchedulerProfile {
	if len(profileResults) > 0 { // the selected profile has already run
		return map[string]fwksched.SchedulerProfile{}
	}

	// With exactly one configured profile there is nothing to disaggregate: always run
	// it, so a deployment scaled down to a single stage works without swapping profile
	// handlers or requiring every caller to send the header.
	if len(profiles) == 1 {
		for name, profile := range profiles {
			return map[string]fwksched.SchedulerProfile{name: profile}
		}
	}

	phase := h.phaseHeader(request)
	resolvedPhase := phase
	if resolvedPhase == "" {
		resolvedPhase = h.defaultProfile
	}

	profile, ok := profiles[resolvedPhase]
	if !ok {
		// A missing or unrecognized header value is a per-request client issue, not a
		// system fault - log at DEBUG so it doesn't page anyone or drown out real
		// errors, matching how parseSLOHeaders logs a malformed client header
		// (pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/plugin.go).
		// A missing header whose defaultProfile substitute also fails to resolve is a
		// distinct, config-time condition - it fails every header-less request, not just
		// an occasional bad caller - so it gets its own message rather than being
		// reported as an ordinary missing header.
		err := h.noMatchError(phase)
		if phase == "" {
			err = h.defaultProfileNotConfiguredError()
		}
		log.FromContext(ctx).V(logging.DEBUG).Error(err, "no scheduling profile selected for request")
		return map[string]fwksched.SchedulerProfile{}
	}

	return map[string]fwksched.SchedulerProfile{resolvedPhase: profile}
}

// ProcessResults handles the outcome of the single profile run selected by Pick.
// It specifies in the SchedulingResult the key of the primary profile that should be
// used to get the request's selected destination.
func (h *HeaderPhaseProfileHandler) ProcessResults(_ context.Context, request *fwksched.InferenceRequest,
	profileResults map[string]*fwksched.ProfileRunResult) (*fwksched.SchedulingResult, error) {
	switch len(profileResults) {
	case 0:
		return nil, h.noMatchError(h.phaseHeader(request))
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
