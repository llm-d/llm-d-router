// Package disagg provides profile handler plugin for the epp.
package disagg

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strconv"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	schedplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling"
)

const (
	// PdProfileHandlerType is a legacy alias for DisaggProfileHandlerType.
	PdProfileHandlerType     = "pd-profile-handler"
	defaultDeciderPluginName = PrefixBasedPDDeciderPluginType
)

// pdDeciderPlugin interface for pd decider plugins

type pdProfileHandlerParameters struct {
	DecodeProfile  string `json:"decodeProfile"`
	PrefillProfile string `json:"prefillProfile"`
	// Deprecated: This field was never used.
	PrefixPluginType string `json:"prefixPluginType"`
	// Deprecated: This field was never used.
	PrefixPluginName            string `json:"prefixPluginName"`
	PrimaryPort                 int    `json:"primaryPort"`
	DeciderPluginName           string `json:"deciderPluginName"`
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName"`
}

// compile-time type assertion
var _ scheduling.ProfileHandler = &PdProfileHandler{}

// PdProfileHandlerFactory defines the factory function for the PdProfileHandler.
//
// Deprecated: Use HandlerFactory instead.
func PdProfileHandlerFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}
	log.FromContext(handle.Context()).Info("Deprecated: pd-profile-handler is deprecated, use disagg-profile-handler instead")
	parameters := pdProfileHandlerParameters{
		DecodeProfile:     defaultDecodeProfile,
		PrefillProfile:    defaultPrefillProfile,
		PrimaryPort:       0,
		DeciderPluginName: defaultDeciderPluginName,
	}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' profile handler - %w", PdProfileHandlerType, err)
		}
	}

	if parameters.PrefixPluginName == "" {
		parameters.PrefixPluginName = parameters.PrefixPluginType
	}

	if parameters.PrimaryPort != 0 {
		log.FromContext(handle.Context()).Info("Deprecated: primaryPort not needed with Istio >= 1.28.1")
		if parameters.PrimaryPort < 1 || parameters.PrimaryPort > 65535 {
			return nil, fmt.Errorf("invalid primaryPort: must be between 1 and 65535, got %d", parameters.PrimaryPort)
		}
	}

	if parameters.DeciderPluginName == "" {
		return nil, errors.New("decider plugin name is not defined")
	}

	plugin := handle.Plugin(parameters.DeciderPluginName)
	if plugin == nil {
		return nil, fmt.Errorf("invalid decider plugin type: %s", parameters.DeciderPluginName)
	}

	deciderPlugin, ok := plugin.(deciderPlugin)
	if !ok {
		return nil, fmt.Errorf("decider plugin of type: %s does not implement pdDeciderPlugin", parameters.DeciderPluginName)
	}

	handler, err := NewPdProfileHandler(name, parameters, deciderPlugin)

	if err != nil {
		return nil, err
	}

	return handler.WithName(name), nil

}

// NewPdProfileHandler initializes a new PdProfileHandler and returns its pointer.
//
// Deprecated: Use NewDisaggProfileHandler instead.
func NewPdProfileHandler(name string, parameters pdProfileHandlerParameters, deciderPlugin deciderPlugin) (*PdProfileHandler, error) {
	result := &PdProfileHandler{
		typedName:      plugin.TypedName{Name: name, Type: PdProfileHandlerType},
		decodeProfile:  parameters.DecodeProfile,
		prefillProfile: parameters.PrefillProfile,
		decider:        deciderPlugin,
		dk:             attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(parameters.PrefixMatchInfoProducerName),
	}
	if parameters.PrimaryPort != 0 {
		result.primaryPort = strconv.Itoa(parameters.PrimaryPort)
	}

	return result, nil
}

// PdProfileHandler handles scheduler profiles for PD.
//
// Deprecated: Use Handler instead.
type PdProfileHandler struct {
	typedName      plugin.TypedName
	decodeProfile  string
	prefillProfile string
	primaryPort    string
	decider        deciderPlugin
	dk             plugin.DataKey
}

// Consumes defines data types consumed by this plugin (through the PD decider).
func (h *PdProfileHandler) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{
			h.dk:                                 attrprefix.PrefixCacheMatchInfo{},
			tokenproducer.TokenizedPromptDataKey: scheduling.TokenizedPrompt{},
		},
	}
}

// TypedName returns the typed name of the plugin.
func (h *PdProfileHandler) TypedName() plugin.TypedName {
	return h.typedName
}

// WithName sets the name of the plugin.
func (h *PdProfileHandler) WithName(name string) *PdProfileHandler {
	h.typedName.Name = name
	return h
}

// Pick selects the SchedulingProfiles to run from the list of candidate profiles, while taking into consideration the request properties and the
// previously executed cycles along with their results.
func (h *PdProfileHandler) Pick(ctx context.Context, request *scheduling.InferenceRequest, profiles map[string]scheduling.SchedulerProfile,
	profileResults map[string]*scheduling.ProfileRunResult) map[string]scheduling.SchedulerProfile {
	// Start tracing span for profile picking operation
	tracer := tracing.Tracer(schedplugins.TracerScope)
	ctx, span := tracer.Start(ctx, "pick_pd_profile",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	// Set initial attributes
	span.SetAttributes(
		attribute.Int("llm_d.profile_handler.total_profiles", len(profiles)),
		attribute.Int("llm_d.profile_handler.executed_profiles", len(profileResults)),
	)

	// Set optional request attributes if request is not nil
	if request != nil {
		if request.TargetModel != "" {
			span.SetAttributes(attribute.String("gen_ai.request.model", request.TargetModel))
		}
		if request.RequestID != "" {
			span.SetAttributes(attribute.String("gen_ai.request.id", request.RequestID))
		}
	}

	if _, executed := profileResults[h.decodeProfile]; !executed {
		// if decode profile was not executed yet, first let the scheduler run the decode profile
		span.SetAttributes(
			attribute.String("llm_d.profile_handler.decision", "run_decode"),
			attribute.String("llm_d.profile_handler.selected_profile", h.decodeProfile),
		)
		return map[string]scheduling.SchedulerProfile{
			h.decodeProfile: profiles[h.decodeProfile],
		}
	}
	// otherwise, decode was already executed.

	// when a profile run fails its result value is nil. we need to check decode result before continuing to prefill
	// check if all configured profiles have been executed, or if decode failed, no need to run more profiles.
	if len(profiles) == len(profileResults) || profileResults[h.decodeProfile] == nil {
		span.SetAttributes(
			attribute.String("llm_d.profile_handler.decision", "complete"),
			attribute.Bool("llm_d.profile_handler.decode_failed", profileResults[h.decodeProfile] == nil),
		)
		return map[string]scheduling.SchedulerProfile{}
	}

	if h.decider != nil && h.decider.disaggregate(ctx, request, profileResults[h.decodeProfile].TargetEndpoints[0]) {
		RecordPDDecision(h.typedName.Name, h.typedName.Type, request.TargetModel, DecisionTypePrefillDecode) //nolint:staticcheck // intentional: pd-profile-handler is itself deprecated
		// run the prefill profile
		span.SetAttributes(
			attribute.String("llm_d.profile_handler.decision", "prefill_decode"),
			attribute.String("llm_d.profile_handler.selected_profile", h.prefillProfile),
		)
		return map[string]scheduling.SchedulerProfile{
			h.prefillProfile: profiles[h.prefillProfile],
		}
	}

	RecordPDDecision(h.typedName.Name, h.typedName.Type, request.TargetModel, DecisionTypeDecodeOnly) //nolint:staticcheck // intentional: pd-profile-handler is itself deprecated
	span.SetAttributes(
		attribute.String("llm_d.profile_handler.decision", "decode_only"),
	)
	return map[string]scheduling.SchedulerProfile{} // do not run prefill
}

// ProcessResults handles the outcome of the profile runs after the selected profiles ran.
// In case of an error in any of the profiles, the matching entry in the profileResults will contain nil, to indicate there was
// an error while running the profile.
func (h *PdProfileHandler) ProcessResults(_ context.Context, request *scheduling.InferenceRequest,
	profileResults map[string]*scheduling.ProfileRunResult) (*scheduling.SchedulingResult, error) {
	decodeRunResults := profileResults[h.decodeProfile]
	if decodeRunResults == nil { // if decode profile failed to run, we should fail
		return nil, errors.New("failed to find available decode workers")
	}
	// otherwise, decode ran successfully

	updatedResults := map[string]*scheduling.ProfileRunResult{}

	// Add decode profile to result
	if h.primaryPort != "" {
		// Data Parallel is active

		targetEndpoint := decodeRunResults.TargetEndpoints[0].GetMetadata()
		request.Headers[routing.DataParallelEndpointHeader] = net.JoinHostPort(targetEndpoint.Address, targetEndpoint.Port)

		updatedResult := scheduling.ProfileRunResult{
			TargetEndpoints: []scheduling.Endpoint{},
		}

		for _, target := range decodeRunResults.TargetEndpoints {
			updatedEndpointInfo := target.GetMetadata().Clone()
			updatedEndpointInfo.Port = h.primaryPort
			targetEndpoint := scheduling.NewEndpoint(updatedEndpointInfo, target.GetMetrics().Clone(), nil)
			updatedResult.TargetEndpoints = append(updatedResult.TargetEndpoints, targetEndpoint)
		}
		updatedResults[h.decodeProfile] = &updatedResult
	} else {
		updatedResults[h.decodeProfile] = decodeRunResults
	}

	// if both prefill and decode ran successfully
	if prefillRunResult, exists := profileResults[h.prefillProfile]; exists && prefillRunResult != nil {
		// Add the prefill profile to the results
		updatedResults[h.prefillProfile] = prefillRunResult
	}

	return &scheduling.SchedulingResult{
		PrimaryProfileName: h.decodeProfile,
		ProfileResults:     updatedResults,
	}, nil
}
