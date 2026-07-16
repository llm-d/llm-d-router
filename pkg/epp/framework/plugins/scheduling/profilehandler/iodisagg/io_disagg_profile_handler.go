// Package iodisagg provides profile handler plugins for the epp.
package iodisagg

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	schedplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling"
)

// ── Constants ───────────────────────────────────────────────────────────────

const (
	// IODisaggProfileHandlerType is the canonical type for the unified io disaggregation profile handler.
	IODisaggProfileHandlerType = "io-disagg-profile-handler"

	defaultPreprocessProfile  = "preprocess"
	defaultInferenceProfile   = "inference"
	defaultPostprocessProfile = "postprocess"

	defaultPreprocessDecider = AlwaysDisaggPreprocessDeciderPluginType
	defaultInferenceDecider  = AlwaysDisaggInferenceDeciderPluginType
)

// ── Factory & constructor ────────────────────────────────────────────────────

type ioDisaggProfilesParameters struct {
	Preprocess  string `json:"preprocess,omitempty"`
	Inference   string `json:"inference,omitempty"`
	Postprocess string `json:"postprocess,omitempty"`
}

type ioDisaggDecidersParameters struct {
	Preprocess string `json:"preprocess,omitempty" pluginRef:""`
	Inference  string `json:"inference,omitempty" pluginRef:""`
}

// IODisaggProfileHandlerParameters is the current parameter format using nested maps.
type IODisaggProfileHandlerParameters struct {
	Profiles ioDisaggProfilesParameters `json:"profiles"`
	Deciders ioDisaggDecidersParameters `json:"deciders"`
}

// HandlerFactory is the unified factory for all disaggregation profile handlers.
//
//	if parameters.deciders.preprocess is set - Preprocess disaggregation will be supported
//	if parameters.deciders.inference is set - Inference disaggregation will be supported
func HandlerFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}

	tmpParameters, err := IODisaggProfileHandlerConfigParser(rawParameters, handle)
	if err != nil {
		return nil, err
	}
	parameters := tmpParameters.(IODisaggProfileHandlerParameters)

	// Resolve preprocess decider.
	var preprocessDecider ioDeciderPlugin
	if parameters.Deciders.Preprocess != "" {
		pre := handle.Plugin(parameters.Deciders.Preprocess)
		if pre == nil {
			return nil, fmt.Errorf("deciders.preprocess plugin not found: %s", parameters.Deciders.Preprocess)
		}
		var ok bool
		preprocessDecider, ok = pre.(ioDeciderPlugin)
		if !ok {
			return nil, fmt.Errorf("plugin %s does not implement preprocessDeciderPlugin", parameters.Deciders.Preprocess)
		}
	} else {
		return nil, fmt.Errorf("No deciders.preprocess configured. Need to configure one.")
	}
	// Resolve inference decider.
	var inferenceDecider ioDeciderPlugin
	if parameters.Deciders.Inference != "" {
		inf := handle.Plugin(parameters.Deciders.Inference)
		if inf == nil {
			return nil, fmt.Errorf("deciders.inference plugin not found: %s", parameters.Deciders.Inference)
		}
		var ok bool
		inferenceDecider, ok = inf.(ioDeciderPlugin)
		if !ok {
			return nil, fmt.Errorf("plugin %s does not implement inferenceDeciderPlugin", parameters.Deciders.Inference)
		}
	} else {
		return nil, fmt.Errorf("No deciders.inference configured. Need to configure one.")
	}
	// Create handler
	handler := NewIODisaggProfileHandler(
		parameters.Profiles.Preprocess, parameters.Profiles.Inference, parameters.Profiles.Postprocess,
		preprocessDecider, inferenceDecider,
	)
	return handler.WithName(name), nil
}

func IODisaggProfileHandlerConfigParser(rawParameters *json.Decoder, handle plugin.Handle) (any, error) {
	parameters := IODisaggProfileHandlerParameters{}
	if rawParameters != nil {
		var raw json.RawMessage
		if err := rawParameters.Decode(&raw); err != nil {
			return nil, fmt.Errorf("failed to parse parameters of the io-disagg-profile-handler - %w", err)
		}

		err := plugin.StrictDecoder(raw).Decode(&parameters)
		if err != nil {
			return nil, fmt.Errorf("failed to parse parameters of the io-disagg-profile-handler: "+
				"schema error: %w", err)
		}
	}

	// Apply profile name defaults for any fields still unset.
	if parameters.Profiles.Preprocess == "" {
		parameters.Profiles.Preprocess = defaultPreprocessProfile
	}
	if parameters.Profiles.Inference == "" {
		parameters.Profiles.Inference = defaultInferenceProfile
	}
	if parameters.Profiles.Postprocess == "" {
		parameters.Profiles.Postprocess = defaultPostprocessProfile
	}
	if parameters.Deciders.Preprocess == "" {
		parameters.Deciders.Preprocess = defaultPreprocessDecider
	}
	if parameters.Deciders.Inference == "" {
		parameters.Deciders.Inference = defaultInferenceDecider
	}

	return parameters, nil
}

// NewIODisaggProfileHandler creates a IOHandler directly.
// Active stages are determined by non-empty deciders.
func NewIODisaggProfileHandler(preprocessProfile, inferenceProfile, postprocessProfile string, preprocessDecider, inferenceDecider ioDeciderPlugin) *IOHandler {
	return newIODisaggProfileHandler(
		IODisaggProfileHandlerType,
		preprocessProfile, inferenceProfile, postprocessProfile,
		preprocessDecider, inferenceDecider,
	)
}

// ── Shared implementation ───────────────────────────────────────────────────

// compile-time assertions
var (
	_ scheduling.ProfileHandler = &IOHandler{}
	_ requestcontrol.PreRequest = &IOHandler{}
)

// IOHandler is the unified io disaggregation profile handler.
// It drives one or more of the following stages:
//
//   - Preprocess    (Pre): schedules a preprocess pod for IO data preprocessing
//   - Inference     (Inf): schedules inference pods for preprocessed data inference
//   - Postprocess  (Post): schedules a postprocess pod inference data postprocessing
type IOHandler struct {
	typedName          plugin.TypedName
	preprocessProfile  string
	inferenceProfile   string
	postprocessProfile string
	preprocessDecider  ioDeciderPlugin
	inferenceDecider   ioDeciderPlugin
}

// TypedName returns the typed name of the plugin.
func (h *IOHandler) TypedName() plugin.TypedName { return h.typedName }

// WithName sets the instance name of the plugin.
func (h *IOHandler) WithName(name string) *IOHandler {
	h.typedName.Name = name
	return h
}

// Consumes defines data types consumed by this plugin (through the PD decider).
// func (*IOHandler) Consumes() plugin.DataDependencies {
// 	return plugin.DataDependencies{
// 		Required: map[plugin.DataKey]any{
// 			attrprefix.PrefixCacheMatchInfoDataKey: attrprefix.PrefixCacheMatchInfo{},
// 			tokenproducer.TokenizedPromptDataKey:   scheduling.TokenizedPrompt{},
// 		},
// 	}
// }

func newIODisaggProfileHandler(handlerType, preprocessProfile, inferenceProfile, postprocessProfile string, preprocessDecider, inferenceDecider ioDeciderPlugin) *IOHandler {
	return &IOHandler{
		typedName:          plugin.TypedName{Type: handlerType},
		preprocessProfile:  preprocessProfile,
		inferenceProfile:   inferenceProfile,
		postprocessProfile: postprocessProfile,
		preprocessDecider:  preprocessDecider,
		inferenceDecider:   inferenceDecider,
	}
}

// Pick implements scheduling.ProfileHandler.
// Stages run in order: postprocess → preprocess → inference.
// Returns the next profile to execute, or an empty map when all stages are done.
func (h *IOHandler) Pick(ctx context.Context, request *scheduling.InferenceRequest, profiles map[string]scheduling.SchedulerProfile,
	profileResults map[string]*scheduling.ProfileRunResult) map[string]scheduling.SchedulerProfile {
	tracer := tracing.Tracer(schedplugins.TracerScope)
	ctx, span := tracer.Start(ctx, "pick_io_disagg_profile",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	if request == nil {
		span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "complete_nil_request"))
		return map[string]scheduling.SchedulerProfile{}
	}

	if request.TargetModel != "" {
		span.SetAttributes(attribute.String("gen_ai.request.model", request.TargetModel))
	}
	span.SetAttributes(attribute.String("gen_ai.request.id", request.RequestID))

	// ── Stage 1: Postprocess ────────────────────────────────────────────────────
	if _, executed := profileResults[h.postprocessProfile]; !executed {
		postprocessProfile, ok := profiles[h.postprocessProfile]
		if !ok {
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "error_missing_postprocess_profile"))
			return map[string]scheduling.SchedulerProfile{}
		}
		span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "run_postprocess"))
		return map[string]scheduling.SchedulerProfile{h.postprocessProfile: postprocessProfile}
	}

	postprocessRes := profileResults[h.postprocessProfile]
	if postprocessRes == nil || len(postprocessRes.TargetEndpoints) == 0 {
		span.SetAttributes(
			attribute.String("llm_d.epp.profile_handler.decision", "complete"),
			attribute.Bool("llm_d.epp.profile_handler.postprocess_failed", true),
		)
		return map[string]scheduling.SchedulerProfile{}
	}

	// ── Stage 2: Preprocess ─────────────────────────────────────────
	if _, executed := profileResults[h.preprocessProfile]; !executed {
		preprocessProfile, ok := profiles[h.preprocessProfile]
		if !ok {
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "error_missing_preprocess_profile"))
			return map[string]scheduling.SchedulerProfile{}
		}
		if h.preprocessDecider == nil {
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "error_missing_preprocess_decider"))
			return map[string]scheduling.SchedulerProfile{}
		}
		if h.preprocessDecider.disaggregate(ctx, request, postprocessRes.TargetEndpoints[0]) {
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "run_preprocess"))
			return map[string]scheduling.SchedulerProfile{h.preprocessProfile: preprocessProfile}
		}
		// TODO(konpap): Need to adapt this if disaggregated preprocessing is optional
		span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "error_preprocess_disaggregation_impossible"))
		return map[string]scheduling.SchedulerProfile{}
	}

	preprocessRes := profileResults[h.preprocessProfile]
	if preprocessRes == nil || len(preprocessRes.TargetEndpoints) == 0 {
		span.SetAttributes(
			attribute.String("llm_d.epp.profile_handler.decision", "complete"),
			attribute.Bool("llm_d.epp.profile_handler.preprocess_failed", true),
		)
		return map[string]scheduling.SchedulerProfile{}
	}

	// ── Stage 3: Inference ─────────────────────────────────────────
	if _, executed := profileResults[h.inferenceProfile]; !executed {
		inferenceProfile, ok := profiles[h.inferenceProfile]
		if !ok {
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "error_missing_inference_profile"))
			return map[string]scheduling.SchedulerProfile{}
		}
		if h.inferenceDecider == nil {
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "error_missing_inference_decider"))
			return map[string]scheduling.SchedulerProfile{}
		}
		if h.inferenceDecider.disaggregate(ctx, request, postprocessRes.TargetEndpoints[0], preprocessRes.TargetEndpoints[0]) {
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "run_inference"))
			return map[string]scheduling.SchedulerProfile{h.inferenceProfile: inferenceProfile}
		}
		// TODO(konpap): Need to adapt this if disaggregated inferecne is optional
		span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "error_inference_disaggregation_impossible"))
		return map[string]scheduling.SchedulerProfile{}
	}

	// ── All stages done: record routing decision ───────────────────────────
	preprocessUsed := profileResults[h.preprocessProfile] != nil
	inferenceUsed := profileResults[h.inferenceProfile] != nil
	postprocessUsed := profileResults[h.postprocessProfile] != nil

	decision := IODisaggDecisionType(preprocessUsed, inferenceUsed, postprocessUsed)
	RecordIODisaggDecision(h.typedName.Name, h.typedName.Type, request.TargetModel, decision)
	span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "complete_"+decision))

	return map[string]scheduling.SchedulerProfile{}
}

// ProcessResults implements scheduling.ProfileHandler.
// Builds the final SchedulingResult from whichever stages ran successfully.
func (h *IOHandler) ProcessResults(
	_ context.Context,
	request *scheduling.InferenceRequest,
	profileResults map[string]*scheduling.ProfileRunResult,
) (*scheduling.SchedulingResult, error) {
	if request == nil {
		return nil, errors.New("request is nil")
	}

	postprocessRunResults := profileResults[h.postprocessProfile]
	if postprocessRunResults == nil || len(postprocessRunResults.TargetEndpoints) == 0 {
		return nil, errors.New("failed to find available postprocess workers")
	}

	preprocessRunResults := profileResults[h.preprocessProfile]
	if preprocessRunResults == nil || len(preprocessRunResults.TargetEndpoints) == 0 {
		return nil, errors.New("failed to find available preprocess workers")
	}

	inferenceRunResults := profileResults[h.inferenceProfile]
	if inferenceRunResults == nil || len(inferenceRunResults.TargetEndpoints) == 0 {
		return nil, errors.New("failed to find available inference workers")
	}

	updatedResults := map[string]*scheduling.ProfileRunResult{}

	updatedResults[h.preprocessProfile] = preprocessRunResults
	updatedResults[h.inferenceProfile] = inferenceRunResults
	updatedResults[h.postprocessProfile] = postprocessRunResults

	return &scheduling.SchedulingResult{
		PrimaryProfileName: h.postprocessProfile,
		ProfileResults:     updatedResults,
	}, nil
}

// ── PreRequest ──────────────────────────────────────────────────────────────

// PreRequest wires preprocess and inference SchedulerProfile results into headers
// so the sidecar knows which pods to contact for disaggregated work.
func (h *IOHandler) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	tracer := tracing.Tracer(schedplugins.TracerScope)
	_, span := tracer.Start(ctx, "prepare_io_disaggregation",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	if request == nil {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.preprocess.disaggregation_used", false),
			attribute.Bool("llm_d.epp.inference.disaggregation_used", false),
			attribute.Bool("llm_d.epp.postprocess.disaggregation_used", false),
			attribute.String("llm_d.epp.io_disagg.reason", "request_is_nil"),
		)
		return
	}
	if schedulingResult == nil {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.preprocess.disaggregation_used", false),
			attribute.Bool("llm_d.epp.inference.disaggregation_used", false),
			attribute.Bool("llm_d.epp.postprocess.disaggregation_used", false),
			attribute.String("llm_d.epp.io_disagg.reason", "scheduling_result_is_nil"),
		)
		return
	}

	if request.TargetModel != "" {
		span.SetAttributes(attribute.String("gen_ai.request.model", request.TargetModel))
	}
	span.SetAttributes(attribute.String("gen_ai.request.id", request.RequestID))

	// Preprocess header
	delete(request.Headers, routing.PreprocessEndpointHeader)
	preprocessProfileRunResult := schedulingResult.ProfileResults[h.preprocessProfile]
	switch {
	case len(preprocessProfileRunResult.TargetEndpoints) == 0:
		span.SetAttributes(
			attribute.Bool("llm_d.epp.preprocess.disaggregation_used", false),
			attribute.String("llm_d.epp.preprocess.reason", "no_preprocess_profile_target_endpoints"),
		)
		return
	default:
		targetPod := preprocessProfileRunResult.TargetEndpoints[0].GetMetadata()
		preprocessHostPort := net.JoinHostPort(targetPod.Address, targetPod.Port)
		request.Headers[routing.PreprocessEndpointHeader] = preprocessHostPort
		span.SetAttributes(
			attribute.Bool("llm_d.epp.preprocess.disaggregation_used", true),
			attribute.String("llm_d.epp.preprocess.pod_address", targetPod.Address),
			attribute.String("llm_d.epp.preprocess.pod_port", targetPod.Port),
		)
	}

	// Inference header
	delete(request.Headers, routing.InfereceEndpointsHeader)
	inferenceProfileRunResult := schedulingResult.ProfileResults[h.inferenceProfile]
	if inferenceProfileRunResult == nil {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.inference.disaggregation_used", false),
			attribute.String("llm_d.epp.inference.reason", "no_inference_profile_result"),
		)
		return
	}

	var inferenceHostPorts []string
	for _, endpoint := range inferenceProfileRunResult.TargetEndpoints {
		targetEndpoint := endpoint.GetMetadata()
		inferenceHostPort := net.JoinHostPort(targetEndpoint.Address, targetEndpoint.Port)
		inferenceHostPorts = append(inferenceHostPorts, inferenceHostPort)
	}
	if len(inferenceHostPorts) == 0 {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.inference.disaggregation_used", false),
			attribute.String("llm_d.epp.inference.reason", "no_inference_profile_target_endpoints"),
		)
		return
	}

	request.Headers[routing.InfereceEndpointsHeader] = strings.Join(inferenceHostPorts, ",")
	span.SetAttributes(
		attribute.Bool("llm_d.epp.inference.disaggregation_used", true),
		attribute.String("llm_d.epp.inference.endpoints", strings.Join(inferenceHostPorts, ",")),
	)

}
