// Package disagg provides profile handler plugins for the epp.
package disagg

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/observability/multimodal"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	schedplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling"
)

// ── Constants ───────────────────────────────────────────────────────────────

const (
	// DisaggProfileHandlerType is the canonical type for the unified disaggregation profile handler.
	DisaggProfileHandlerType = "disagg-profile-handler"

	defaultDecodeProfile  = "decode"
	defaultPrefillProfile = "prefill"
	defaultEncodeProfile  = "encode"
)

// StageOrder defines the execution order of stages in the disaggregation profile handler.
type StageOrder string

const (
	// StageOrderDecodeFirst runs Decode -> Encode -> Prefill (default).
	// Decode runs first; PD decider inspects the picked decode endpoint to determine if prefill is needed.
	StageOrderDecodeFirst StageOrder = "decode-first"

	// StageOrderPrefillFirst runs Prefill -> Encode -> Decode.
	// Prefill runs first; decode endpoint can use topology affinity against the picked prefill endpoint.
	StageOrderPrefillFirst StageOrder = "prefill-first"
)

// ParseStageOrder parses a stage order string into a StageOrder.
func ParseStageOrder(s string) (StageOrder, error) {
	switch strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(s, "-", ""), "_", "")) {
	case "", "decodefirst", "decode":
		return StageOrderDecodeFirst, nil
	case "prefillfirst", "prefill":
		return StageOrderPrefillFirst, nil
	default:
		return "", fmt.Errorf("invalid stage order %q, must be %q or %q", s, StageOrderDecodeFirst, StageOrderPrefillFirst)
	}
}

// PeerEndpointAttributeKey is the request-attribute key under which this
// handler publishes the endpoint selected in an earlier scheduling phase
// (the decode pick in decode-first mode, or the prefill pick in prefill-first mode),
// for plugins in a later profile to compare against (e.g. topology affinity).
// The value is an Endpoint.
var PeerEndpointAttributeKey = plugin.NewDataKey("peer-endpoint", DisaggProfileHandlerType)

// prefillDeclinedAttributeKey marks, for decode-first mode, that the PD
// decider itself chose not to run the prefill profile (or no PD decider is
// configured). ProcessResults reads this to tell that intentional skip apart
// from a prefill profile that was picked to run and then failed to find any
// endpoint: both leave profileResults[prefillProfile] as a nil entry, but
// only the former is safe to complete decode-only. The value is a bool.
var prefillDeclinedAttributeKey = plugin.NewDataKey("prefill-declined", DisaggProfileHandlerType)

// ── Factory & constructor ────────────────────────────────────────────────────

type disaggProfilesParameters struct {
	Decode  string `json:"decode,omitempty"`
	Prefill string `json:"prefill,omitempty"`
	Encode  string `json:"encode,omitempty"`
}

type disaggDecidersParameters struct {
	Prefill string `json:"prefill,omitempty" pluginRef:""`
	Encode  string `json:"encode,omitempty" pluginRef:""`
}

// DisaggProfileHandlerParameters is the current parameter format using nested maps.
type DisaggProfileHandlerParameters struct {
	StageOrder StageOrder               `json:"stageOrder,omitempty"`
	Profiles   disaggProfilesParameters `json:"profiles"`
	Deciders   disaggDecidersParameters `json:"deciders"`
}

// HandlerFactory is the unified factory for all disaggregation profile handlers.
//
//	if parameters.deciders.prefill is set - P disaggregation will be supported
//	if parameters.deciders.encode is set - E disaggregation will be supported
func HandlerFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}
	logger := log.FromContext(handle.Context())

	tmpParameters, err := DisaggProfileHandlerConfigParser(rawParameters, handle)
	if err != nil {
		return nil, err
	}
	parameters := tmpParameters.(DisaggProfileHandlerParameters)

	// Resolve PD decider (optional).
	var pdDecider deciderPlugin
	if parameters.Deciders.Prefill != "" {
		p := handle.Plugin(parameters.Deciders.Prefill)
		if p == nil {
			return nil, fmt.Errorf("deciders.prefill plugin not found: %s", parameters.Deciders.Prefill)
		}
		var ok bool
		pdDecider, ok = p.(deciderPlugin)
		if !ok {
			return nil, fmt.Errorf("plugin %s does not implement prefillDeciderPlugin", parameters.Deciders.Prefill)
		}
	} else {
		logger.Info("No deciders.prefill configured, P/D disaggregation disabled")
	}
	// Resolve encode decider (optional).
	var encodeDecider deciderPlugin
	if parameters.Deciders.Encode != "" {
		ep := handle.Plugin(parameters.Deciders.Encode)
		if ep == nil {
			return nil, fmt.Errorf("deciders.encode plugin not found: %s", parameters.Deciders.Encode)
		}
		var ok bool
		encodeDecider, ok = ep.(deciderPlugin)
		if !ok {
			return nil, fmt.Errorf("plugin %s does not implement encodeDeciderPlugin", parameters.Deciders.Encode)
		}
	} else {
		logger.Info("No deciders.encode configured, E disaggregation disabled")
	}
	// Create handler
	handler := NewDisaggProfileHandler(
		parameters.Profiles.Decode, parameters.Profiles.Prefill, parameters.Profiles.Encode,
		pdDecider, encodeDecider,
	).WithStageOrder(parameters.StageOrder)
	return handler.WithName(name), nil
}

func DisaggProfileHandlerConfigParser(rawParameters *json.Decoder, _ plugin.Handle) (any, error) {
	parameters := DisaggProfileHandlerParameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse parameters of the disagg-profile-handler - %w", err)
		}
	}

	// Apply stage order defaults and validation.
	parsedOrder, err := ParseStageOrder(string(parameters.StageOrder))
	if err != nil {
		return nil, err
	}
	parameters.StageOrder = parsedOrder

	if parameters.StageOrder == StageOrderPrefillFirst && parameters.Deciders.Prefill != "" {
		return nil, fmt.Errorf("prefill decider is not supported in %s stage order", StageOrderPrefillFirst)
	}

	// Apply profile name defaults for any fields still unset.
	if parameters.Profiles.Decode == "" {
		parameters.Profiles.Decode = defaultDecodeProfile
	}
	if parameters.Profiles.Prefill == "" {
		parameters.Profiles.Prefill = defaultPrefillProfile
	}
	if parameters.Profiles.Encode == "" {
		parameters.Profiles.Encode = defaultEncodeProfile
	}

	return parameters, nil
}

// NewDisaggProfileHandler creates a Handler directly.
// Active stages are determined by non-empty deciders.
func NewDisaggProfileHandler(decodeProfile, prefillProfile, encodeProfile string, pdDecider, encodeDecider deciderPlugin) *Handler {
	return newDisaggProfileHandler(
		DisaggProfileHandlerType,
		decodeProfile, prefillProfile, encodeProfile,
		pdDecider, encodeDecider,
	)
}

// ── Shared implementation ───────────────────────────────────────────────────

// compile-time assertions
var (
	_ scheduling.ProfileHandler = &Handler{}
	_ requestcontrol.PreRequest = &Handler{}
)

// Handler is the unified disaggregation profile handler.
// It drives one or more of the following stages, each optional except decode:
//
//   - Encode  (E): schedules encoder pods for multimodal content
//   - Prefill (P): schedules a prefill pod for KV-cache disaggregation
//   - Decode  (D): schedules the decode pod
//
// In decode-first mode (default), stages run: decode → encode → prefill.
// In prefill-first mode, stages run: prefill → encode → decode.
//
// All four handler types (D, P/D, E/PD, E/P/D) share this single implementation;
// active stages are selected by setting encodeProfile / prefillProfile.
type Handler struct {
	typedName      plugin.TypedName
	stageOrder     StageOrder
	decodeProfile  string
	prefillProfile string
	encodeProfile  string
	pdDecider      deciderPlugin
	encodeDecider  deciderPlugin
}

// TypedName returns the typed name of the plugin.
func (h *Handler) TypedName() plugin.TypedName { return h.typedName }

// WithName sets the instance name of the plugin.
func (h *Handler) WithName(name string) *Handler {
	h.typedName.Name = name
	return h
}

// WithStageOrder sets the stage execution order for the handler.
func (h *Handler) WithStageOrder(stageOrder StageOrder) *Handler {
	h.stageOrder = stageOrder
	return h
}

// Consumes defines data types consumed by this plugin (through the PD decider).
func (*Handler) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{
			attrprefix.PrefixCacheMatchInfoDataKey: attrprefix.PrefixCacheMatchInfo{},
			tokenproducer.TokenizedPromptDataKey:   scheduling.TokenizedRequest{},
		},
	}
}

func newDisaggProfileHandler(handlerType, decodeProfile, prefillProfile, encodeProfile string, pdDecider, encodeDecider deciderPlugin) *Handler {
	return &Handler{
		typedName:      plugin.TypedName{Type: handlerType},
		stageOrder:     StageOrderDecodeFirst,
		decodeProfile:  decodeProfile,
		prefillProfile: prefillProfile,
		encodeProfile:  encodeProfile,
		pdDecider:      pdDecider,
		encodeDecider:  encodeDecider,
	}
}

// Pick implements scheduling.ProfileHandler.
// In decode-first mode (default), stages run: decode → encode (optional) → prefill (optional).
// In prefill-first mode, stages run: prefill (optional) → encode (optional) → decode.
// Returns the next profile to execute, or an empty map when all stages are done.
func (h *Handler) Pick(ctx context.Context, request *scheduling.InferenceRequest, profiles map[string]scheduling.SchedulerProfile,
	profileResults map[string]*scheduling.ProfileRunResult) map[string]scheduling.SchedulerProfile {
	tracer := tracing.Tracer(schedplugins.TracerScope)
	ctx, span := tracer.Start(ctx, "pick_disagg_profile",
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
	span.SetAttributes(mmobs.SpanAttributes(request)...)

	if h.stageOrder == StageOrderPrefillFirst {
		return h.pickPrefillFirst(ctx, span, request, profiles, profileResults)
	}
	return h.pickDecodeFirst(ctx, span, request, profiles, profileResults)
}

func (h *Handler) pickDecodeFirst(ctx context.Context, span trace.Span, request *scheduling.InferenceRequest,
	profiles map[string]scheduling.SchedulerProfile, profileResults map[string]*scheduling.ProfileRunResult,
) map[string]scheduling.SchedulerProfile {
	// ── Stage 1: Decode ────────────────────────────────────────────────────
	if _, executed := profileResults[h.decodeProfile]; !executed {
		decodeProfile, ok := profiles[h.decodeProfile]
		if !ok {
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "error_missing_decode_profile"))
			return map[string]scheduling.SchedulerProfile{}
		}
		span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "run_decode"))
		return map[string]scheduling.SchedulerProfile{h.decodeProfile: decodeProfile}
	}

	decodeRes := profileResults[h.decodeProfile]
	if decodeRes == nil || len(decodeRes.TargetEndpoints) == 0 {
		span.SetAttributes(
			attribute.String("llm_d.epp.profile_handler.decision", "complete"),
			attribute.Bool("llm_d.epp.profile_handler.decode_failed", true),
		)
		return map[string]scheduling.SchedulerProfile{}
	}

	// ── Stage 2: Encode (optional) ─────────────────────────────────────────
	if _, hasEncodeProfile := profiles[h.encodeProfile]; hasEncodeProfile {
		if _, executed := profileResults[h.encodeProfile]; !executed {
			if h.encodeDecider != nil && h.encodeDecider.disaggregate(ctx, request, decodeRes.TargetEndpoints[0]) {
				span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "run_encode"))
				return map[string]scheduling.SchedulerProfile{h.encodeProfile: profiles[h.encodeProfile]}
			}
			// Decider rejected encode - mark as evaluated so we don't re-run the decider.
			profileResults[h.encodeProfile] = nil
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "skip_encode"))
		}
	}

	// ── Stage 3: Prefill (optional) ────────────────────────────────────────
	if _, hasPrefillProfile := profiles[h.prefillProfile]; hasPrefillProfile {
		if _, executed := profileResults[h.prefillProfile]; !executed {
			if h.pdDecider != nil && h.pdDecider.disaggregate(ctx, request, decodeRes.TargetEndpoints[0]) {
				// Publish the decode pick so plugins in the prefill profile (e.g.
				// topology affinity) can compare candidates against it.
				request.PutAttribute(PeerEndpointAttributeKey, decodeRes.TargetEndpoints[0])
				span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "run_prefill"))
				return map[string]scheduling.SchedulerProfile{h.prefillProfile: profiles[h.prefillProfile]}
			}
			// Decider rejected prefill - mark as evaluated so we don't re-run the decider,
			// and record that this is an intentional skip, not a failed run.
			profileResults[h.prefillProfile] = nil
			request.PutAttribute(prefillDeclinedAttributeKey, true)
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "skip_prefill"))
		}
	}

	// ── All stages done: record routing decision ───────────────────────────
	encodeUsed := profileResults[h.encodeProfile] != nil
	prefillUsed := profileResults[h.prefillProfile] != nil

	decision := DisaggDecisionType(encodeUsed, prefillUsed)
	RecordDisaggDecision(h.typedName.Name, h.typedName.Type, request.TargetModel, decision)
	span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "complete_"+decision))

	return map[string]scheduling.SchedulerProfile{}
}

func (h *Handler) pickPrefillFirst(ctx context.Context, span trace.Span, request *scheduling.InferenceRequest,
	profiles map[string]scheduling.SchedulerProfile, profileResults map[string]*scheduling.ProfileRunResult,
) map[string]scheduling.SchedulerProfile {
	// If decode has already run, we are done.
	if _, decodeExecuted := profileResults[h.decodeProfile]; decodeExecuted {
		decodeRes := profileResults[h.decodeProfile]
		if decodeRes == nil || len(decodeRes.TargetEndpoints) == 0 {
			span.SetAttributes(
				attribute.String("llm_d.epp.profile_handler.decision", "complete"),
				attribute.Bool("llm_d.epp.profile_handler.decode_failed", true),
			)
			return map[string]scheduling.SchedulerProfile{}
		}

		encodeUsed := profileResults[h.encodeProfile] != nil
		prefillUsed := profileResults[h.prefillProfile] != nil

		decision := DisaggDecisionType(encodeUsed, prefillUsed)
		RecordDisaggDecision(h.typedName.Name, h.typedName.Type, request.TargetModel, decision)
		span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "complete_"+decision))
		return map[string]scheduling.SchedulerProfile{}
	}

	// ── Stage 1: Prefill (optional) ────────────────────────────────────────
	// In prefill-first mode, prefill runs whenever the prefill profile is configured.
	if _, hasPrefillProfile := profiles[h.prefillProfile]; hasPrefillProfile {
		if _, executed := profileResults[h.prefillProfile]; !executed {
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "run_prefill"))
			return map[string]scheduling.SchedulerProfile{h.prefillProfile: profiles[h.prefillProfile]}
		}
	}

	// ── Stage 2: Encode (optional) ─────────────────────────────────────────
	if _, hasEncodeProfile := profiles[h.encodeProfile]; hasEncodeProfile {
		if _, executed := profileResults[h.encodeProfile]; !executed {
			if h.encodeDecider != nil && h.encodeDecider.disaggregate(ctx, request, nil) {
				span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "run_encode"))
				return map[string]scheduling.SchedulerProfile{h.encodeProfile: profiles[h.encodeProfile]}
			}
			// Decider rejected encode - mark as evaluated so we don't re-run.
			profileResults[h.encodeProfile] = nil
			span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "skip_encode"))
		}
	}

	// ── Stage 3: Decode (mandatory) ────────────────────────────────────────
	decodeProfile, ok := profiles[h.decodeProfile]
	if !ok {
		span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "error_missing_decode_profile"))
		return map[string]scheduling.SchedulerProfile{}
	}

	// Publish the prefill pick (if prefill ran and succeeded) so plugins in the
	// decode profile (e.g. topology affinity) can compare candidates against it.
	if prefillRes := profileResults[h.prefillProfile]; prefillRes != nil && len(prefillRes.TargetEndpoints) > 0 {
		request.PutAttribute(PeerEndpointAttributeKey, prefillRes.TargetEndpoints[0])
	}

	span.SetAttributes(attribute.String("llm_d.epp.profile_handler.decision", "run_decode"))
	return map[string]scheduling.SchedulerProfile{h.decodeProfile: decodeProfile}
}

// ProcessResults implements scheduling.ProfileHandler.
// Builds the final SchedulingResult from whichever stages ran successfully.
func (h *Handler) ProcessResults(
	_ context.Context,
	request *scheduling.InferenceRequest,
	profileResults map[string]*scheduling.ProfileRunResult,
) (*scheduling.SchedulingResult, error) {
	if request == nil {
		return nil, errors.New("request is nil")
	}

	decodeRunResults := profileResults[h.decodeProfile]
	if decodeRunResults == nil || len(decodeRunResults.TargetEndpoints) == 0 {
		return nil, errors.New("failed to find available decode workers")
	}

	updatedResults := map[string]*scheduling.ProfileRunResult{}

	updatedResults[h.decodeProfile] = decodeRunResults

	if prefillRes, ok := profileResults[h.prefillProfile]; ok {
		if prefillRes != nil {
			updatedResults[h.prefillProfile] = prefillRes
		} else if declined, _ := scheduling.ReadRequestAttribute[bool](request, prefillDeclinedAttributeKey); !declined {
			// The PD decider picked the prefill profile to run and it found no
			// endpoint, instead of the decider declining to run it at all.
			// Completing decode-only here would silently run prefill work on a
			// decode pod instead of failing the request.
			return nil, fmt.Errorf("prefill profile %q was required but produced no result", h.prefillProfile)
		}
	}

	if encodeRes, ok := profileResults[h.encodeProfile]; ok && encodeRes != nil {
		updatedResults[h.encodeProfile] = encodeRes
	}

	return &scheduling.SchedulingResult{
		PrimaryProfileName: h.decodeProfile,
		ProfileResults:     updatedResults,
	}, nil
}

// ── PreRequest ──────────────────────────────────────────────────────────────

// PreRequest wires prefill and encode SchedulerProfile results into headers
// so the sidecar knows which pods to contact for disaggregated work.
func (h *Handler) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) error {
	tracer := tracing.Tracer(schedplugins.TracerScope)
	_, span := tracer.Start(ctx, "prepare_disaggregation",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	if request == nil {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.pd.disaggregation_used", false),
			attribute.Bool("llm_d.epp.encode.disaggregation_used", false),
			attribute.String("llm_d.epp.disagg.reason", "request_is_nil"),
		)
		return nil
	}
	if schedulingResult == nil {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.pd.disaggregation_used", false),
			attribute.Bool("llm_d.epp.encode.disaggregation_used", false),
			attribute.String("llm_d.epp.disagg.reason", "scheduling_result_is_nil"),
		)
		return nil
	}

	if request.TargetModel != "" {
		span.SetAttributes(attribute.String("gen_ai.request.model", request.TargetModel))
	}
	span.SetAttributes(attribute.String("gen_ai.request.id", request.RequestID))
	span.SetAttributes(mmobs.SpanAttributes(request)...)

	// Prefill header
	delete(request.Headers, routing.PrefillEndpointHeader)
	prefillProfileRunResult := schedulingResult.ProfileResults[h.prefillProfile]
	switch {
	case prefillProfileRunResult == nil:
		span.SetAttributes(
			attribute.Bool("llm_d.epp.pd.disaggregation_used", false),
			attribute.String("llm_d.epp.pd.reason", "no_prefill_profile_result"),
		)
	case len(prefillProfileRunResult.TargetEndpoints) == 0:
		span.SetAttributes(
			attribute.Bool("llm_d.epp.pd.disaggregation_used", false),
			attribute.String("llm_d.epp.pd.reason", "no_prefill_profile_target_endpoints"),
		)
	default:
		targetPod := prefillProfileRunResult.TargetEndpoints[0].GetMetadata()
		prefillHostPort := net.JoinHostPort(targetPod.Address, targetPod.Port)
		request.Headers[routing.PrefillEndpointHeader] = prefillHostPort
		span.SetAttributes(
			attribute.Bool("llm_d.epp.pd.disaggregation_used", true),
			attribute.String("llm_d.epp.pd.prefill_pod_address", targetPod.Address),
			attribute.String("llm_d.epp.pd.prefill_pod_port", targetPod.Port),
		)
	}

	// Encode header
	delete(request.Headers, routing.EncoderEndpointsHeader)
	encodeProfileRunResult := schedulingResult.ProfileResults[h.encodeProfile]
	if encodeProfileRunResult == nil {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.encode.disaggregation_used", false),
			attribute.String("llm_d.epp.encode.reason", "no_encode_profile_result"),
		)
		return nil
	}

	var encodeHostPorts []string
	for _, endpoint := range encodeProfileRunResult.TargetEndpoints {
		targetEndpoint := endpoint.GetMetadata()
		encodeHostPort := net.JoinHostPort(targetEndpoint.Address, targetEndpoint.Port)
		encodeHostPorts = append(encodeHostPorts, encodeHostPort)
	}
	if len(encodeHostPorts) == 0 {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.encode.disaggregation_used", false),
			attribute.String("llm_d.epp.encode.reason", "no_encode_profile_target_endpoints"),
		)
		return nil
	}

	request.Headers[routing.EncoderEndpointsHeader] = strings.Join(encodeHostPorts, ",")
	span.SetAttributes(
		attribute.Bool("llm_d.epp.encode.disaggregation_used", true),
		attribute.String("llm_d.epp.encode.endpoints", strings.Join(encodeHostPorts, ",")),
	)
	return nil
}
