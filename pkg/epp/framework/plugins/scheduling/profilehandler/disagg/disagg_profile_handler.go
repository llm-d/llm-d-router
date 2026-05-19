// Package disagg provides profile handler plugins for the epp.
package disagg

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strings"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/metrics"
	"github.com/llm-d/llm-d-router/pkg/telemetry"
)

// ── Constants ───────────────────────────────────────────────────────────────

const (
	// DisaggProfileHandlerType is the canonical type for the unified disaggregation profile handler.
	DisaggProfileHandlerType = "disagg-profile-handler"

	defaultDecodeProfile  = "decode"
	defaultPrefillProfile = "prefill"
	defaultEncodeProfile  = "encode"
)

// ── Factory & constructor ────────────────────────────────────────────────────

type disaggProfilesParameters struct {
	Decode  string `json:"decode,omitempty"`
	Prefill string `json:"prefill,omitempty"`
	Encode  string `json:"encode,omitempty"`
}

type disaggDecidersParameters struct {
	Prefill string `json:"prefill,omitempty"`
	Encode  string `json:"encode,omitempty"`
}

// disaggProfileHandlerParameters is the current parameter format using nested maps.
type disaggProfileHandlerParameters struct {
	Profiles disaggProfilesParameters `json:"profiles"`
	Deciders disaggDecidersParameters `json:"deciders"`
}

// legacyDisaggProfileHandlerParameters is the deprecated flat parameter format.
// Unknown fields (e.g. pd-profile-handler's prefixPluginType, primaryPort) are
// silently ignored by json.Unmarshal, so they need not be declared here.
type legacyDisaggProfileHandlerParameters struct {
	DecodeProfile            string `json:"decodeProfile"`
	PrefillProfile           string `json:"prefillProfile"`
	EncodeProfile            string `json:"encodeProfile"`
	PrefillDeciderPluginName string `json:"prefillDeciderPluginName"`
	EncodeDeciderPluginName  string `json:"encodeDeciderPluginName"`
	// DeciderPluginName is a legacy alias from pd-profile-handler, maps to deciders.prefill.
	DeciderPluginName string `json:"deciderPluginName"`
}

// toDisaggParams copies legacy flat fields into the nested format, logging a
// deprecation warning for each field in use.
func (l *legacyDisaggProfileHandlerParameters) toDisaggParams(logger logr.Logger) disaggProfileHandlerParameters {
	p := disaggProfileHandlerParameters{}
	if l.DecodeProfile != "" {
		logger.Info("Deprecated parameter 'decodeProfile', use 'profiles.decode' instead")
		p.Profiles.Decode = l.DecodeProfile
	}
	if l.PrefillProfile != "" {
		logger.Info("Deprecated parameter 'prefillProfile', use 'profiles.prefill' instead")
		p.Profiles.Prefill = l.PrefillProfile
	}
	if l.EncodeProfile != "" {
		logger.Info("Deprecated parameter 'encodeProfile', use 'profiles.encode' instead")
		p.Profiles.Encode = l.EncodeProfile
	}
	if l.PrefillDeciderPluginName != "" {
		logger.Info("Deprecated parameter 'prefillDeciderPluginName', use 'deciders.prefill' instead")
		p.Deciders.Prefill = l.PrefillDeciderPluginName
	}
	// DeciderPluginName is a lower-priority alias for prefill decider (from pd-profile-handler).
	if l.DeciderPluginName != "" && p.Deciders.Prefill == "" {
		logger.Info("Deprecated parameter 'deciderPluginName', use 'deciders.prefill' instead")
		p.Deciders.Prefill = l.DeciderPluginName
	}
	if l.EncodeDeciderPluginName != "" {
		logger.Info("Deprecated parameter 'encodeDeciderPluginName', use 'deciders.encode' instead")
		p.Deciders.Encode = l.EncodeDeciderPluginName
	}
	return p
}

// HandlerFactory is the unified factory for all disaggregation profile handlers.
//
//	if parameters.deciders.prefill is set - P disaggregation will be supported
//	if parameters.deciders.encode is set - E disaggregation will be supported
func HandlerFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	logger := log.FromContext(handle.Context())

	parameters := disaggProfileHandlerParameters{}
	if rawParameters != nil {
		legacy := legacyDisaggProfileHandlerParameters{}

		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse parameters of the disagg-profile-handler - %w", err)
		}
		if err := json.Unmarshal(rawParameters, &legacy); err != nil {
			return nil, fmt.Errorf("failed to parse parameters of the disagg-profile-handler - %w", err)
		}

		if parameters.Profiles != (disaggProfilesParameters{}) ||
			parameters.Deciders != (disaggDecidersParameters{}) {
			// Make sure the legacy parameters were not used
			if legacy != (legacyDisaggProfileHandlerParameters{}) {
				return nil, errors.New("cannot mix deprecated flat parameters (decodeProfile, prefillProfile, encodeProfile, " +
					"deciderPluginName, prefillDeciderPluginName, encodeDeciderPluginName) " +
					"with nested parameters (profiles, deciders): use one format or the other")
			}
		} else {
			logger.Info("Deprecated: using flat parameter format, migrate to nested profiles/deciders format")
			parameters = legacy.toDisaggParams(logger)
		}
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
	)
	return handler.WithName(name), nil
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
//   - Decode  (D): schedules the decode pod (always runs first)
//
// All four handler types (D, P/D, E/PD, E/P/D) share this single implementation;
// active stages are selected by setting encodeProfile / prefillProfile.
type Handler struct {
	typedName      plugin.TypedName
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

// Consumes defines data types consumed by this plugin (through the PD decider).
func (*Handler) Consumes() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

func newDisaggProfileHandler(handlerType, decodeProfile, prefillProfile, encodeProfile string, pdDecider, encodeDecider deciderPlugin) *Handler {
	return &Handler{
		typedName:      plugin.TypedName{Type: handlerType},
		decodeProfile:  decodeProfile,
		prefillProfile: prefillProfile,
		encodeProfile:  encodeProfile,
		pdDecider:      pdDecider,
		encodeDecider:  encodeDecider,
	}
}

// Pick implements scheduling.ProfileHandler.
// Stages run in order: decode → encode (optional) → prefill (optional).
// Returns the next profile to execute, or an empty map when all stages are done.
func (h *Handler) Pick(ctx context.Context, _ *scheduling.CycleState, request *scheduling.InferenceRequest, profiles map[string]scheduling.SchedulerProfile,
	profileResults map[string]*scheduling.ProfileRunResult) map[string]scheduling.SchedulerProfile {
	tracer := telemetry.Tracer()
	ctx, span := tracer.Start(ctx, "llm_d.epp.disagg.profile_handler.pick",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	if request == nil {
		span.SetAttributes(attribute.String("llm_d.profile_handler.decision", "complete_nil_request"))
		return map[string]scheduling.SchedulerProfile{}
	}

	if request.TargetModel != "" {
		span.SetAttributes(attribute.String("gen_ai.request.model", request.TargetModel))
	}
	span.SetAttributes(attribute.String("gen_ai.request.id", request.RequestID))

	// ── Stage 1: Decode ────────────────────────────────────────────────────
	if _, executed := profileResults[h.decodeProfile]; !executed {
		decodeProfile, ok := profiles[h.decodeProfile]
		if !ok {
			span.SetAttributes(attribute.String("llm_d.profile_handler.decision", "error_missing_decode_profile"))
			return map[string]scheduling.SchedulerProfile{}
		}
		span.SetAttributes(attribute.String("llm_d.profile_handler.decision", "run_decode"))
		return map[string]scheduling.SchedulerProfile{h.decodeProfile: decodeProfile}
	}

	decodeRes := profileResults[h.decodeProfile]
	if decodeRes == nil || len(decodeRes.TargetEndpoints) == 0 {
		span.SetAttributes(
			attribute.String("llm_d.profile_handler.decision", "complete"),
			attribute.Bool("llm_d.profile_handler.decode_failed", true),
		)
		return map[string]scheduling.SchedulerProfile{}
	}

	// ── Stage 2: Encode (optional) ─────────────────────────────────────────
	if _, hasEncodeProfile := profiles[h.encodeProfile]; hasEncodeProfile {
		if _, executed := profileResults[h.encodeProfile]; !executed {
			if h.encodeDecider != nil && h.encodeDecider.disaggregate(ctx, request, decodeRes.TargetEndpoints[0]) {
				span.SetAttributes(attribute.String("llm_d.profile_handler.decision", "run_encode"))
				return map[string]scheduling.SchedulerProfile{h.encodeProfile: profiles[h.encodeProfile]}
			}
			// Decider rejected encode - mark as evaluated so we don't re-run the decider.
			profileResults[h.encodeProfile] = nil
			span.SetAttributes(attribute.String("llm_d.profile_handler.decision", "skip_encode"))
		}
	}

	// ── Stage 3: Prefill (optional) ────────────────────────────────────────
	if _, hasPrefillProfile := profiles[h.prefillProfile]; hasPrefillProfile {
		if _, executed := profileResults[h.prefillProfile]; !executed {
			if h.pdDecider != nil && h.pdDecider.disaggregate(ctx, request, decodeRes.TargetEndpoints[0]) {
				span.SetAttributes(attribute.String("llm_d.profile_handler.decision", "run_prefill"))
				return map[string]scheduling.SchedulerProfile{h.prefillProfile: profiles[h.prefillProfile]}
			}
			// Decider rejected prefill - mark as evaluated so we don't re-run the decider.
			profileResults[h.prefillProfile] = nil
			span.SetAttributes(attribute.String("llm_d.profile_handler.decision", "skip_prefill"))
		}
	}

	// ── All stages done: record routing decision ───────────────────────────
	encodeUsed := profileResults[h.encodeProfile] != nil
	prefillUsed := profileResults[h.prefillProfile] != nil

	decision := metrics.DisaggDecisionType(encodeUsed, prefillUsed)
	metrics.RecordDisaggDecision(request.TargetModel, decision)
	span.SetAttributes(attribute.String("llm_d.profile_handler.decision", "complete_"+decision))

	return map[string]scheduling.SchedulerProfile{}
}

// ProcessResults implements scheduling.ProfileHandler.
// Builds the final SchedulingResult from whichever stages ran successfully.
func (h *Handler) ProcessResults(
	_ context.Context,
	_ *scheduling.CycleState,
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

	if prefillRes, ok := profileResults[h.prefillProfile]; ok && prefillRes != nil {
		updatedResults[h.prefillProfile] = prefillRes
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
func (h *Handler) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	tracer := telemetry.Tracer()
	_, span := tracer.Start(ctx, "llm_d.epp.prerequest.disaggregation",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	if request == nil {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.pd.disaggregation_used", false),
			attribute.Bool("llm_d.epp.encode.disaggregation_used", false),
			attribute.String("llm_d.epp.disagg.reason", "request_is_nil"),
		)
		return
	}
	if schedulingResult == nil {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.pd.disaggregation_used", false),
			attribute.Bool("llm_d.epp.encode.disaggregation_used", false),
			attribute.String("llm_d.epp.disagg.reason", "scheduling_result_is_nil"),
		)
		return
	}

	if request.TargetModel != "" {
		span.SetAttributes(attribute.String("gen_ai.request.model", request.TargetModel))
	}
	span.SetAttributes(attribute.String("gen_ai.request.id", request.RequestID))

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
		return
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
		return
	}

	request.Headers[routing.EncoderEndpointsHeader] = strings.Join(encodeHostPorts, ",")
	span.SetAttributes(
		attribute.Bool("llm_d.epp.encode.disaggregation_used", true),
		attribute.String("llm_d.epp.encode.endpoints", strings.Join(encodeHostPorts, ",")),
	)
}
