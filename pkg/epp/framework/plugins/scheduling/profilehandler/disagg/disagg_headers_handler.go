// Package disagg provides pre-request plugins for GIE.
package disagg

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/routing"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/telemetry"
)

const (
	// DisaggHeadersHandlerType is the type of the HeadersHandler.
	//
	// Deprecated: Use DisaggProfileHandlerType instead, disagg-profile-handler now implements PreRequest natively.
	DisaggHeadersHandlerType = "disagg-headers-handler"

	// PrefillHeaderHandlerType is a deprecated alias for DisaggHeadersHandlerType.
	//
	// Deprecated: Use DisaggProfileHandlerType instead, disagg-profile-handler now implements PreRequest natively.
	PrefillHeaderHandlerType = "prefill-header-handler"
)

// compile-time type assertion
var _ requestcontrol.PreRequest = &HeadersHandler{}

type disaggHeadersHandlerParameters struct {
	PrefillProfile string `json:"prefillProfile"`
	EncodeProfile  string `json:"encodeProfile"`
}

// HeadersHandlerFactory defines the factory function for the HeadersHandler.
//
// Deprecated: Use HandlerFactory instead, disagg-profile-handler now implements PreRequest natively.
func HeadersHandlerFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	parameters := disaggHeadersHandlerParameters{
		PrefillProfile: defaultPrefillProfile,
		EncodeProfile:  defaultEncodeProfile,
	}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' pre-request plugin - %w", DisaggHeadersHandlerType, err)
		}
	}
	return NewHeadersHandler(parameters.PrefillProfile, parameters.EncodeProfile).WithName(name), nil
}

// NewHeadersHandler initializes a new HeadersHandler and returns its pointer.
//
// Deprecated: Use NewDisaggProfileHandler instead, disagg-profile-handler now implements PreRequest natively.
func NewHeadersHandler(prefillProfile, encodeProfile string) *HeadersHandler {
	return &HeadersHandler{
		typedName:      plugin.TypedName{Type: DisaggHeadersHandlerType},
		prefillProfile: prefillProfile,
		encodeProfile:  encodeProfile,
	}
}

// HeadersHandler PreRequest plugin that sets both prefill and encode disaggregation headers.
//
// Deprecated: Use Handler instead, disagg-profile-handler now implements PreRequest natively.
type HeadersHandler struct {
	typedName      plugin.TypedName
	prefillProfile string
	encodeProfile  string
}

// TypedName returns the typed name of the plugin.
func (p *HeadersHandler) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin.
func (p *HeadersHandler) WithName(name string) *HeadersHandler {
	p.typedName.Name = name
	return p
}

// PreRequest wires prefill and encode SchedulerProfile results into headers to indicate disaggregation workers.
func (p *HeadersHandler) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
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
	delete(request.Headers, routing.PrefillEndpointHeader) // clear header, if already set
	prefillProfileRunResult := schedulingResult.ProfileResults[p.prefillProfile]
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
		request.Headers[routing.PrefillEndpointHeader] = prefillHostPort // in the form of <ip:port>
		span.SetAttributes(
			attribute.Bool("llm_d.epp.pd.disaggregation_used", true),
			attribute.String("llm_d.epp.pd.prefill_pod_address", targetPod.Address),
			attribute.String("llm_d.epp.pd.prefill_pod_port", targetPod.Port),
		)
	}

	// Encode header
	delete(request.Headers, routing.EncoderEndpointsHeader) // clear header, if already set
	encodeProfileRunResult := schedulingResult.ProfileResults[p.encodeProfile]
	if encodeProfileRunResult == nil {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.encode.disaggregation_used", false),
			attribute.String("llm_d.epp.encode.reason", "no_encode_profile_result"),
		)
		return // encode profile failed to run or we chose not to run it, no-op in this case
	}

	// Collect all target endpoints as comma-separated host:port pairs
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
		return // no target endpoints, no-op in this case
	}

	request.Headers[routing.EncoderEndpointsHeader] = strings.Join(encodeHostPorts, ",")
	span.SetAttributes(
		attribute.Bool("llm_d.epp.encode.disaggregation_used", true),
		attribute.String("llm_d.epp.encode.endpoints", strings.Join(encodeHostPorts, ",")),
	)
}
