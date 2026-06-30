// Package prerequest provides pre-request plugins for GIE.
package prerequest

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/telemetry"
)

const (
	// DPRankHeaderHandlerType is the type of the DPRankHeaderHandler plugin.
	DPRankHeaderHandlerType = "dp-rank-header-handler"
)

// compile-time type assertion
var _ requestcontrol.PreRequest = &DPRankHeaderHandler{}

// DPRankHeaderHandlerFactory defines the factory function for the DPRankHeaderHandler.
func DPRankHeaderHandlerFactory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	return NewDPRankHeaderHandler().WithName(name), nil
}

// NewDPRankHeaderHandler initializes a new DPRankHeaderHandler and returns its pointer.
func NewDPRankHeaderHandler() *DPRankHeaderHandler {
	return &DPRankHeaderHandler{
		typedName: plugin.TypedName{Type: DPRankHeaderHandlerType},
	}
}

// DPRankHeaderHandler is a PreRequest plugin that injects the X-data-parallel-rank
// header based on winning DP rank data produced by the scorer.
//
// In vLLM Internal LB and Hybrid LB modes, multiple DP rank engines share a single
// HTTP port. The scorer determines which rank has the best KV cache match for each pod
// and encodes this as a JSON map in the internal x-llm-d-dp-winning-ranks header.
// This plugin reads that internal header, looks up the selected pod's address from
// the SchedulingResult, sets the x-data-parallel-rank header so vLLM can pin the
// request to the best-scoring rank, and removes the internal header.
//
// This design works with any profile handler (single-profile-handler,
// pd-profile-handler, etc.) because it operates in the PreRequest phase,
// after scheduling is complete.
type DPRankHeaderHandler struct {
	typedName plugin.TypedName
}

// TypedName returns the typed name of the plugin.
func (p *DPRankHeaderHandler) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin.
func (p *DPRankHeaderHandler) WithName(name string) *DPRankHeaderHandler {
	p.typedName.Name = name
	return p
}

// PreRequest reads winning DP ranks from the internal header, resolves the selected
// pod's rank, sets X-data-parallel-rank, and removes the internal header.
//
// The X-data-parallel-rank header is single-valued, so this plugin can only pin
// the request to the rank associated with the primary target endpoint. The
// framework routes each HTTP request to one pod (the primary target), so using
// TargetEndpoints[0] matches the request's actual destination. If a profile
// ever returns multiple TargetEndpoints (e.g. future fan-out), only the first
// endpoint's rank is honored and the remainder are ignored with a log line so
// the divergence is observable.
func (p *DPRankHeaderHandler) PreRequest(ctx context.Context, request *scheduling.LLMRequest, schedulingResult *scheduling.SchedulingResult) {
	logger := log.FromContext(ctx)
	tracer := telemetry.Tracer()
	_, span := tracer.Start(ctx, "llm_d.epp.prerequest.dp_rank_header",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	// Read and remove the internal header regardless of outcome.
	encoded, exists := request.Headers[common.DPWinningRanksHeader]
	delete(request.Headers, common.DPWinningRanksHeader)

	if !exists || encoded == "" {
		// No DP winning ranks (External LB mode, non-DP, or scorer didn't produce any).
		span.SetAttributes(attribute.Bool("llm_d.epp.dp.rank_header_set", false))
		return
	}

	winningRanks, err := common.DecodeWinningRanks(encoded)
	if err != nil {
		if !errors.Is(err, common.ErrEmptyWinningRanks) {
			logger.Error(err, "Failed to decode DP winning ranks header")
		}
		span.SetAttributes(attribute.Bool("llm_d.epp.dp.rank_header_set", false))
		return
	}

	if schedulingResult == nil || len(schedulingResult.ProfileResults) == 0 {
		span.SetAttributes(attribute.Bool("llm_d.epp.dp.rank_header_set", false))
		return
	}

	primaryResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if primaryResult == nil || len(primaryResult.TargetEndpoints) == 0 {
		span.SetAttributes(attribute.Bool("llm_d.epp.dp.rank_header_set", false))
		return
	}

	if len(primaryResult.TargetEndpoints) > 1 {
		// Surface the divergence instead of silently dropping extra ranks.
		logger.V(logutil.DEBUG).Info("multiple target endpoints in DP request; honoring rank for first only",
			"count", len(primaryResult.TargetEndpoints))
	}

	targetPod := primaryResult.TargetEndpoints[0].GetMetadata()
	if targetPod == nil {
		span.SetAttributes(attribute.Bool("llm_d.epp.dp.rank_header_set", false))
		return
	}

	podAddress := common.PodAddress(targetPod.GetIPAddress(), targetPod.GetPort())
	rank, found := winningRanks[podAddress]
	if !found {
		span.SetAttributes(
			attribute.Bool("llm_d.epp.dp.rank_header_set", false),
			attribute.String("llm_d.epp.dp.pod_address", podAddress),
		)
		return
	}

	request.Headers[common.DataParallelRankHeader] = strconv.Itoa(rank)
	span.SetAttributes(
		attribute.Bool("llm_d.epp.dp.rank_header_set", true),
		attribute.Int("llm_d.epp.dp.winning_rank", rank),
		attribute.String("llm_d.epp.dp.pod_address", podAddress),
	)
}
