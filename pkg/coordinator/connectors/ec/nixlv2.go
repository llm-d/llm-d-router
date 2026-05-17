package ec

import (
	"github.com/llm-d/coordinator/pkg/pipeline"
	logutil "github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
)

// nixlV2 is the NIXL EC connector: each encoder response carries an
// ec_transfer_params object keyed by the encoded image's mm_hash. The
// coordinator collects them in order and forwards them as a per-modality
// list on the prefill request: {"image": [{"hash1": {...}}, ...]}.
type nixlV2 struct{}

func (nixlV2) Name() string { return NIXLv2 }

func (nixlV2) MergeEncodeResponse(reqCtx *pipeline.RequestContext, encResp map[string]any) {
	if len(encResp) == 0 {
		return
	}
	reqCtx.ECTransferParams = append(reqCtx.ECTransferParams, encResp)
	logger.V(logutil.TRACE).Info("merged encode response", "total", len(reqCtx.ECTransferParams))
}

func (nixlV2) PreparePrefillECParams(reqCtx *pipeline.RequestContext) map[string]any {
	if len(reqCtx.ECTransferParams) == 0 {
		return nil
	}
	params := map[string]any{"image": reqCtx.ECTransferParams}
	logger.V(logutil.TRACE).Info("preparing prefill ec params", "images", len(reqCtx.ECTransferParams))
	return params
}
