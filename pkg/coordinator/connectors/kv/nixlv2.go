package kv

import (
	"github.com/llm-d/coordinator/pkg/pipeline"
	logutil "github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
)

// nixlV2 implements the NIXL v2 P2P KV transfer protocol. The prefill request
// declares the request will be remote-decoded; the decode request forwards
// the prefill response's kv_transfer_params verbatim plus do_remote_prefill
// so the decode pod can pull KV blocks from the prefill pod.
type nixlV2 struct{}

func (nixlV2) Name() string { return NIXLv2 }

func (nixlV2) PreparePrefillKVParams(_ *pipeline.RequestContext) map[string]any {
	params := map[string]any{
		"do_remote_decode":  true,
		"do_remote_prefill": false,
		"remote_engine_id":  nil,
		"remote_block_ids":  nil,
		"remote_host":       nil,
		"remote_port":       nil,
	}
	logger.V(logutil.TRACE).Info("preparing prefill kv params", "params", params)
	return params
}

func (nixlV2) PrepareDecodeKVParams(reqCtx *pipeline.RequestContext) map[string]any {
	out := make(map[string]any, len(reqCtx.KVTransferParams)+1)
	for k, v := range reqCtx.KVTransferParams {
		out[k] = v
	}
	out["do_remote_prefill"] = true
	logger.V(logutil.TRACE).Info("preparing decode kv params", "params", out)
	return out
}
