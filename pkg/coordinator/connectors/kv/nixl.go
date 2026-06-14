package kv

import (
	"github.com/llm-d/coordinator/pkg/pipeline"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// nixlKV implements the NIXL P2P KV transfer protocol. The prefill request
// declares the request will be remote-decoded; the decode request forwards
// the prefill response's kv_transfer_params verbatim plus do_remote_prefill
// so the decode pod can pull KV blocks from the prefill pod.
type nixlKV struct{}

func (nixlKV) Name() string { return NIXL }

func (nixlKV) PreparePrefillKVParams(_ *pipeline.RequestContext) map[string]any {
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

func (nixlKV) PrepareDecodeKVParams(reqCtx *pipeline.RequestContext) map[string]any {
	out := make(map[string]any, len(reqCtx.KVTransferParams)+2)
	for k, v := range reqCtx.KVTransferParams {
		out[k] = v
	}
	out["do_remote_decode"] = false
	out["do_remote_prefill"] = true
	logger.V(logutil.TRACE).Info("preparing decode kv params", "params", out)
	return out
}
