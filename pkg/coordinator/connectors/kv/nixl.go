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

package kv

import (
	"context"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// nixlKV implements the NIXL P2P KV transfer protocol. The prefill request
// declares the request will be remote-decoded; when the prefill response
// carries remote_block_ids (NIXLHasRemoteBlockIDs), the decode request
// forwards its kv_transfer_params verbatim plus do_remote_prefill so the
// decode pod can pull the blocks from the prefill pod.
type nixlKV struct{}

// NIXLHasRemoteBlockIDs reports whether kv_transfer_params carries a non-empty
// remote_block_ids, the truthiness vLLM's NIXL scheduler applies to the field.
func NIXLHasRemoteBlockIDs(params map[string]any) bool {
	ids, ok := params[reqcommon.FieldRemoteBlockIDs].([]any)
	return ok && len(ids) > 0
}

func (nixlKV) Name() string { return NIXL }

func (nixlKV) PreparePrefillKVParams(ctx context.Context, _ *pipeline.RequestContext) map[string]any {
	params := map[string]any{
		reqcommon.FieldDoRemoteDecode:  true,
		reqcommon.FieldDoRemotePrefill: false,
		reqcommon.FieldRemoteEngineID:  nil,
		reqcommon.FieldRemoteBlockIDs:  nil,
		reqcommon.FieldRemoteHost:      nil,
		reqcommon.FieldRemotePort:      nil,
	}
	log.FromContext(ctx).WithName(loggerName).V(logutil.TRACE).Info("preparing prefill kv params", "params", params)
	return params
}

func (nixlKV) PrepareDecodeKVParams(ctx context.Context, reqCtx *pipeline.RequestContext) map[string]any {
	logger := log.FromContext(ctx).WithName(loggerName)
	// Without kv_transfer_params.remote_block_ids, do_remote_prefill fails a vLLM assertion that exits the engine.
	if !NIXLHasRemoteBlockIDs(reqCtx.KVTransferParams) {
		logger.V(logutil.TRACE).Info("no prefill kv_transfer_params.remote_block_ids; decode computes the prompt")
		return map[string]any{}
	}
	out := make(map[string]any, len(reqCtx.KVTransferParams)+2)
	for k, v := range reqCtx.KVTransferParams {
		out[k] = v
	}
	out[reqcommon.FieldDoRemoteDecode] = false
	out[reqcommon.FieldDoRemotePrefill] = true
	logger.V(logutil.TRACE).Info("preparing decode kv params", "params", out)
	return out
}
