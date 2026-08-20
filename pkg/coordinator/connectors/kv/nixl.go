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
	"strconv"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// nixlKV implements the NIXL P2P KV transfer protocol. The prefill request
// declares the request will be remote-decoded; the decode request forwards
// the prefill response's kv_transfer_params verbatim plus do_remote_prefill
// so the decode pod can pull KV blocks from the prefill pod.
type nixlKV struct {
	// dpSize is the model server's data-parallel world size. 1 disables
	// DP-rank pinning (PrefillHeaders/DecodeHeaders return nil).
	dpSize int
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
	out := make(map[string]any, len(reqCtx.KVTransferParams)+2)
	for k, v := range reqCtx.KVTransferParams {
		out[k] = v
	}
	out[reqcommon.FieldDoRemoteDecode] = false
	out[reqcommon.FieldDoRemotePrefill] = true
	log.FromContext(ctx).WithName(loggerName).V(logutil.TRACE).Info("preparing decode kv params", "params", out)
	return out
}

// PrefillHeaders pins the prefill request to a deterministic DP rank hashed
// from the request ID, so a DP>1 backend that shares its HTTP port across
// ranks serves both legs of this disaggregated pair from the same rank.
// Returns nil when DP is disabled (dpSize<=1).
func (n nixlKV) PrefillHeaders(_ context.Context, reqCtx *pipeline.RequestContext) map[string]string {
	if n.dpSize <= 1 {
		return nil
	}
	rank := pickDPRank(reqCtx.RequestID, n.dpSize)
	return map[string]string{routing.DataParallelRankHeader: strconv.Itoa(rank)}
}

// DecodeHeaders pins the decode request to the same DP rank as the prefill
// leg: the rank the prefill response reported (if valid), else the same
// deterministic hash PrefillHeaders used. Returns nil when DP is disabled
// (dpSize<=1).
func (n nixlKV) DecodeHeaders(_ context.Context, reqCtx *pipeline.RequestContext) map[string]string {
	if n.dpSize <= 1 {
		return nil
	}
	rank, _ := resolveDecodeDPRank(reqCtx.KVTransferParams, reqCtx.RequestID, n.dpSize)
	return map[string]string{routing.DataParallelRankHeader: strconv.Itoa(rank)}
}
