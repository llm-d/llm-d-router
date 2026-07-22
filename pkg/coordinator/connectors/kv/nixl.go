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
// declares the request will be remote-decoded; the decode request forwards
// the prefill response's kv_transfer_params verbatim plus do_remote_prefill
// so the decode pod can pull KV blocks from the prefill pod.
type nixlKV struct{}

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
