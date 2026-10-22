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
	"fmt"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// offloadingKV implements the vLLM OffloadingConnector P2P tier protocol. The
// prefill request declares a remote decoder under a fresh kv_request_id, so
// the prefill pod stores the KV in its P2P tier for a pull. The decode request
// names the prefill pod under the same kv_request_id, so the decode pod pulls
// the KV from it.
type offloadingKV struct{}

func (offloadingKV) Name() string { return Offloading }

func (offloadingKV) PreparePrefillKVParams(ctx context.Context, reqCtx *pipeline.RequestContext) map[string]any {
	reqCtx.KVRequestID = uuid.NewString()
	params := reqcommon.RemoteDecoderParams(reqCtx.KVRequestID)
	log.FromContext(ctx).WithName(loggerName).V(logutil.TRACE).Info("preparing prefill kv params", "params", params)
	return params
}

// ReadPrefillResponse reads the prefill pod's P2P tier host and port from the
// header the EPP p2p-source-producer sets on the prefill request and
// request-header-copy-handler copies onto the response. The decode request
// cannot be built without it, so a missing or malformed header is an error.
func (offloadingKV) ReadPrefillResponse(reqCtx *pipeline.RequestContext, headers http.Header) error {
	value := strings.TrimSpace(headers.Get(routing.KVCachePrefillHeader))
	if value == "" {
		return fmt.Errorf("prefill response has no %s header; the prefill EPP must run p2p-source-producer and request-header-copy-handler, and its ext_proc filter must set response_header_mode: SEND",
			routing.KVCachePrefillHeader)
	}
	host, port, ok := routing.ParseHostPort(value)
	if !ok {
		return fmt.Errorf("prefill response header %s %q is not a host:port with a TCP port",
			routing.KVCachePrefillHeader, value)
	}
	reqCtx.PrefillPeer = &pipeline.P2PPeer{Host: host, Port: port}
	return nil
}

// PrepareDecodeKVParams is empty unless this connector read the prefill
// response, so a pipeline whose prefill step ran another connector, or has no
// prefill step, leaves kv_transfer_params to the EPP plugin.
func (offloadingKV) PrepareDecodeKVParams(ctx context.Context, reqCtx *pipeline.RequestContext) map[string]any {
	logger := log.FromContext(ctx).WithName(loggerName)
	params := map[string]any{}
	if peer := reqCtx.PrefillPeer; peer != nil {
		params = reqcommon.RemotePrefillerParams(reqCtx.KVRequestID, peer.Host, peer.Port)
	} else {
		// The connector cannot tell the two causes apart, so the line names
		// both: the first is the normal case, the second is the one to act on.
		logger.Info("no prefill P2P peer; kv_transfer_params is left to the EPP, as expected without a prefill step; a prefill step on another kv_connector stored KV that decode will not pull")
	}
	logger.V(logutil.TRACE).Info("preparing decode kv params", "params", params)
	return params
}
