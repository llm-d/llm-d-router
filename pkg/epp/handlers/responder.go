/*
Copyright 2026 The Kubernetes Authors.

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

package handlers

import (
	"context"
	"net/http"
	"strings"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	envoy "github.com/llm-d/llm-d-router/pkg/common/envoy"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
)

// tryRespondLocally offers the request to each configured Responder in order. The first one
// to return a response answers it, and the caller must stop and return the error it gets
// back (nil on success; the response is stored on reqCtx and sent by the state machine).
//
// It reports false when no responder claimed the request, in which case normal routing
// continues.
func (s *StreamingServer) tryRespondLocally(ctx context.Context, reqCtx *RequestContext, req *extProcPb.ProcessingRequest_RequestHeaders) (bool, error) {
	if len(s.responders) == 0 {
		return false, nil
	}
	// Only GET is supported; extend this check when a responder needs another method.
	if envoy.ExtractHeaderValue(req, ":method") != http.MethodGet {
		return false, nil
	}

	path, _, _ := strings.Cut(envoy.ExtractHeaderValue(req, ":path"), "?")
	request := &fwkrc.RequestLine{
		Method: http.MethodGet,
		Path:   path,
	}

	var endpoints = s.datastore.PodList(func(_ fwkdl.Endpoint) bool { return true })
	for _, responder := range s.responders {
		resp, err := responder.Respond(ctx, request, endpoints)
		if err != nil {
			return true, err
		}
		if resp == nil {
			continue
		}
		reqCtx.localResp = immediateResponse(resp)
		reqCtx.requestState = requestAnsweredLocal
		log.FromContext(ctx).V(logutil.DEFAULT).Info("EPP answered request locally",
			"responder", responder.TypedName(), "path", request.Path)
		return true, nil
	}
	return false, nil
}

// immediateResponse renders a plugin response as the ext-proc message that ends the exchange.
func immediateResponse(resp *fwkrc.LocalResponse) *extProcPb.ProcessingResponse {
	headers := make([]*configPb.HeaderValueOption, 0, len(resp.Headers))
	for key, value := range resp.Headers {
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{Key: key, RawValue: []byte(value)},
		})
	}

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extProcPb.ImmediateResponse{
				Status:  &envoyTypePb.HttpStatus{Code: envoyTypePb.StatusCode(resp.StatusCode)},
				Headers: &extProcPb.HeaderMutation{SetHeaders: headers},
				Body:    resp.Body,
			},
		},
	}
}
