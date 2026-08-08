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
	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// modelsPath is the OpenAI-compatible model discovery endpoint.
const modelsPath = "/v1/models"

// tryServeModelList handles GET /v1/models. It returns the combined list of every model and LoRA
// adapter across all pods. We build this list ourselves instead of forwarding to one pod, because
// a single pod only knows about the adapters loaded on itself.
//
// It returns true when the request was /v1/models and we handled it. When true, the caller should
// stop and return the error it gets back (nil means success; the response is already saved on reqCtx).
func (s *StreamingServer) tryServeModelList(ctx context.Context, reqCtx *RequestContext, req *extProcPb.ProcessingRequest_RequestHeaders) (bool, error) {
	if envoy.ExtractHeaderValue(req, ":method") != http.MethodGet {
		return false, nil
	}
	path, _, _ := strings.Cut(envoy.ExtractHeaderValue(req, ":path"), "?")
	if strings.TrimRight(path, "/") != modelsPath {
		return false, nil
	}

	body, collected := s.datastore.AggregateModels()
	if collected == 0 {
		// No endpoint has reported its model list yet: either the pool is empty, or the pods are
		// still being scraped after a cold start or scale-up (scraping is interval-based). Return
		// 503 so the client retries, instead of a misleading empty 200.
		return true, errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "no model data collected yet from any endpoint"}
	}

	reqCtx.localResp = &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{Code: envoyTypePb.StatusCode_OK},
				Headers: &extProcPb.HeaderMutation{
					SetHeaders: []*configPb.HeaderValueOption{
						{
							Header: &configPb.HeaderValue{
								Key:      "content-type",
								RawValue: []byte("application/json"),
							},
						},
					},
				},
				Body: body,
			},
		},
	}
	reqCtx.RequestState = RequestAnsweredLocal
	log.FromContext(ctx).V(logutil.DEFAULT).Info("EPP answered /v1/models from datalayer state")
	return true, nil
}
