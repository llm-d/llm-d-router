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
	"encoding/json"
	"net/http"
	"slices"
	"strings"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	envoy "github.com/llm-d/llm-d-router/pkg/common/envoy"
	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	extmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/models"
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
	if !strings.EqualFold(envoy.ExtractHeaderValue(req, ":method"), http.MethodGet) {
		return false, nil
	}
	path, _, _ := strings.Cut(envoy.ExtractHeaderValue(req, ":path"), "?")
	if strings.TrimRight(path, "/") != modelsPath {
		return false, nil
	}

	models, collected := s.aggregateModels()
	if collected == 0 {
		// No endpoint has reported its model list yet: either the pool is empty, or the pods are
		// still being scraped after a cold start or scale-up (scraping is interval-based). Return
		// 503 so the client retries, instead of a misleading empty 200.
		return true, errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "no model data collected yet from any endpoint"}
	}

	body, err := json.Marshal(models)
	if err != nil {
		return true, errcommon.Error{Code: errcommon.Internal, Msg: "failed to marshal /v1/models response: " + err.Error()}
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

// aggregateModels collects the models from all endpoints in the datastore and returns one entry per
// unique model ID, sorted alphabetically so the response is the same every time. The second return
// value is how many endpoints have actually reported a model list (been scraped), letting the caller
// tell "nothing collected yet" (empty pool or still warming up) from a genuinely empty result.
func (s *StreamingServer) aggregateModels() (extmodels.ModelResponse, int) {
	endpoints := s.datastore.PodList(func(fwkdl.Endpoint) bool { return true })
	// The same model appears on many pods, but we keep only one copy of it below.
	// Sort the pods first so we always keep the same pod's copy and the answer stays consistent.
	slices.SortFunc(endpoints, func(a, b fwkdl.Endpoint) int {
		return strings.Compare(a.GetMetadata().ID.String(), b.GetMetadata().ID.String())
	})

	seen := make(map[string]struct{})
	data := make([]attrmodels.ModelData, 0)
	collected := 0 // endpoints that have reported their model list at least once (been scraped)

	for _, ep := range endpoints {
		c, ok := fwkdl.ReadAttribute[attrmodels.ModelDataCollection](ep.GetAttributes(), attrmodels.ModelsAttributeKey.String())
		if !ok {
			continue // registered but not scraped yet; do not count it
		}
		collected++
		for _, model := range c {
			if _, dup := seen[model.ID]; dup {
				continue
			}
			seen[model.ID] = struct{}{}
			data = append(data, model)
		}
	}
	// return the models in a fixed (alphabetical) order every time
	slices.SortFunc(data, func(a, b attrmodels.ModelData) int { return strings.Compare(a.ID, b.ID) })
	return extmodels.ModelResponse{Object: "list", Data: data}, collected
}
