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

package steps

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/coordinator/pkg/common/httplog"
	"github.com/llm-d/coordinator/pkg/connectors/ec"
	"github.com/llm-d/coordinator/pkg/connectors/kv"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

const PrefillStepName = "prefill"

func init() {
	pipeline.Register(PrefillStepName, NewPrefillStep)
}

type PrefillStep struct {
	useOpenAIFormat bool
	gwClient        *gateway.Client
	kv              kv.Connector
	ec              ec.Connector
}

func NewPrefillStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("prefill: gateway client is required")
	}
	useOpenAI := parseUseOpenAIFormat(params)
	kvName, _ := params[ParamKVConnector].(string)
	kvConn, err := kv.Build(kvName)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	ecName, _ := params[ParamECConnector].(string)
	ecConn, err := ec.Build(ecName)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	return &PrefillStep{useOpenAIFormat: useOpenAI, gwClient: gwClient, kv: kvConn, ec: ecConn}, nil
}

func (s *PrefillStep) Name() string { return PrefillStepName }

func (s *PrefillStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(PrefillStepName)

	features := buildMMFeatures(reqCtx.MultimodalEntries, true)

	format := resolveFormat(s.useOpenAIFormat, reqCtx.OriginalPath)
	body, err := s.buildPrefillBody(ctx, reqCtx, features, format)
	if err != nil {
		return fmt.Errorf("prefill: %w", err)
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("prefill: marshal: %w", err)
	}

	path := gateway.PathForFormat(format)
	logger.V(logutil.DEFAULT).Info("sending request", "path", path)

	headers := reqCtx.ForwardedHeaders()
	headers[reqcommon.RequestIDHeaderKey] = reqCtx.RequestID
	headers[gateway.EPPPhaseHeader] = gateway.PhasePrefill

	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info("request body", "method", "POST", "path", path, "bodyLen", len(bodyBytes), "headers", httplog.RedactedHeaders(headers))
	}

	resp, err := s.gwClient.Post(ctx, path, bodyBytes, headers)
	if err != nil {
		return fmt.Errorf("prefill: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		respBody := readErrorBody(resp.Body)
		return upstreamError(PrefillStepName, resp.StatusCode, respBody)
	}

	var prefillResp prefillResponse
	if err := json.NewDecoder(resp.Body).Decode(&prefillResp); err != nil {
		return fmt.Errorf("prefill: decode response: %w", err)
	}

	reqCtx.KVTransferParams = coerceParamsMap(logger, prefillResp.KVTransferParams, "kv_transfer_params")

	logger.V(logutil.DEFAULT).Info("complete")
	return nil
}

func (s *PrefillStep) buildPrefillBody(ctx context.Context, reqCtx *pipeline.RequestContext, features map[string]any, format gateway.RequestFormat) (map[string]any, error) {
	ecParams, err := s.ec.PreparePrefillECParams(ctx, reqCtx)
	if err != nil {
		return nil, err
	}
	kvParams := s.kv.PreparePrefillKVParams(ctx, reqCtx)

	switch format {
	case gateway.FormatChatCompletions:
		body := maps.Clone(reqCtx.Body)
		tokens := map[string]any{
			"token_ids": reqCtx.TokenIDs,
		}
		if features != nil {
			tokensFeatures := map[string]any{
				"mm_hashes":       features["mm_hashes"],
				"mm_placeholders": features["mm_placeholders"],
			}
			tokens["features"] = tokensFeatures
		}
		body["tokens"] = tokens
		body["max_tokens"] = 1
		body["kv_transfer_params"] = kvParams
		if len(ecParams) > 0 {
			body["ec_transfer_params"] = ecParams
		}
		return body, nil

	case gateway.FormatCompletions:
		prompt := reqCtx.Body["prompt"]
		if len(reqCtx.TokenIDs) > 0 {
			prompt = reqCtx.TokenIDs
		}
		body := map[string]any{
			"request_id":         reqCtx.RequestID,
			"model":              reqCtx.Model,
			"prompt":             prompt,
			"max_tokens":         1,
			"kv_transfer_params": kvParams,
		}
		if features != nil {
			body["features"] = features
		}
		if len(ecParams) > 0 {
			body["ec_transfer_params"] = ecParams
		}
		return body, nil

	default:
		body := map[string]any{
			"request_id": reqCtx.RequestID,
			"token_ids":  reqCtx.TokenIDs,
			"model":      reqCtx.Model,
			"sampling_params": map[string]any{
				"max_tokens": 1,
				"extra_args": map[string]any{
					"kv_transfer_params": kvParams,
				},
			},
		}
		if features != nil {
			body["features"] = features
		}
		if len(ecParams) > 0 {
			body["ec_transfer_params"] = ecParams
		}
		return body, nil
	}
}

type prefillResponse struct {
	// KVTransferParams is decoded as any (not map[string]any) so a non-object
	// value does not fail the decode; coerceParamsMap coerces it.
	KVTransferParams any `json:"kv_transfer_params"`
}
