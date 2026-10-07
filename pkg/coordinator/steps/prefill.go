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

	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
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
	useOpenAI, err := parseUseOpenAIFormat(params)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	kvConn, err := buildKVConnector(params)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	ecConn, err := buildECConnector(params)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	return &PrefillStep{
		useOpenAIFormat: useOpenAI,
		gwClient:        gwClient,
		kv:              kvConn,
		ec:              ecConn,
	}, nil
}

func (s *PrefillStep) Name() string { return PrefillStepName }

func (s *PrefillStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(PrefillStepName)

	format := resolveFormat(s.useOpenAIFormat, reqCtx.OriginalPath)
	body, err := s.buildPrefillBody(ctx, reqCtx, format)
	if err != nil {
		return fmt.Errorf("prefill: %w", err)
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("prefill: marshal: %w", err)
	}

	path := format.Path()
	logger.V(logutil.DEFAULT).Info("sending request", "path", path)

	resp, err := postToGateway(ctx, logger, s.gwClient, gatewayRequest{
		logMsg:   "request body",
		step:     PrefillStepName,
		upstream: coordmetrics.UpstreamPrefill,
		path:     path,
		body:     bodyBytes,
		headers:  gatewayHeaders(reqCtx, gateway.PhasePrefill),
	})
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var prefillResp prefillResponse
	if err := json.NewDecoder(resp.Body).Decode(&prefillResp); err != nil {
		return fmt.Errorf("prefill: decode response: %w", err)
	}

	reqCtx.KVTransferParams = coerceParamsMap(logger, prefillResp.KVTransferParams, "kv_transfer_params")
	if len(reqCtx.KVTransferParams) == 0 && s.kv.Name() == kv.NIXL {
		// kv-nixl always requests a remote-decode handoff, so missing
		// kv_transfer_params means decode will recompute the whole prompt.
		// Older engines (< v0.29.0) silently drop top-level transfer params
		// on this route, which is the most common cause.
		logger.Info("prefill returned no kv_transfer_params; decode will recompute the prompt",
			"kvConnector", s.kv.Name(), "path", path)
	}
	reqCtx.CaptureResponseHeaders(resp.Header)

	logger.V(logutil.DEFAULT).Info("complete")
	return nil
}

func (s *PrefillStep) buildPrefillBody(ctx context.Context, reqCtx *pipeline.RequestContext, format reqcommon.APIType) (map[string]any, error) {
	ecParams, err := s.ec.PreparePrefillECParams(ctx, reqCtx)
	if err != nil {
		return nil, err
	}
	kvParams := s.kv.PreparePrefillKVParams(ctx, reqCtx)

	switch format {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses:
		body := maps.Clone(reqCtx.Body)
		reqcommon.CapSingleToken(body, format)
		body[reqcommon.FieldKVTransferParams] = kvParams
		if len(ecParams) > 0 {
			body[reqcommon.FieldECTransferParams] = ecParams
		}
		return body, nil

	case reqcommon.APITypeCompletions:
		prompt := reqCtx.Body["prompt"]
		if len(reqCtx.TokenIDs) > 0 {
			prompt = reqCtx.TokenIDs
		}
		body := map[string]any{
			"request_id":                    reqCtx.RequestID,
			"model":                         reqCtx.Model,
			"prompt":                        prompt,
			reqcommon.FieldKVTransferParams: kvParams,
		}
		reqcommon.CapSingleToken(body, format)
		// No features here: the Completions API is text-in and vLLM ignores
		// multimodal feature fields on this route, so attaching them (as this
		// branch historically did) only padded the wire.
		if len(ecParams) > 0 {
			body[reqcommon.FieldECTransferParams] = ecParams
		}
		return body, nil

	case reqcommon.APITypeVLLMGenerate:
		body := map[string]any{
			"request_id":                    reqCtx.RequestID,
			"token_ids":                     reqCtx.TokenIDs,
			"model":                         reqCtx.Model,
			reqcommon.FieldKVTransferParams: kvParams,
		}
		reqcommon.CapSingleToken(body, format)
		// The only consumer of the per-entry mm_metadata/kwargs_data choice;
		// see buildPrefillMMFeatures. Built inside this branch rather than
		// above the switch so chat-completions requests never build-and-drop
		// a features map they do not ship.
		if features := buildPrefillMMFeatures(reqCtx.MultimodalEntries, ecParams); features != nil {
			body["features"] = features
		}
		if len(ecParams) > 0 {
			body[reqcommon.FieldECTransferParams] = ecParams
		}
		return body, nil

	default:
		// resolveFormat above never returns anything but the formats handled
		// here, so this case is unreachable unless resolveFormat's contract
		// changes.
		return nil, unreachableFormatError(format)
	}
}

type prefillResponse struct {
	// KVTransferParams is decoded as any (not map[string]any) so a non-object
	// value does not fail the decode; coerceParamsMap coerces it.
	KVTransferParams any `json:"kv_transfer_params"`
}
