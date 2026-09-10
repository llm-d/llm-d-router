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
	"net/http"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/common/httplog"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/engine/vllm"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const PrefillStepName = "prefill"

func init() {
	pipeline.Register(PrefillStepName, NewPrefillStep)
}

type PrefillStep struct {
	engine   vllm.Engine
	gwClient *gateway.Client
	kv       kv.Connector
	ec       ec.Connector
}

func NewPrefillStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("prefill: gateway client is required")
	}
	useOpenAI, err := parseUseOpenAIFormat(params)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	selectedEngine := vllm.New(useOpenAI)
	kvName, err := paramString(params, ParamKVConnector)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	kvConn, err := kv.Build(kvName)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	ecName, err := paramString(params, ParamECConnector)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	ecConn, err := ec.Build(ecName)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	return &PrefillStep{engine: selectedEngine, gwClient: gwClient, kv: kvConn, ec: ecConn}, nil
}

func (s *PrefillStep) Name() string { return PrefillStepName }

func (s *PrefillStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(PrefillStepName)

	ecParams, err := s.ec.PreparePrefillECParams(ctx, reqCtx)
	if err != nil {
		return fmt.Errorf("prefill: %w", err)
	}
	kvParams := s.kv.PreparePrefillKVParams(ctx, reqCtx)
	prepared, err := s.engine.PreparePrefill(reqCtx, kvParams, ecParams)
	if err != nil {
		return err
	}
	bodyBytes, err := json.Marshal(prepared.Body)
	if err != nil {
		return fmt.Errorf("prefill: marshal: %w", err)
	}
	path := prepared.Path
	logger.V(logutil.DEFAULT).Info("sending request", "path", path)

	headers := reqCtx.ForwardedHeaders()
	headers[reqcommon.RequestIDHeaderKey] = reqCtx.RequestID
	headers[gateway.EPPProfileHeader] = gateway.PhasePrefill

	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info("request body", "method", "POST", "path", path, "bodyLen", len(bodyBytes), "headers", httplog.RedactedHeaders(headers))
	}

	call := coordmetrics.StartUpstreamCall(coordmetrics.UpstreamPrefill)
	resp, err := s.gwClient.Post(ctx, path, bodyBytes, headers)
	call.Done()
	if err != nil {
		return fmt.Errorf("prefill: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody := readErrorBody(resp.Body)
		return upstreamError(PrefillStepName, resp.StatusCode, respBody)
	}

	params, err := s.engine.ReadPrefillResponse(resp.Body)
	if err != nil {
		return err
	}
	reqCtx.KVTransferParams = coerceParamsMap(logger, params, "kv_transfer_params")

	logger.V(logutil.DEFAULT).Info("complete")
	return nil
}
