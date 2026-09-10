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
	"github.com/llm-d/llm-d-router/pkg/coordinator/engine/vllm"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"golang.org/x/sync/errgroup"
)

const EncodeStepName = "encode"

func init() {
	pipeline.Register(EncodeStepName, NewEncodeStep)
}

type EncodeStep struct {
	engine      vllm.Engine
	maxParallel int
	gwClient    *gateway.Client
	ec          ec.Connector
}

func NewEncodeStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("encode: gateway client is required")
	}
	useOpenAI, err := parseUseOpenAIFormat(params)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	selectedEngine := vllm.New(useOpenAI)
	maxParallel := 8
	if v, ok, err := paramInt(params, "max_parallel"); err != nil {
		return nil, err
	} else if ok {
		if v <= 0 {
			return nil, fmt.Errorf("max_parallel must be positive, got %d", v)
		}
		maxParallel = v
	}
	ecName, err := paramString(params, ParamECConnector)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	ecConn, err := ec.Build(ecName)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	return &EncodeStep{
		engine:      selectedEngine,
		maxParallel: maxParallel,
		gwClient:    gwClient,
		ec:          ecConn,
	}, nil
}

func (s *EncodeStep) Name() string { return EncodeStepName }

func (s *EncodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	if len(reqCtx.MultimodalEntries) == 0 {
		return nil
	}

	logger := log.FromContext(ctx).WithName(EncodeStepName)

	prepare := s.engine.PrepareEncode(reqCtx)
	if prepare == nil {
		logger.V(logutil.DEFAULT).Info("skipping encode for generate request")
		return nil
	}

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxParallel)

	results := make([]map[string]any, len(reqCtx.MultimodalEntries))

	for i, entry := range reqCtx.MultimodalEntries {
		g.Go(func() error {
			prepared := prepare(entry)
			bodyBytes, err := json.Marshal(prepared.Body)
			if err != nil {
				err = fmt.Errorf("encode[%d]: marshal: %w", i, err)
				logger.Error(err, "encode fanout marshal", "index", i)
				return err
			}
			path := prepared.Path
			logger.V(logutil.DEFAULT).Info("sending sub-request", "index", i, "path", path)

			headers := reqCtx.ForwardedHeaders()
			headers[reqcommon.RequestIDHeaderKey] = reqCtx.RequestID
			headers[gateway.EPPProfileHeader] = gateway.PhaseEncode

			if v := logger.V(logutil.DEBUG); v.Enabled() {
				v.Info("sub-request body", "index", i, "method", "POST", "path", path, "bodyLen", len(bodyBytes), "headers", httplog.RedactedHeaders(headers))
			}

			call := coordmetrics.StartUpstreamCall(coordmetrics.UpstreamEncode)
			resp, err := s.gwClient.Post(gCtx, path, bodyBytes, headers)
			call.Done()
			if err != nil {
				err = fmt.Errorf("encode[%d]: request: %w", i, err)
				logger.Error(err, "encode fanout request", "index", i, "path", path)
				return err
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				respBody := readErrorBody(resp.Body)
				err := upstreamError(fmt.Sprintf("%s[%d]", EncodeStepName, i), resp.StatusCode, respBody)
				logger.Error(err, "encode fanout status", "index", i, "status", resp.StatusCode)
				return err
			}

			params, err := s.engine.ReadEncodeResponse(resp.Body)
			if err != nil {
				err = fmt.Errorf("encode[%d]: decode response: %w", i, err)
				logger.Error(err, "encode fanout decode", "index", i)
				return err
			}

			results[i] = coerceParamsMap(logger.WithValues("index", i), params, "ec_transfer_params")
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	for _, r := range results {
		s.ec.MergeEncodeResponse(ctx, reqCtx, r)
	}

	logger.V(logutil.DEFAULT).Info("all sub-requests complete", "count", len(results))
	return nil
}
