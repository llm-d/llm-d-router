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
	"errors"
	"maps"
	"net/http"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const ConditionalDecodeStepName = "conditional-decode"

func init() {
	pipeline.Register(ConditionalDecodeStepName, NewConditionalDecodeStep)
}

type ConditionalDecodeStep struct {
	useOpenAIFormat bool
	gwClient        *gateway.Client
}

func NewConditionalDecodeStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("conditional-decode: gateway client is required")
	}
	return &ConditionalDecodeStep{useOpenAIFormat: parseUseOpenAIFormat(params), gwClient: gwClient}, nil
}

func (s *ConditionalDecodeStep) Name() string { return ConditionalDecodeStepName }

func (s *ConditionalDecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(ConditionalDecodeStepName)

	body := maps.Clone(reqCtx.Body)
	s.prepareBody(reqCtx, body)

	logger.V(logutil.DEFAULT).Info("sending request", "path", reqCtx.OriginalPath)

	proxyReq, err := newDecodeProxyRequest(ctx, logger, ConditionalDecodeStepName, reqCtx, s.gwClient, body, map[string]string{"Prefer": "if-available"})
	if err != nil {
		return err
	}

	var cacheMiss bool
	proxy := newDecodeProxy(logger, s.gwClient.Transport(), func(resp *http.Response) error {
		if resp.StatusCode == http.StatusPreconditionFailed {
			cacheMiss = true
			return errCacheMiss
		}
		return nil
	})
	proxy.ServeHTTP(reqCtx.ResponseWriter, proxyReq)

	if cacheMiss {
		logger.V(logutil.DEFAULT).Info("cache miss (412), continuing pipeline")
		return nil
	}

	logger.V(logutil.DEFAULT).Info("cache hit, response forwarded")
	return pipeline.ErrPipelineDone
}

func (s *ConditionalDecodeStep) prepareBody(reqCtx *pipeline.RequestContext, body map[string]any) {
	format := resolveFormat(s.useOpenAIFormat, reqCtx.OriginalPath)
	switch format {
	case gateway.FormatChatCompletions:
		if len(reqCtx.TokenIDs) > 0 {
			tokens := map[string]any{
				"token_ids": reqCtx.TokenIDs,
			}
			if features := buildMMFeatures(reqCtx.MultimodalEntries, false); features != nil {
				tokens["features"] = features
			}
			body["tokens"] = tokens
		}
	case gateway.FormatCompletions:
		if len(reqCtx.TokenIDs) > 0 {
			body["prompt"] = reqCtx.TokenIDs
		}
	}
}
