package steps

import (
	"context"
	"net/http"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

const ConditionalDecodeStepName = "conditional-decode"

func init() {
	pipeline.Register(ConditionalDecodeStepName, NewConditionalDecodeStep)
}

type ConditionalDecodeStep struct {
	useOpenAIFormat bool
	gwClient        *gateway.Client
}

func NewConditionalDecodeStep(params map[string]any) (pipeline.Step, error) {
	return &ConditionalDecodeStep{useOpenAIFormat: parseUseOpenAIFormat(params)}, nil
}

func (s *ConditionalDecodeStep) SetGatewayClient(c *gateway.Client) {
	s.gwClient = c
}

func (s *ConditionalDecodeStep) Name() string { return ConditionalDecodeStepName }

func (s *ConditionalDecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(ConditionalDecodeStepName)

	body := copyBody(reqCtx.Body)
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
