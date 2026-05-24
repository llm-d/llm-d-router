package steps

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

const ConditionalDecodeStepName = "conditional-decode"

var errCacheMiss = errors.New("cache miss")

func init() {
	pipeline.Register(ConditionalDecodeStepName, NewConditionalDecodeStep)
}

type ConditionalDecodeStep struct {
	useOpenAIFormat bool
	gwClient        *gateway.Client
}

func NewConditionalDecodeStep(params map[string]any) (pipeline.Step, error) {
	useOpenAI := true
	if params != nil {
		if v, ok := params["use_openai_format"].(bool); ok {
			useOpenAI = v
		}
	}
	return &ConditionalDecodeStep{useOpenAIFormat: useOpenAI}, nil
}

func (s *ConditionalDecodeStep) SetGatewayClient(c *gateway.Client) {
	s.gwClient = c
}

func (s *ConditionalDecodeStep) Name() string { return ConditionalDecodeStepName }

func (s *ConditionalDecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(ConditionalDecodeStepName)

	body := copyBody(reqCtx.Body)
	s.prepareBody(reqCtx, body)

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("conditional-decode: marshal: %w", err)
	}

	path := reqCtx.OriginalPath
	logger.V(logutil.DEFAULT).Info("sending request", "path", path)

	upstreamURL, err := url.Parse(s.gwClient.BaseURL() + path)
	if err != nil {
		return fmt.Errorf("conditional-decode: parse url: %w", err)
	}

	proxyReq, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL.String(), bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("conditional-decode: creating request: %w", err)
	}
	proxyReq.ContentLength = int64(len(bodyBytes))
	proxyReq.Header.Set("Content-Type", "application/json")
	proxyReq.Header.Set(reqcommon.RequestIDHeaderKey, reqCtx.RequestID)
	proxyReq.Header.Set(gateway.EPPPhaseHeader, gateway.PhaseDecode)
	proxyReq.Header.Set("Prefer", "if-available")

	var cacheMiss bool
	proxy := &httputil.ReverseProxy{
		Director:      func(_ *http.Request) {},
		FlushInterval: -1,
		Transport:     s.gwClient.Transport(),
		ModifyResponse: func(resp *http.Response) error {
			if resp.StatusCode == http.StatusPreconditionFailed {
				cacheMiss = true
				return errCacheMiss
			}
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, proxyErr error) {
			if errors.Is(proxyErr, errCacheMiss) {
				return
			}
			logger.Error(proxyErr, "proxy error")
			w.WriteHeader(http.StatusBadGateway)
		},
	}
	proxy.ServeHTTP(reqCtx.ResponseWriter, proxyReq)

	if cacheMiss {
		logger.V(logutil.DEFAULT).Info("cache miss (412), continuing pipeline")
		return nil
	}

	logger.V(logutil.DEFAULT).Info("cache hit, response forwarded")
	return pipeline.ErrPipelineDone
}

func (s *ConditionalDecodeStep) prepareBody(reqCtx *pipeline.RequestContext, body map[string]any) {
	format := s.resolveFormat(reqCtx)
	switch format {
	case gateway.FormatChatCompletions:
		if len(reqCtx.TokenIDs) > 0 {
			tokens := map[string]any{
				"token_ids": reqCtx.TokenIDs,
			}
			if len(reqCtx.MultimodalEntries) > 0 {
				allHashes := make([]string, len(reqCtx.MultimodalEntries))
				allPlaceholders := make([]any, len(reqCtx.MultimodalEntries))
				for i, entry := range reqCtx.MultimodalEntries {
					allHashes[i] = entry.Hash
					allPlaceholders[i] = map[string]any{
						"offset": entry.Placeholder.Offset,
						"length": entry.Placeholder.Length,
					}
				}
				tokens["features"] = map[string]any{
					"mm_hashes":       map[string][]string{ModalityImage: allHashes},
					"mm_placeholders": map[string][]any{ModalityImage: allPlaceholders},
				}
			}
			body["tokens"] = tokens
		}
	case gateway.FormatCompletions:
		if len(reqCtx.TokenIDs) > 0 {
			body["prompt"] = reqCtx.TokenIDs
		}
	}
}

func (s *ConditionalDecodeStep) resolveFormat(reqCtx *pipeline.RequestContext) gateway.RequestFormat {
	detected := gateway.DetectFormat(reqCtx.OriginalPath)
	if detected == gateway.FormatCompletions {
		return gateway.FormatCompletions
	}
	if !s.useOpenAIFormat {
		return gateway.FormatGenerate
	}
	return detected
}
