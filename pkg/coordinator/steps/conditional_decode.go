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
	gwClient *gateway.Client
}

func NewConditionalDecodeStep(_ map[string]any) (pipeline.Step, error) {
	return &ConditionalDecodeStep{}, nil
}

func (s *ConditionalDecodeStep) SetGatewayClient(c *gateway.Client) {
	s.gwClient = c
}

func (s *ConditionalDecodeStep) Name() string { return ConditionalDecodeStepName }

func (s *ConditionalDecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName("conditional-decode")

	bodyBytes, err := json.Marshal(reqCtx.Body)
	if err != nil {
		return fmt.Errorf("conditional-decode: marshal: %w", err)
	}

	path := fmt.Sprintf("%s%s", gateway.DecodePrefix, reqCtx.OriginalPath)
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
