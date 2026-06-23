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

	"github.com/go-logr/logr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/coordinator/pkg/common/httplog"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

// errCacheMiss signals that the conditional-decode cache probe returned 412.
// It flows from the proxy's ModifyResponse to its ErrorHandler, which swallows
// it so the miss can fall through to the rest of the pipeline.
var errCacheMiss = errors.New("cache miss")

// newDecodeProxyRequest builds the decode-phase POST to the gateway: it marshals
// body, targets gwClient.BaseURL()+reqCtx.OriginalPath, and stamps the JSON
// content-type, forwarded headers, request id, and decode phase header. step
// names the caller for error wrapping; extraHeaders carries step-specific
// headers (the conditional cache probe sets Prefer).
func newDecodeProxyRequest(ctx context.Context, logger logr.Logger, step string, reqCtx *pipeline.RequestContext, gwClient *gateway.Client, body map[string]any, extraHeaders map[string]string) (*http.Request, error) {
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("%s: marshal: %w", step, err)
	}

	upstreamURL, err := url.Parse(gwClient.BaseURL() + reqCtx.OriginalPath)
	if err != nil {
		return nil, fmt.Errorf("%s: parse url: %w", step, err)
	}

	proxyReq, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL.String(), bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("%s: creating request: %w", step, err)
	}
	proxyReq.ContentLength = int64(len(bodyBytes))
	proxyReq.Header.Set(gateway.ContentTypeHeader, gateway.ContentTypeJSON)
	for k, v := range reqCtx.ForwardedHeaders() {
		proxyReq.Header.Set(k, v)
	}
	proxyReq.Header.Set(reqcommon.RequestIDHeaderKey, reqCtx.RequestID)
	proxyReq.Header.Set(gateway.EPPPhaseHeader, gateway.PhaseDecode)
	for k, v := range extraHeaders {
		proxyReq.Header.Set(k, v)
	}

	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info("request body", "method", "POST", "path", reqCtx.OriginalPath, "bodyLen", len(bodyBytes), "headers", httplog.RedactedHeaders(proxyReq.Header))
	}

	return proxyReq, nil
}

// newDecodeProxy builds the streaming reverse proxy for a decode-phase request.
// modifyResponse, when non-nil, inspects each upstream response (the conditional
// cache probe uses it to detect a 412). Transport errors are logged and answered
// 502, except errCacheMiss, which is swallowed so the miss falls through.
func newDecodeProxy(logger logr.Logger, transport http.RoundTripper, modifyResponse func(*http.Response) error) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Director:       func(_ *http.Request) {},
		FlushInterval:  -1,
		Transport:      transport,
		ModifyResponse: modifyResponse,
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, proxyErr error) {
			if errors.Is(proxyErr, errCacheMiss) {
				return
			}
			logger.Error(proxyErr, "proxy error")
			w.WriteHeader(http.StatusBadGateway)
		},
	}
}
