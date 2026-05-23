package steps

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/coordinator/pkg/connectors/kv"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

const DecodeStepName = "decode"

func init() {
	pipeline.Register(DecodeStepName, NewDecodeStep)
}

type DecodeStep struct {
	gatewayPath string
	gwClient    *gateway.Client
	kv          kv.Connector
}

func NewDecodeStep(params map[string]any) (pipeline.Step, error) {
	path := gateway.DefaultGeneratePath
	if v, ok := params[ParamGatewayPath].(string); ok {
		path = v
	}
	kvName, _ := params[ParamKVConnector].(string)
	kvConn, err := kv.Build(kvName)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &DecodeStep{gatewayPath: path, kv: kvConn}, nil
}

// SetGatewayClient injects the shared gateway client.
func (s *DecodeStep) SetGatewayClient(c *gateway.Client) {
	s.gwClient = c
}

func (s *DecodeStep) Name() string { return DecodeStepName }

func (s *DecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName("decode")
	reqCtx.Body["kv_transfer_params"] = s.kv.PrepareDecodeKVParams(reqCtx)
	s.injectUUIDs(reqCtx)

	bodyBytes, err := json.Marshal(reqCtx.Body)
	if err != nil {
		return fmt.Errorf("decode: marshal: %w", err)
	}

	path := fmt.Sprintf("%s%s", gateway.DecodePrefix, reqCtx.OriginalPath)
	logger.V(logutil.DEFAULT).Info("sending request", "path", path, "stream", reqCtx.Stream)

	upstreamURL, err := url.Parse(s.gwClient.BaseURL() + path)
	if err != nil {
		return fmt.Errorf("decode: parse url: %w", err)
	}

	proxyReq, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL.String(), bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("decode: creating request: %w", err)
	}
	proxyReq.ContentLength = int64(len(bodyBytes))
	proxyReq.Header.Set("Content-Type", "application/json")
	proxyReq.Header.Set(reqcommon.RequestIDHeaderKey, reqCtx.RequestID)

	proxy := &httputil.ReverseProxy{
		Director:      func(_ *http.Request) {},
		FlushInterval: -1,
		Transport:     s.gwClient.Transport(),
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, proxyErr error) {
			logger.Error(proxyErr, "proxy error")
			w.WriteHeader(http.StatusBadGateway)
		},
	}
	proxy.ServeHTTP(reqCtx.ResponseWriter, proxyReq)
	return nil
}

func (s *DecodeStep) injectUUIDs(reqCtx *pipeline.RequestContext) {
	messages, ok := reqCtx.Body["messages"].([]any)
	if !ok {
		return
	}

	hashIdx := 0
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		content, ok := msgMap["content"].([]any)
		if !ok {
			continue
		}
		for _, part := range content {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}
			if partMap["type"] != "image_url" {
				continue
			}
			if hashIdx < len(reqCtx.MultimodalEntries) {
				partMap["uuid"] = reqCtx.MultimodalEntries[hashIdx].Hash
				hashIdx++
			}
		}
	}
}
