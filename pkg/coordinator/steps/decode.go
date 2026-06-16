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

	"github.com/llm-d/coordinator/pkg/common/httplog"
	"github.com/llm-d/coordinator/pkg/connectors/kv"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

const DecodeStepName = "decode"

func init() {
	pipeline.Register(DecodeStepName, NewDecodeStep)
}

type DecodeStep struct {
	useOpenAIFormat bool
	gwClient        *gateway.Client
	kv              kv.Connector
}

func NewDecodeStep(params map[string]any) (pipeline.Step, error) {
	useOpenAI := true
	if v, ok := params["use_openai_format"].(bool); ok {
		useOpenAI = v
	}
	kvName, _ := params[ParamKVConnector].(string)
	kvConn, err := kv.Build(kvName)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &DecodeStep{useOpenAIFormat: useOpenAI, kv: kvConn}, nil
}

func (s *DecodeStep) SetGatewayClient(c *gateway.Client) {
	s.gwClient = c
}

func (s *DecodeStep) Name() string { return DecodeStepName }

func (s *DecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(DecodeStepName)

	s.prepareDecodeBody(ctx, reqCtx)

	bodyBytes, err := json.Marshal(reqCtx.Body)
	if err != nil {
		return fmt.Errorf("decode: marshal: %w", err)
	}

	path := reqCtx.OriginalPath
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
	proxyReq.Header.Set(gateway.ContentTypeHeader, gateway.ContentTypeJSON)
	for k, v := range reqCtx.ForwardedHeaders() {
		proxyReq.Header.Set(k, v)
	}
	proxyReq.Header.Set(reqcommon.RequestIDHeaderKey, reqCtx.RequestID)
	proxyReq.Header.Set(gateway.EPPPhaseHeader, gateway.PhaseDecode)

	logger.V(logutil.DEBUG).Info("request body", "method", "POST", "path", path, "bodyLen", len(bodyBytes), "headers", httplog.RedactedHeaders(proxyReq.Header))

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

func (s *DecodeStep) prepareDecodeBody(ctx context.Context, reqCtx *pipeline.RequestContext) {
	reqCtx.Body["kv_transfer_params"] = s.kv.PrepareDecodeKVParams(ctx, reqCtx)
	s.injectUUIDs(reqCtx)

	format := s.resolveFormat(reqCtx)
	switch format {
	case gateway.FormatChatCompletions:
		s.injectTokensField(reqCtx)
	case gateway.FormatCompletions:
		if len(reqCtx.TokenIDs) > 0 {
			reqCtx.Body["prompt"] = reqCtx.TokenIDs
		}
	}
}

func (s *DecodeStep) resolveFormat(reqCtx *pipeline.RequestContext) gateway.RequestFormat {
	detected := gateway.DetectFormat(reqCtx.OriginalPath)
	if detected == gateway.FormatCompletions {
		return gateway.FormatCompletions
	}
	if !s.useOpenAIFormat {
		return gateway.FormatGenerate
	}
	return detected
}

func (s *DecodeStep) injectTokensField(reqCtx *pipeline.RequestContext) {
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
	reqCtx.Body["tokens"] = tokens
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
