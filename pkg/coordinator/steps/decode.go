package steps

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-inference-scheduler/pkg/common/request"

	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
	"github.com/llm-d/coordinator/pkg/server"
)

const DecodeStepName = "decode"

func init() {
	pipeline.Register(DecodeStepName, NewDecodeStep)
}

type DecodeStep struct {
	gatewayPath string
	gwClient    *gateway.Client
}

func NewDecodeStep(params map[string]any) (pipeline.Step, error) {
	path := gateway.DefaultGeneratePath
	if v, ok := params["gateway_path"].(string); ok {
		path = v
	}
	return &DecodeStep{gatewayPath: path}, nil
}

// SetGatewayClient injects the shared gateway client.
func (s *DecodeStep) SetGatewayClient(c *gateway.Client) {
	s.gwClient = c
}

func (s *DecodeStep) Name() string { return DecodeStepName }

func (s *DecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName("decode")

	kvParams := reqCtx.KVTransferParams
	kvParams["do_remote_prefill"] = true
	reqCtx.Body["kv_transfer_params"] = kvParams
	s.injectUUIDs(reqCtx)

	bodyBytes, err := json.Marshal(reqCtx.Body)
	if err != nil {
		return fmt.Errorf("decode: marshal: %w", err)
	}

	path := fmt.Sprintf("%s%s", gateway.DecodePrefix, reqCtx.OriginalPath)
	logger.V(logutil.DEFAULT).Info("sending request", "path", path, "stream", reqCtx.Stream)

	resp, err := s.gwClient.Post(ctx, path, bodyBytes, map[string]string{
		reqcommon.RequestIDHeaderKey: reqCtx.RequestID,
	})
	if err != nil {
		return fmt.Errorf("decode: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("decode: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	if reqCtx.Stream {
		return s.streamResponse(reqCtx, resp)
	}
	return s.bufferResponse(reqCtx, resp)
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
			if partMap["type"] != "image_url" && partMap["image_url"] == nil {
				continue
			}
			if hashIdx < len(reqCtx.MultimodalEntries) {
				partMap["uuid"] = reqCtx.MultimodalEntries[hashIdx].Hash
				partMap["image_url"] = nil
				hashIdx++
			}
		}
	}
}

func (s *DecodeStep) streamResponse(reqCtx *pipeline.RequestContext, resp *http.Response) error {
	server.SetSSEHeaders(reqCtx.ResponseWriter)
	reqCtx.ResponseWriter.WriteHeader(http.StatusOK)
	reqCtx.Flusher.Flush()

	return server.StreamSSE(reqCtx.ResponseWriter, reqCtx.Flusher, resp.Body)
}

func (s *DecodeStep) bufferResponse(reqCtx *pipeline.RequestContext, resp *http.Response) error {
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("decode: reading response: %w", err)
	}

	reqCtx.ResponseWriter.Header().Set("Content-Type", "application/json")
	reqCtx.ResponseWriter.WriteHeader(http.StatusOK)
	_, err = reqCtx.ResponseWriter.Write(respBody)
	return err
}
