package steps

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-inference-scheduler/pkg/common/request"

	"github.com/llm-d/coordinator/pkg/connectors/ec"
	"github.com/llm-d/coordinator/pkg/connectors/kv"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

const PrefillStepName = "prefill"

func init() {
	pipeline.Register(PrefillStepName, NewPrefillStep)
}

type PrefillStep struct {
	gatewayPath string
	gwClient    *gateway.Client
	kv          kv.Connector
	ec          ec.Connector
}

func NewPrefillStep(params map[string]any) (pipeline.Step, error) {
	path := gateway.DefaultGeneratePath
	if v, ok := params[ParamGatewayPath].(string); ok {
		path = v
	}
	kvName, _ := params[ParamKVConnector].(string)
	kvConn, err := kv.Build(kvName)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	ecName, _ := params[ParamECConnector].(string)
	ecConn, err := ec.Build(ecName)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	return &PrefillStep{gatewayPath: path, kv: kvConn, ec: ecConn}, nil
}

func (s *PrefillStep) SetGatewayClient(c *gateway.Client) {
	s.gwClient = c
}

func (s *PrefillStep) Name() string { return PrefillStepName }

func (s *PrefillStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName("prefill")

	allHashes := make([]string, len(reqCtx.MultimodalEntries))
	allPlaceholders := make([]any, len(reqCtx.MultimodalEntries))
	allKwargsData := make([]string, len(reqCtx.MultimodalEntries))
	for i, entry := range reqCtx.MultimodalEntries {
		allHashes[i] = entry.Hash
		allPlaceholders[i] = map[string]any{
			"offset": entry.Placeholder.Offset,
			"length": entry.Placeholder.Length,
		}
		allKwargsData[i] = entry.KwargsData
	}

	var features any
	if len(reqCtx.MultimodalEntries) > 0 {
		features = map[string]any{
			"mm_hashes":       map[string][]string{"image": allHashes},
			"mm_placeholders": map[string][]any{"image": allPlaceholders},
			"kwargs_data":     map[string][]string{"image": allKwargsData},
		}
	}

	body := map[string]any{
		"request_id":         reqCtx.RequestID,
		"token_ids":          reqCtx.TokenIDs,
		"model":              reqCtx.Model,
		"sampling_params":    map[string]any{"max_tokens": 1},
		"kv_transfer_params": s.kv.PreparePrefillKVParams(reqCtx),
	}

	if features != nil {
		body["features"] = features
	}
	if ecParams := s.ec.PreparePrefillECParams(reqCtx); len(ecParams) > 0 {
		body["ec_transfer_params"] = ecParams
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("prefill: marshal: %w", err)
	}

	path := fmt.Sprintf("%s%s", gateway.PrefillPrefix, s.gatewayPath)
	logger.V(logutil.DEFAULT).Info("sending request", "path", path)

	resp, err := s.gwClient.Post(ctx, path, bodyBytes, map[string]string{
		reqcommon.RequestIDHeaderKey: reqCtx.RequestID,
	})
	if err != nil {
		return fmt.Errorf("prefill: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("prefill: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var prefillResp prefillResponse
	if err := json.NewDecoder(resp.Body).Decode(&prefillResp); err != nil {
		return fmt.Errorf("prefill: decode response: %w", err)
	}

	reqCtx.KVTransferParams = prefillResp.KVTransferParams

	logger.V(logutil.DEFAULT).Info("complete")
	return nil
}

type prefillResponse struct {
	KVTransferParams map[string]any `json:"kv_transfer_params"`
}
