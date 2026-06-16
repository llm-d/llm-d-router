package steps

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/coordinator/pkg/common/httplog"
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
	useOpenAIFormat bool
	gwClient        *gateway.Client
	kv              kv.Connector
	ec              ec.Connector
}

func NewPrefillStep(params map[string]any) (pipeline.Step, error) {
	useOpenAI := true
	if v, ok := params["use_openai_format"].(bool); ok {
		useOpenAI = v
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
	return &PrefillStep{useOpenAIFormat: useOpenAI, kv: kvConn, ec: ecConn}, nil
}

func (s *PrefillStep) SetGatewayClient(c *gateway.Client) {
	s.gwClient = c
}

func (s *PrefillStep) Name() string { return PrefillStepName }

func (s *PrefillStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(PrefillStepName)

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

	var features map[string]any
	if len(reqCtx.MultimodalEntries) > 0 {
		features = map[string]any{
			"mm_hashes":       map[string][]string{ModalityImage: allHashes},
			"mm_placeholders": map[string][]any{ModalityImage: allPlaceholders},
			"kwargs_data":     map[string][]string{ModalityImage: allKwargsData},
		}
	}

	format := s.resolveFormat(reqCtx)
	body, err := s.buildPrefillBody(ctx, reqCtx, features, format)
	if err != nil {
		return fmt.Errorf("prefill: %w", err)
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("prefill: marshal: %w", err)
	}

	path := gateway.PathForFormat(format)
	logger.V(logutil.DEFAULT).Info("sending request", "path", path)

	headers := reqCtx.ForwardedHeaders()
	headers[reqcommon.RequestIDHeaderKey] = reqCtx.RequestID
	headers[gateway.EPPPhaseHeader] = gateway.PhasePrefill

	logger.V(logutil.DEBUG).Info("request body", "method", "POST", "path", path, "bodyLen", len(bodyBytes), "headers", httplog.RedactedHeaders(headers))

	resp, err := s.gwClient.Post(ctx, path, bodyBytes, headers)
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

	reqCtx.KVTransferParams = kvParamsFromResponse(logger, prefillResp.KVTransferParams)

	logger.V(logutil.DEFAULT).Info("complete")
	return nil
}

func (s *PrefillStep) resolveFormat(reqCtx *pipeline.RequestContext) gateway.RequestFormat {
	detected := gateway.DetectFormat(reqCtx.OriginalPath)
	if detected == gateway.FormatCompletions {
		return gateway.FormatCompletions
	}
	if !s.useOpenAIFormat {
		return gateway.FormatGenerate
	}
	return detected
}

func (s *PrefillStep) buildPrefillBody(ctx context.Context, reqCtx *pipeline.RequestContext, features map[string]any, format gateway.RequestFormat) (map[string]any, error) {
	ecParams, err := s.ec.PreparePrefillECParams(ctx, reqCtx)
	if err != nil {
		return nil, err
	}
	kvParams := s.kv.PreparePrefillKVParams(ctx, reqCtx)

	switch format {
	case gateway.FormatChatCompletions:
		body := copyBody(reqCtx.Body)
		tokens := map[string]any{
			"token_ids": reqCtx.TokenIDs,
		}
		if features != nil {
			tokensFeatures := map[string]any{
				"mm_hashes":       features["mm_hashes"],
				"mm_placeholders": features["mm_placeholders"],
			}
			tokens["features"] = tokensFeatures
		}
		body["tokens"] = tokens
		body["max_tokens"] = 1
		body["kv_transfer_params"] = kvParams
		if len(ecParams) > 0 {
			body["ec_transfer_params"] = ecParams
		}
		return body, nil

	case gateway.FormatCompletions:
		prompt := reqCtx.Body["prompt"]
		if len(reqCtx.TokenIDs) > 0 {
			prompt = reqCtx.TokenIDs
		}
		body := map[string]any{
			"request_id":         reqCtx.RequestID,
			"model":              reqCtx.Model,
			"prompt":             prompt,
			"max_tokens":         1,
			"kv_transfer_params": kvParams,
		}
		if features != nil {
			body["features"] = features
		}
		if len(ecParams) > 0 {
			body["ec_transfer_params"] = ecParams
		}
		return body, nil

	default:
		body := map[string]any{
			"request_id": reqCtx.RequestID,
			"token_ids":  reqCtx.TokenIDs,
			"model":      reqCtx.Model,
			"sampling_params": map[string]any{
				"max_tokens": 1,
				"extra_args": map[string]any{
					"kv_transfer_params": kvParams,
				},
			},
		}
		if features != nil {
			body["features"] = features
		}
		if len(ecParams) > 0 {
			body["ec_transfer_params"] = ecParams
		}
		return body, nil
	}
}

type prefillResponse struct {
	// KVTransferParams is decoded as any (not map[string]any) so a non-object
	// value does not fail the decode; kvParamsFromResponse coerces it.
	KVTransferParams any `json:"kv_transfer_params"`
}

// kvParamsFromResponse coerces the kv_transfer_params value from a prefill
// response to a map, mirroring ecParamsFromResponse: a non-object value is
// logged at debug and skipped (returns nil) rather than failing the request.
// A missing or null value is already nil; an empty map passes through so the
// connector's own no-metadata handling applies.
func kvParamsFromResponse(logger logr.Logger, v any) map[string]any {
	switch m := v.(type) {
	case nil:
		return nil
	case map[string]any:
		return m
	default:
		logger.V(logutil.DEBUG).Info("kv_transfer_params is not a JSON object; skipping",
			"type", fmt.Sprintf("%T", v))
		return nil
	}
}
