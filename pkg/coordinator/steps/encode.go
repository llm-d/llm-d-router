package steps

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"

	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
	"golang.org/x/sync/errgroup"
)

func init() {
	pipeline.Register("encode", NewEncodeStep)
}

type EncodeStep struct {
	gatewayPath string
	maxParallel int
	gwClient    *gateway.Client
}

func NewEncodeStep(params map[string]any) (pipeline.Step, error) {
	path := gateway.DefaultGeneratePath
	if v, ok := params["gateway_path"].(string); ok {
		path = v
	}
	maxParallel := 8
	if v, ok := params["max_parallel"].(int); ok {
		if v <= 0 {
			return nil, fmt.Errorf("max_parallel must be positive, got %d", v)
		}
		maxParallel = v
	}
	return &EncodeStep{
		gatewayPath: path,
		maxParallel: maxParallel,
	}, nil
}

func (s *EncodeStep) SetGatewayClient(c *gateway.Client) {
	s.gwClient = c
}

func (s *EncodeStep) Name() string { return "encode" }

func (s *EncodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	if len(reqCtx.MultimodalEntries) == 0 {
		return nil
	}

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxParallel)

	results := make([]map[string]any, len(reqCtx.MultimodalEntries))

	for i, entry := range reqCtx.MultimodalEntries {
		g.Go(func() error {
			tokenIDs := s.buildEncodeTokenIDs(reqCtx.TokenIDs, entry)

			body := map[string]any{
				"token_ids": tokenIDs,
				"features": map[string]any{
					"mm_hashes":       map[string][]string{"image": {entry.Hash}},
					"mm_placeholders": map[string][]any{"image": {map[string]any{"offset": 1, "length": entry.Placeholder.Length}}},
					"kwargs_data":     map[string][]string{"image": {entry.KwargsData}},
				},
				"sampling_params": map[string]any{"max_tokens": 1},
			}

			bodyBytes, err := json.Marshal(body)
			if err != nil {
				return fmt.Errorf("encode[%d]: marshal: %w", i, err)
			}

			path := fmt.Sprintf("%s%s", gateway.EncodePrefix, s.gatewayPath)
			slog.Info("encode: sending sub-request", "index", i, "path", path)

			resp, err := s.gwClient.Post(gCtx, path, bodyBytes, map[string]string{
				"X-Request-ID": reqCtx.RequestID,
			})
			if err != nil {
				return fmt.Errorf("encode[%d]: request: %w", i, err)
			}
			defer resp.Body.Close()

			if resp.StatusCode/100 != 2 {
				respBody, _ := io.ReadAll(resp.Body)
				return fmt.Errorf("encode[%d]: HTTP %d: %s", i, resp.StatusCode, string(respBody))
			}

			var encResp encodeResponse
			if err := json.NewDecoder(resp.Body).Decode(&encResp); err != nil {
				return fmt.Errorf("encode[%d]: decode response: %w", i, err)
			}

			results[i] = encResp.ECTransferParams
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	reqCtx.ECTransferParams = results

	slog.Info("encode: all sub-requests complete", "count", len(results))
	return nil
}

func (s *EncodeStep) buildEncodeTokenIDs(fullTokenIDs []int, entry pipeline.MultimodalEntry) []int {
	bos := 1
	placeholderTokenID := 0
	if len(fullTokenIDs) > 0 {
		bos = fullTokenIDs[0]
		if entry.Placeholder.Offset < len(fullTokenIDs) {
			placeholderTokenID = fullTokenIDs[entry.Placeholder.Offset]
		}
	}

	tokenIDs := make([]int, 1+entry.Placeholder.Length)
	tokenIDs[0] = bos
	for j := 1; j <= entry.Placeholder.Length; j++ {
		tokenIDs[j] = placeholderTokenID
	}
	return tokenIDs
}

type encodeResponse struct {
	ECTransferParams map[string]any `json:"ec_transfer_params"`
}
