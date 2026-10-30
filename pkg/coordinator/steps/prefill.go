/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package steps

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"mime"
	"net"
	"net/http"
	"time"

	"github.com/openai/openai-go/packages/ssestream"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/common/usage"

	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
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
	prefillGrace    time.Duration
}

func NewPrefillStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("prefill: gateway client is required")
	}
	useOpenAI, err := parseUseOpenAIFormat(params)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	kvConn, err := buildKVConnector(params)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	ecConn, err := buildECConnector(params)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	grace, present, err := paramDuration(params, "prefill_grace")
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	if !present {
		grace = 30 * time.Second
	}
	if grace <= 0 {
		return nil, errors.New("prefill: prefill_grace must be positive")
	}
	return &PrefillStep{
		useOpenAIFormat: useOpenAI,
		gwClient:        gwClient,
		kv:              kvConn,
		ec:              ecConn,
		prefillGrace:    grace,
	}, nil
}

func (s *PrefillStep) Name() string { return PrefillStepName }

func (s *PrefillStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(PrefillStepName)

	format := resolveFormat(s.useOpenAIFormat, reqCtx.OriginalPath)
	if reqCtx.OriginalPath == reqcommon.PathSGLangGenerate {
		format = reqcommon.APITypeSGLangGenerate
	}
	body, err := s.buildPrefillBody(ctx, reqCtx, format)
	if err != nil {
		return fmt.Errorf("prefill: %w", err)
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("prefill: marshal: %w", err)
	}

	path := format.Path()
	logger.V(logutil.DEFAULT).Info("sending request", "path", path)

	prefillCtx := ctx
	var drain *pipeline.PrefillDrain
	handedOff := false
	if format == reqcommon.APITypeSGLangGenerate {
		if reqCtx.PrefillDrain != nil {
			return errors.New("prefill: a prefill stream is already running")
		}
		var cancel context.CancelFunc
		prefillCtx, cancel = context.WithCancel(context.WithoutCancel(ctx))
		stop := context.AfterFunc(ctx, cancel)
		drain = pipeline.NewPrefillDrain(
			func() {
				stop()
				go func() {
					defer cancel()
					timer := time.NewTimer(s.prefillGrace)
					defer timer.Stop()
					select {
					case <-drain.Done():
					case <-timer.C:
						logger.Info("prefill grace expired", "grace", s.prefillGrace)
					}
				}()
			},
			func() {
				stop()
				cancel()
			},
		)
		defer func() {
			if !handedOff {
				drain.Cancel()
			}
		}()
	}
	resp, err := postToGateway(prefillCtx, logger, s.gwClient, gatewayRequest{
		logMsg:   "request body",
		step:     PrefillStepName,
		upstream: coordmetrics.UpstreamPrefill,
		path:     path,
		body:     bodyBytes,
		headers:  gatewayHeaders(reqCtx, gateway.PhasePrefill),
	})
	if err != nil {
		return err
	}
	defer func() {
		if !handedOff {
			_ = resp.Body.Close()
		}
	}()

	reqCtx.PeerTopology = resp.Header.Get(reqcommon.PeerTopologyHeaderKey)
	reqCtx.CaptureResponseHeaders(resp.Header)
	if drain != nil {
		mediaType, _, mediaErr := mime.ParseMediaType(resp.Header.Get(reqcommon.HeaderContentType))
		if mediaErr != nil || mediaType != reqcommon.MediaTypeEventStream {
			return errors.New("prefill: streaming SGLang requires an event stream")
		}
		host, _, endpointErr := net.SplitHostPort(resp.Header.Get(routing.PrefillEndpointHeader))
		if endpointErr != nil || host == "" {
			return errors.New("prefill: streaming SGLang requires a valid selected endpoint response header")
		}
		reqCtx.KVTransferParams[reqcommon.FieldBootstrapHost] = host
		go func() {
			defer resp.Body.Close()
			err := drainSGLangPrefill(resp)
			if err != nil && !errors.Is(err, context.Canceled) {
				logger.Error(err, "prefill stream failed")
			}
			drain.Complete(err)
		}()
		reqCtx.PrefillDrain = drain
		handedOff = true
		return nil
	}

	var prefillResp prefillResponse
	if err := json.NewDecoder(resp.Body).Decode(&prefillResp); err != nil {
		return fmt.Errorf("prefill: decode response: %w", err)
	}

	reqCtx.KVTransferParams = coerceParamsMap(logger, prefillResp.KVTransferParams, "kv_transfer_params")
	// kv-nixl always requests a remote-decode handoff, so missing
	// kv_transfer_params means decode will recompute the whole prompt. Older
	// engines (< v0.29.0) silently drop top-level transfer params on this route,
	// which is the most common cause. Empty params are not a failure for the
	// other connectors: kv-shared-storage carries no wire fields at all.
	decodeRecomputesPrompt := len(reqCtx.KVTransferParams) == 0 && s.kv.Name() == kv.NIXL
	if decodeRecomputesPrompt {
		logger.Info("prefill returned no kv_transfer_params; decode will recompute the prompt",
			"kvConnector", s.kv.Name(), "path", path)
	}
	// Only a prefill response that measured the prompt cache, for a request
	// whose blocks decode will reuse, can correct the count decode reports; see
	// RequestContext.PrefillCachedTokens for what the two states mean.
	if prefillResp.Usage != nil && !decodeRecomputesPrompt {
		cachedTokens, _ := usage.CachedTokens(prefillResp.Usage)
		reqCtx.PrefillCachedTokens = &cachedTokens
	}

	logger.V(logutil.DEFAULT).Info("complete")
	return nil
}

func drainSGLangPrefill(resp *http.Response) error {
	stream := ssestream.NewDecoder(resp)
	terminal := false
	for stream.Next() {
		data := bytes.TrimSpace(stream.Event().Data)
		if len(data) == 0 || bytes.Equal(data, []byte(reqcommon.SSEDoneMarker)) {
			continue
		}
		var event struct {
			Error    *json.RawMessage `json:"error"`
			MetaInfo struct {
				FinishReason struct {
					Type       string `json:"type"`
					StatusCode *int   `json:"status_code"`
					Message    string `json:"message"`
				} `json:"finish_reason"`
			} `json:"meta_info"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode SGLang prefill event: %w", err)
		}
		if event.Error != nil {
			return fmt.Errorf("SGLang prefill error: %s", *event.Error)
		}
		finish := event.MetaInfo.FinishReason
		// A null status_code indicates an abort without an initiator-supplied reason.
		if finish.Type == "abort" && finish.StatusCode != nil {
			return fmt.Errorf("SGLang prefill aborted (%d): %s", *finish.StatusCode, finish.Message)
		}
		terminal = terminal || finish.Type != ""
	}
	if err := stream.Err(); err != nil {
		return err
	}
	if !terminal {
		return errors.New("SGLang prefill stream closed without a terminal event")
	}
	return nil
}

func (s *PrefillStep) buildPrefillBody(ctx context.Context, reqCtx *pipeline.RequestContext, format reqcommon.APIType) (map[string]any, error) {
	if format == reqcommon.APITypeSGLangGenerate {
		if s.kv.Name() != kv.SGLang || !reqCtx.Stream {
			return nil, fmt.Errorf("/generate prefill requires kv-sglang and stream=true: %w", pipeline.ErrBadRequest)
		}
		if reqCtx.Body[reqcommon.FieldDisaggPrefillDPRank] != nil {
			return nil, fmt.Errorf("client-supplied disagg_prefill_dp_rank is unsupported: %w", pipeline.ErrBadRequest)
		}
		if sampling, present := reqCtx.Body[reqcommon.FieldSamplingParams]; present {
			params, ok := sampling.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("SGLang prefill requires one sampling_params object: %w", pipeline.ErrBadRequest)
			}
			n, present, err := paramInt(params, reqcommon.FieldN)
			if err != nil || (present && n != 1) {
				return nil, fmt.Errorf("SGLang prefill requires sampling_params.n=1: %w", pipeline.ErrBadRequest)
			}
		}
		body := maps.Clone(reqCtx.Body)
		// Prefill registers the room locally; decode uses the selected endpoint address.
		body[reqcommon.FieldBootstrapHost] = "127.0.0.1"
		reqCtx.KVTransferParams = s.kv.PreparePrefillKVParams(ctx, reqCtx)
		maps.Copy(body, reqCtx.KVTransferParams)
		return body, nil
	}
	ecParams, err := s.ec.PreparePrefillECParams(ctx, reqCtx)
	if err != nil {
		return nil, err
	}
	kvParams := s.kv.PreparePrefillKVParams(ctx, reqCtx)

	switch format {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses, reqcommon.APITypeMessages:
		body := maps.Clone(reqCtx.Body)
		reqcommon.CapSingleToken(body, format)
		body[reqcommon.FieldKVTransferParams] = kvParams
		if len(ecParams) > 0 {
			body[reqcommon.FieldECTransferParams] = ecParams
		}
		return body, nil

	case reqcommon.APITypeCompletions:
		prompt := reqCtx.Body["prompt"]
		if len(reqCtx.TokenIDs) > 0 {
			prompt = reqCtx.TokenIDs
		}
		body := map[string]any{
			"request_id":                    reqCtx.RequestID,
			"model":                         reqCtx.Model,
			"prompt":                        prompt,
			reqcommon.FieldKVTransferParams: kvParams,
		}
		reqcommon.CapSingleToken(body, format)
		// Completions never carries multimodal entries (see render.executeCompletions).
		if len(ecParams) > 0 {
			body[reqcommon.FieldECTransferParams] = ecParams
		}
		return body, nil

	case reqcommon.APITypeVLLMGenerate:
		body := map[string]any{
			"request_id":                    reqCtx.RequestID,
			"token_ids":                     reqCtx.TokenIDs,
			"model":                         reqCtx.Model,
			reqcommon.FieldKVTransferParams: kvParams,
		}
		reqcommon.CapSingleToken(body, format)
		// The only consumer of the per-entry mm_metadata/kwargs_data choice;
		// see buildPrefillMMFeatures. Built inside this branch rather than
		// above the switch so chat-completions requests never build-and-drop
		// a features map they do not ship.
		if features := buildPrefillMMFeatures(reqCtx.MultimodalEntries, ecParams); features != nil {
			body["features"] = features
		}
		if len(ecParams) > 0 {
			body[reqcommon.FieldECTransferParams] = ecParams
		}
		return body, nil

	default:
		// resolveFormat above never returns anything but the formats handled
		// here, so this case is unreachable unless resolveFormat's contract
		// changes.
		return nil, unreachableFormatError(format)
	}
}

type prefillResponse struct {
	// KVTransferParams is decoded as any (not map[string]any) so a non-object
	// value does not fail the decode; coerceParamsMap coerces it.
	KVTransferParams any `json:"kv_transfer_params"`
	// Usage is decoded as any for the same reason, and is read by
	// usage.CachedTokens.
	Usage any `json:"usage"`
}
