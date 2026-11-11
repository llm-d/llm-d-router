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
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const ConditionalDecodeStepName = "conditional-decode"

func init() {
	pipeline.Register(ConditionalDecodeStepName, NewConditionalDecodeStep)
}

type ConditionalDecodeStep struct {
	gwClient          *gateway.Client
	promptTokenParity string
	forwardTokenIDs   bool
	// budget is set when the pipeline enables force-streaming.
	budget *forceStreamBudget
}

func NewConditionalDecodeStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("conditional-decode: gateway client is required")
	}
	if err := rejectUseOpenAIFormatOverride(ConditionalDecodeStepName, params); err != nil {
		return nil, err
	}
	if err := rejectForceStreamParams(ConditionalDecodeStepName, params); err != nil {
		return nil, err
	}
	parity, err := parsePromptTokenParity(params)
	if err != nil {
		return nil, fmt.Errorf("conditional-decode: %w", err)
	}
	forwardTokenIDs, _, err := paramBool(params, ParamForwardPromptTokenIDs)
	if err != nil {
		return nil, fmt.Errorf("conditional-decode: %w", err)
	}
	return &ConditionalDecodeStep{gwClient: gwClient, promptTokenParity: parity, forwardTokenIDs: forwardTokenIDs}, nil
}

func (s *ConditionalDecodeStep) Name() string { return ConditionalDecodeStepName }

func (s *ConditionalDecodeStep) enableForceStream(budget *forceStreamBudget) { s.budget = budget }

func (s *ConditionalDecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	if reqCtx.IsSGLangGRPC() {
		return nil
	}
	logger := log.FromContext(ctx).WithName(ConditionalDecodeStepName)

	body, err := s.prepareBody(reqCtx)
	if err != nil {
		return err
	}

	// A served hit is the KV-reuse case the parity check exists for; body is a
	// clone, so an injected include_usage does not reach the decode step on a
	// miss. On a 412 nothing is written, so the writer stays idle.
	parity, injectedUsage := setupParity(ConditionalDecodeStepName, s.promptTokenParity, logger, reqCtx, body)

	probeHeaders := map[string]string{routing.PreferHeader: routing.PreferIfAvailable}
	var cacheMiss bool
	onProbeResponse := func(resp *http.Response) error {
		switch {
		case resp.StatusCode == http.StatusPreconditionFailed:
			cacheMiss = true
			reqCtx.CaptureResponseHeaders(resp.Header)
			coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultDeferred)
			return errCacheMiss
		case resp.StatusCode >= http.StatusBadRequest:
			// Worker error (any 4xx/5xx except 412): the response is still
			// streamed to the client, but the outcome is not a served hit.
			coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultError)
		default:
			coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultServed)
		}
		return nil
	}

	// A non-streaming request the worker serves is force-streamed as decode
	// would serve it. A decline or a fallback re-sends body through the probe
	// below; the probe body names no KV handoff, so the re-send is safe. Its
	// outcome is recorded only when the probe is not deferred: a deferred
	// request is decode's to record.
	var forceStreamResult string
	if s.budget != nil && !reqCtx.Stream {
		if shape, ok := shapeForAPIType(reqcommon.DetectAPIType(reqCtx.OriginalPath)); ok {
			// The probe result of a forced attempt is recorded once its outcome is
			// known: a fallback re-sends the probe, which records its own.
			var forcedStatus int
			leg := forceStreamLeg{
				step:     ConditionalDecodeStepName,
				upstream: coordmetrics.UpstreamConditionalDecode,
				headers:  probeHeaders,
				onResponse: func(resp *http.Response) error {
					forcedStatus = resp.StatusCode
					if resp.StatusCode == http.StatusPreconditionFailed {
						reqCtx.CaptureResponseHeaders(resp.Header)
						coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultDeferred)
						return errCacheMiss
					}
					return nil
				},
			}
			result, _, err := tryForceStream(ctx, logger, s.gwClient, s.budget, leg, reqCtx, body, shape, parity)
			if errors.Is(err, errCacheMiss) {
				logger.V(logutil.DEFAULT).Info("probe deferred (412), continuing pipeline")
				return nil
			}
			if result == "" {
				if probe := forcedProbeResult(forcedStatus, err); probe != "" {
					coordmetrics.IncConditionalDecodeProbes(probe)
				}
				if err != nil {
					return err
				}
				logger.V(logutil.DEFAULT).Info("probe served, response forwarded")
				return pipeline.ErrPipelineDone
			}
			forceStreamResult = result
		}
	}

	logger.V(logutil.DEFAULT).Info("sending request", "path", reqCtx.OriginalPath)

	proxyReq, err := newDecodeProxyRequest(ctx, logger, ConditionalDecodeStepName, reqCtx, s.gwClient, body, probeHeaders)
	if err != nil {
		return err
	}

	writer, finalize := parityResponseWriter(reqCtx, parity, injectedUsage)
	out := serveDecode(logger, s.gwClient.Transport(), writer, proxyReq, coordmetrics.UpstreamConditionalDecode, onProbeResponse)

	if err := finalize(); err != nil {
		logger.Error(err, "flushing the rewritten usage block failed: client received a partial response")
	}

	if cacheMiss {
		logger.V(logutil.DEFAULT).Info("probe deferred (412), continuing pipeline")
		return nil
	}
	if forceStreamResult != "" && out.Cancelled == nil {
		coordmetrics.IncForceStreamTotal(reqCtx.Model, ConditionalDecodeStepName, forceStreamResult)
	}
	if out.TransportErr != nil {
		coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultTransportError)
	}
	if err := out.streamedError(ConditionalDecodeStepName); err != nil {
		return err
	}

	logger.V(logutil.DEFAULT).Info("probe served, response forwarded")
	return pipeline.ErrPipelineDone
}

// forcedProbeResult is the probe result of a forced attempt that ended the
// request, given the status the worker answered (0 when none arrived). It is the
// worker's answer, as the pass-through records it, except that an error frame
// streamed after the 200 is the worker error it reports. A client that left
// before the worker answered records none.
func forcedProbeResult(status int, err error) string {
	var upstream *pipeline.UpstreamError
	var streamed *pipeline.UpstreamStreamedError
	switch {
	case status == 0 && errors.As(err, &upstream) && upstream.StatusCode == 0:
		return coordmetrics.ProbeResultTransportError
	case status == 0:
		return ""
	case errors.As(err, &streamed):
		return coordmetrics.ProbeResultError
	default:
		return coordmetrics.ProbeResultServed
	}
}

func (s *ConditionalDecodeStep) prepareBody(reqCtx *pipeline.RequestContext) (map[string]any, error) {
	body := maps.Clone(reqCtx.Body)
	format := reqcommon.DetectAPIType(reqCtx.OriginalPath)

	switch format {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses, reqcommon.APITypeMessages:
		// The client's chat-completions, responses, or messages body is
		// forwarded as-is. The server drops any client kv_transfer_params, so
		// this map holds only the IDs.
		if sendsPromptTokenIDs(s.forwardTokenIDs, reqCtx, format) {
			body[reqcommon.FieldKVTransferParams] = map[string]any{reqcommon.FieldPromptTokenIDs: reqCtx.TokenIDs}
		}
	case reqcommon.APITypeCompletions:
		if len(reqCtx.TokenIDs) > 0 {
			body[reqcommon.FieldPrompt] = reqCtx.TokenIDs
		}
	case reqcommon.APITypeVLLMGenerate:
		// The client's generate body already carries token_ids.
	default:
		return nil, unreachableFormatError(format)
	}
	return body, nil
}
