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
	"net/http"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const DecodeStepName = "decode"

func init() {
	pipeline.Register(DecodeStepName, NewDecodeStep)
}

type DecodeStep struct {
	gwClient *gateway.Client
	kv       kv.Connector
}

func NewDecodeStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("decode: gateway client is required")
	}
	if err := rejectUseOpenAIFormatOverride(DecodeStepName, params); err != nil {
		return nil, err
	}
	kvName, err := paramString(params, ParamKVConnector)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	kvConn, err := kv.Build(kvName)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &DecodeStep{gwClient: gwClient, kv: kvConn}, nil
}

func (s *DecodeStep) Name() string { return DecodeStepName }

func (s *DecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(DecodeStepName)

	if err := s.prepareDecodeBody(ctx, reqCtx); err != nil {
		return err
	}

	logger.V(logutil.DEFAULT).Info("sending request", "path", reqCtx.OriginalPath, "stream", reqCtx.Stream)

	proxyReq, err := newDecodeProxyRequest(ctx, logger, DecodeStepName, reqCtx, s.gwClient, reqCtx.Body, nil)
	if err != nil {
		return err
	}

	transport := instrumentedTransport(s.gwClient.Transport(), coordmetrics.UpstreamDecode)
	proxy, out := newDecodeProxy(logger, transport, nil)
	proxy.ServeHTTP(reqCtx.ResponseWriter, proxyReq)
	if out.TransportErr != nil {
		return &pipeline.UpstreamStreamedError{Step: DecodeStepName, Cause: out.TransportErr}
	}
	if out.Status >= http.StatusBadRequest {
		return &pipeline.UpstreamStreamedError{Step: DecodeStepName, StatusCode: out.Status}
	}
	return nil
}

// prepareDecodeBody mutates reqCtx.Body in place rather than on a clone (unlike
// prefill and conditional-decode). decode is the terminal pipeline step: its body
// is streamed straight to the client and no later step reads reqCtx.Body. A clone
// would also be insufficient, since injectUUIDs mutates nested values that a shallow
// maps.Clone would still share. This is sound only while the pipeline runs steps
// sequentially; if it ever goes concurrent, decode must copy like the others.
func (s *DecodeStep) prepareDecodeBody(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	format := reqcommon.DetectAPIType(reqCtx.OriginalPath)

	kvParams := s.kv.PrepareDecodeKVParams(ctx, reqCtx)
	s.injectUUIDs(reqCtx)

	switch format {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeVLLMGenerate:
		reqCtx.Body[reqcommon.FieldKVTransferParams] = kvParams
	case reqcommon.APITypeCompletions:
		reqCtx.Body[reqcommon.FieldKVTransferParams] = kvParams
		if len(reqCtx.TokenIDs) > 0 {
			reqCtx.Body["prompt"] = reqCtx.TokenIDs
		}
	default:
		// kvParams and injectUUIDs above already ran; both are harmless here
		// since the request fails on this return and reqCtx.Body is never sent.
		return unreachableFormatError(format)
	}
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
