/*
Copyright 2025 The Kubernetes Authors.

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

package vllmgrpc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/controller-runtime/pkg/log"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers"
	grpcutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
	pb "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"
)

const (
	VllmGRPCParserType = "vllmgrpc-parser"

	vllmGeneratePath = "/vllm.grpc.engine.VllmEngine/Generate"
	vllmEmbedPath    = "/vllm.grpc.engine.VllmEngine/Embed"
)

// compile-time type validation
var _ fwkrh.Parser = &VllmGRPCParser{}

// VllmGRPCParser implements the fwkrh.Parser interface for vLLM gRPC.
type VllmGRPCParser struct {
	typedName fwkplugin.TypedName
}

// NewVllmGRPCParser creates a new VllmGRPCParser.
func NewVllmGRPCParser() *VllmGRPCParser {
	return &VllmGRPCParser{
		typedName: fwkplugin.TypedName{
			Type: VllmGRPCParserType,
			Name: VllmGRPCParserType,
		},
	}
}

func VllmGRPCParserPluginFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewVllmGRPCParser().WithName(name), nil
}

func (p *VllmGRPCParser) WithName(name string) *VllmGRPCParser {
	p.typedName.Name = name
	return p
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *VllmGRPCParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *VllmGRPCParser) SupportedAppProtocols() []v1.AppProtocol {
	return []v1.AppProtocol{v1.AppProtocolH2C}
}

// ParseRequest parses the gRPC request body and headers and returns an InferenceRequestBody.
func (p *VllmGRPCParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	logger := log.FromContext(ctx)

	parsedPayload, err := grpcutil.ParseGrpcPayload(body)
	if err != nil {
		return nil, errors.New("invalid or unsupported gRPC payload")
	}

	path := headers[parsers.MethodPathKey]
	switch path {
	case vllmEmbedPath:
		var req pb.EmbedRequest
		if err := proto.Unmarshal(parsedPayload, &req); err != nil {
			return nil, fmt.Errorf("unmarshaling EmbedRequest: %w", err)
		}
		extractedBody, err := convertEmbedToInferenceRequestBody(&req)
		if err != nil {
			return nil, err
		}
		logger.V(logutil.TRACE).Info("parsed EmbedRequest")
		return &fwkrh.ParseResult{Body: extractedBody, Skip: false}, nil

	case vllmGeneratePath:
		var req pb.GenerateRequest
		if err := proto.Unmarshal(parsedPayload, &req); err != nil {
			return nil, fmt.Errorf("unmarshaling GenerateRequest: %w", err)
		}
		extractedBody, err := convertToInferenceRequestBody(&req)
		if err != nil {
			return nil, err
		}
		logger.V(logutil.TRACE).Info("parsed GenerateRequest")
		return &fwkrh.ParseResult{Body: extractedBody, Skip: false}, nil

	default:
		logger.V(logutil.TRACE).Info("unsupported gRPC path, skipping", "path", headers[parsers.MethodPathKey])
		return &fwkrh.ParseResult{Skip: true}, nil
	}
}

// ParseResponse parses the response body and returns a ParsedResponse
func (p *VllmGRPCParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endofStream bool) (*fwkrh.ParsedResponse, error) {
	logger := log.FromContext(ctx)
	resp := &pb.GenerateResponse{}

	// Try to parse as GenerateResponse first. If it fails or the response field is nil,
	// try to parse as EmbedResponse.
	if err := toGenerateResponse(body, resp); err != nil || resp.Response == nil {
		embedResp := &pb.EmbedResponse{}
		if err := toEmbedResponse(body, embedResp); err == nil && (len(embedResp.Embedding) > 0 || embedResp.PromptTokens > 0) {
			logger.V(logutil.DEBUG).Info("parsed EmbedResponse", "promptTokens", embedResp.PromptTokens)
			result := &fwkrh.ParsedResponse{}
			if embedResp.PromptTokens > 0 {
				result.Usage = &fwkrh.Usage{
					PromptTokens:     int(embedResp.PromptTokens),
					CompletionTokens: 0,
					TotalTokens:      int(embedResp.PromptTokens),
				}
			}
			return result, nil
		}
		return nil, fmt.Errorf("failed to parse gRPC response payload as GenerateResponse or EmbedResponse: %w", err)
	}

	result := &fwkrh.ParsedResponse{}

	switch v := resp.Response.(type) {
	case *pb.GenerateResponse_Chunk:
		logger.V(logutil.DEBUG).Info("parsed GenerateResponse_Chunk", "tokenLength", len(v.Chunk.TokenIds))
		// Only populate Usage if the chunk actually contains token data.
		// Streaming chunks often leave this empty until the final chunk.
		promptToken, completionToken, cachedToken := int(v.Chunk.PromptTokens), int(v.Chunk.CompletionTokens), int(v.Chunk.CachedTokens)
		if promptToken > 0 || completionToken > 0 {
			result.Usage = requestControlUsage(promptToken, completionToken, cachedToken)
		}

	case *pb.GenerateResponse_Complete:
		logger.V(logutil.DEBUG).Info("parsed GenerateResponse_Complete", "finishReason", v.Complete.FinishReason)
		// Populate Usage for complete, non-streaming responses.
		promptToken, completionToken, cachedToken := int(v.Complete.PromptTokens), int(v.Complete.CompletionTokens), int(v.Complete.CachedTokens)
		if promptToken > 0 || completionToken > 0 {
			result.Usage = requestControlUsage(promptToken, completionToken, cachedToken)
		}

	default:
		return nil, errors.New("unrecognized response type in GenerateResponse")
	}

	return result, nil
}

func requestControlUsage(promptToken, completionToken, cachedToken int) *fwkrh.Usage {
	return &fwkrh.Usage{
		PromptTokens:     promptToken,
		CompletionTokens: completionToken,
		TotalTokens:      promptToken + completionToken,
		PromptTokenDetails: &fwkrh.PromptTokenDetails{
			CachedTokens: cachedToken,
		},
	}
}

func toGenerateResponse(payload []byte, resp *pb.GenerateResponse) error {
	parsedPayload, err := grpcutil.ParseGrpcPayload(payload)
	if err != nil {
		return err
	}
	return proto.Unmarshal(parsedPayload, resp)
}

func convertToInferenceRequestBody(pbReq *pb.GenerateRequest) (*fwkrh.InferenceRequestBody, error) {
	var body *fwkrh.InferenceRequestBody
	switch pbReq.Input.(type) {
	case *pb.GenerateRequest_Text:
		body = &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: pbReq.GetText()},
			},
			Payload: fwkrh.PayloadProto{Message: pbReq},
		}
	case *pb.GenerateRequest_Tokenized:
		tokenized := pbReq.GetTokenized()
		if tokenized == nil {
			return nil, errors.New("missing tokenized input")
		}
		inputIDs := tokenized.GetInputIds()
		copiedTokenIDsInt := make([]uint32, len(inputIDs))
		copy(copiedTokenIDsInt, inputIDs)
		body = &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{TokenIDs: copiedTokenIDsInt},
			},
			Payload: fwkrh.PayloadProto{Message: pbReq},
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				TokenIDs:           copiedTokenIDsInt,
				MultiModalFeatures: convertMultiModalFeatures(pbReq.GetMmInputs()),
			},
		}
	default:
		return nil, errors.New("not supported request inputType")
	}
	body.Stream = pbReq.GetStream()
	return body, nil
}

func convertMultiModalFeatures(mmInputs *pb.MultimodalInputs) []fwkrh.MultiModalFeature {
	if mmInputs == nil {
		return nil
	}

	placeholders := mmInputs.GetMmPlaceholders()
	if len(placeholders) == 0 {
		return nil
	}

	hashes := mmInputs.GetMmHashes()
	features := make([]fwkrh.MultiModalFeature, 0, len(placeholders))
	for i, placeholder := range placeholders {
		hash := ""
		if i < len(hashes) {
			hash = hashes[i]
		}

		feature := fwkrh.MultiModalFeature{
			Modality: fwkrh.ModalityImage,
			Hash:     hash,
		}
		if placeholder != nil {
			feature.Offset = int(placeholder.GetOffset())
			feature.Length = int(placeholder.GetLength())
		}

		features = append(features, feature)
	}

	return features
}

func convertEmbedToInferenceRequestBody(pbReq *pb.EmbedRequest) (*fwkrh.InferenceRequestBody, error) {
	var body *fwkrh.InferenceRequestBody
	if pbReq.Tokenized != nil {
		inputIDs := pbReq.GetTokenized().InputIds
		tokenIDs := make([]uint32, len(inputIDs))
		copy(tokenIDs, inputIDs)
		body = &fwkrh.InferenceRequestBody{
			Embeddings: &fwkrh.EmbeddingsRequest{
				Input: fwkrh.EmbeddingsInput{
					TokenIDs: tokenIDs,
				},
			},
			Payload: fwkrh.PayloadProto{Message: pbReq},
		}
	} else {
		return nil, errors.New("missing tokenized input in EmbedRequest")
	}
	return body, nil
}

func toEmbedResponse(payload []byte, resp *pb.EmbedResponse) error {
	parsedPayload, err := grpcutil.ParseGrpcPayload(payload)
	if err != nil {
		return err
	}
	return proto.Unmarshal(parsedPayload, resp)
}
