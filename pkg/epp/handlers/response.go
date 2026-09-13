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

package handlers

import (
	"bytes"
	"context"
	"strconv"
	"strings"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/envoy"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/util/request"
)

// HandleResponseBody processes response data for both streaming and non-streaming models.
//
// Streaming case:
//
//	Invoked multiple times as data chunks arrive. The final call is identified by
//	endOfStream=true, triggering final metric collection and plugin cleanup.
//
// Non-streaming case:
//
//	Invoked exactly once with endOfStream=true. It processes the entire response
//
// body as a single "stream" event.
func (s *StreamingServer) HandleResponseBody(ctx context.Context, reqCtx *RequestContext, responseBytes []byte, endOfStream bool) *RequestContext {
	logger := log.FromContext(ctx)
	// The Enabled() guard is intentional: passing arguments to a disabled logger
	// still boxes them into a heap-allocated slice, and this runs per chunk.
	if debug := logger.V(logutil.DEBUG); debug.Enabled() {
		debug.Info("HandleResponseBody is triggered", "len(responseBytes)", len(responseBytes), "endOfStream", endOfStream)
	}

	fairnessID, priority := extractFairnessAndPriority(reqCtx)

	reqCtx.responseSize += len(responseBytes)

	if reqCtx.firstTokenTimestamp.IsZero() && len(responseBytes) > 0 {
		reqCtx.firstTokenTimestamp = time.Now()
	}

	if reqCtx.modelServerStreaming && len(responseBytes) > 0 {
		now := time.Now()
		if !reqCtx.lastChunkReceivedTimestamp.IsZero() {
			itl := now.Sub(reqCtx.lastChunkReceivedTimestamp).Seconds()
			metrics.RecordInterTokenLatency(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, itl)
		}
		reqCtx.lastChunkReceivedTimestamp = now
	}

	var parsedResp *fwkrh.ParsedResponse
	parser, err := s.getOrResolveParser(ctx, reqCtx)
	if err != nil {
		logger.Error(err, "parsing response: failed to resolve parser")
	} else {
		before := time.Now()
		parseBytes := responseBytes
		if strings.Contains(reqCtx.Response.Headers["content-type"], "text/event-stream") {
			parseBytes = reqCtx.completeUsageLines(responseBytes, endOfStream)
		}
		parsedResp, err = parser.ParseResponse(ctx, parseBytes, reqCtx.Response.Headers, endOfStream)
		metrics.RecordPluginProcessingLatency(fwkrh.ResponseParsingExtensionPoint, parser.TypedName().Type, parser.TypedName().Name, time.Since(before))
		if err != nil {
			logger.Error(err, "parsing response")
		}
	}
	if parsedResp != nil {
		reqCtx.StreamedEvents += parsedResp.StreamedEvents
	}
	if parsedResp != nil && parsedResp.Usage != nil {
		reqCtx.Usage.MergeCumulative(*parsedResp.Usage)
		reqCtx.Usage.TotalTokens = max(reqCtx.Usage.TotalTokens, reqCtx.Usage.PromptTokens+reqCtx.Usage.CompletionTokens)
	}
	if endOfStream && !reqCtx.usageMetricsRecorded {
		reqCtx.usageMetricsRecorded = true
		metrics.RecordInputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, reqCtx.Usage.PromptTokens)
		metrics.RecordOutputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, reqCtx.Usage.CompletionTokens)
		if reqCtx.Usage.PromptTokenDetails != nil {
			metrics.RecordPromptCachedTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, reqCtx.Usage.PromptTokenDetails.CachedTokens)
		}
	}
	if endOfStream {
		metrics.RecordNormalizedTimePerOutputToken(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, reqCtx.RequestReceivedTimestamp, reqCtx.responseCompleteTimestamp, reqCtx.Usage.CompletionTokens)
		metrics.RecordRequestLatencies(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, reqCtx.RequestReceivedTimestamp, reqCtx.responseCompleteTimestamp)
		metrics.RecordResponseSizes(reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, reqCtx.responseSize)
		metrics.RecordRequestTTFT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, reqCtx.modelServerStreaming, reqCtx.RequestReceivedTimestamp, reqCtx.firstTokenTimestamp)
		metrics.RecordRequestTPOT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, reqCtx.modelServerStreaming, reqCtx.RequestReceivedTimestamp, reqCtx.firstTokenTimestamp, reqCtx.responseCompleteTimestamp, reqCtx.Usage.CompletionTokens)
	}
	return s.director.HandleResponseBody(ctx, reqCtx, endOfStream)
}

// HTTP fragments need not end at SSE line boundaries. Forwarding uses the
// original bytes; this buffer is only for response parsing.
func (r *RequestContext) completeUsageLines(chunk []byte, end bool) []byte {
	if r.discardUsageLine {
		i := bytes.IndexByte(chunk, '\n')
		if i < 0 {
			return nil
		}
		chunk = chunk[i+1:]
		r.discardUsageLine = false
	}
	r.responseUsageTail = append(r.responseUsageTail, chunk...)
	data := r.responseUsageTail
	r.responseUsageTail = nil
	if end {
		return data
	}
	i := bytes.LastIndexByte(data, '\n')
	if len(data)-i-1 > 1<<20 {
		r.discardUsageLine = true
	} else {
		r.responseUsageTail = bytes.Clone(data[i+1:])
	}
	return data[:i+1]
}

func (s *StreamingServer) HandleResponseHeaders(ctx context.Context, reqCtx *RequestContext, resp *extProcPb.ProcessingRequest_ResponseHeaders) *RequestContext {
	for _, header := range resp.ResponseHeaders.Headers.Headers {
		reqCtx.Response.Headers[header.Key] = envoy.GetHeaderValue(header)
	}
	return s.director.HandleResponseHeader(ctx, reqCtx)
}

func (s *StreamingServer) generateResponseHeaderResponse(reqCtx *RequestContext) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: s.generateResponseHeaders(reqCtx),
					},
				},
			},
		},
	}
}

func generateResponseBodyResponses(responseBodyBytes []byte, setEoS bool, dynamicMetadata *structpb.Struct) []*extProcPb.ProcessingResponse {
	commonResponses := envoy.BuildChunkedBodyResponses(responseBodyBytes, setEoS)
	responses := make([]*extProcPb.ProcessingResponse, 0, len(commonResponses))
	for _, commonResp := range commonResponses {
		resp := &extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ResponseBody{
				ResponseBody: &extProcPb.BodyResponse{
					Response: commonResp,
				},
			},
		}
		responses = append(responses, resp)
	}

	// Attach dynamic metadata to the last response if available.
	if len(responses) > 0 && dynamicMetadata != nil {
		responses[len(responses)-1].DynamicMetadata = dynamicMetadata
	}
	return responses
}

func (s *StreamingServer) generateResponseHeaders(reqCtx *RequestContext) []*configPb.HeaderValueOption {
	// can likely refactor these two bespoke headers to be updated in PostDispatch, to centralize logic.
	headers := []*configPb.HeaderValueOption{
		{
			Header: &configPb.HeaderValue{
				// This is for debugging purpose only.
				Key:      "x-went-into-resp-headers",
				RawValue: []byte("true"),
			},
		},
	}

	// Stamp the flow control queue duration ahead of the streamed body so it reaches the client before the
	// first token. Absent when flow control did not process the request; zero means a dispatch with no
	// measurable queueing.
	if reqCtx.FlowControlAdmitted {
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      metadata.FlowQueueDurationHeaderKey,
				RawValue: []byte(strconv.FormatInt(reqCtx.FlowControlQueueDuration.Milliseconds(), 10)),
			},
		})
	}

	// Include any non-system-owned headers.
	for key, value := range reqCtx.Response.Headers {
		if request.IsSystemOwnedHeader(key) {
			continue
		}
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      key,
				RawValue: []byte(value),
			},
		})
	}
	return headers
}
