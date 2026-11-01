/*
Copyright 2025 The Kubernetes Authors.
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
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
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
			metrics.RecordInterTokenLatency(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.apiSurface, fairnessID, priority, itl)
		}
		reqCtx.lastChunkReceivedTimestamp = now
	}

	var parsedResp *fwkrh.ParsedResponse
	parseBytes := responseBytes
	if isEventStream(reqCtx.Response.Headers[reqcommon.HeaderContentType]) {
		parseBytes = completeSSELines(ctx, reqCtx, responseBytes, endOfStream)
	}
	parser, err := s.getOrResolveParser(reqCtx)
	if err != nil {
		logger.Error(err, "parsing response: failed to resolve parser")
	} else if len(parseBytes) > 0 || endOfStream {
		before := time.Now()
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
		mergeUsage(&reqCtx.Usage, *parsedResp.Usage)
	}
	if endOfStream {
		// Recorded once here rather than per chunk: a server reporting cumulative
		// usage on more than one chunk would otherwise be observed once per chunk.
		// A stream that ends without reaching end of stream records no token counts,
		// since the abnormal-termination path calls the director directly.
		metrics.RecordInputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.apiSurface, fairnessID, priority, reqCtx.Usage.PromptTokens)
		metrics.RecordOutputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.apiSurface, fairnessID, priority, reqCtx.Usage.CompletionTokens)
		if reqCtx.Usage.PromptTokenDetails != nil {
			metrics.RecordPromptCachedTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.apiSurface, fairnessID, priority, reqCtx.Usage.PromptTokenDetails.CachedTokens)
		}
		metrics.RecordNormalizedTimePerOutputToken(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.apiSurface, fairnessID, priority, reqCtx.RequestReceivedTimestamp, reqCtx.responseCompleteTimestamp, reqCtx.Usage.CompletionTokens)
		metrics.RecordRequestLatencies(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.apiSurface, fairnessID, priority, reqCtx.RequestReceivedTimestamp, reqCtx.responseCompleteTimestamp)
		metrics.RecordResponseSizes(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.apiSurface, fairnessID, priority, reqCtx.responseSize)
		metrics.RecordRequestTTFT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.apiSurface, fairnessID, priority, reqCtx.modelServerStreaming, reqCtx.RequestReceivedTimestamp, reqCtx.firstTokenTimestamp)
		metrics.RecordRequestTPOT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.apiSurface, fairnessID, priority, reqCtx.modelServerStreaming, reqCtx.RequestReceivedTimestamp, reqCtx.firstTokenTimestamp, reqCtx.responseCompleteTimestamp, reqCtx.Usage.CompletionTokens)
	}
	return s.director.HandleResponseBody(ctx, reqCtx, endOfStream)
}

// maxSSELineBytes caps one buffered SSE line. It is above the tool-calling 1 MiB event cap
// because response.completed repeats the request's instructions and tools plus all output.
const maxSSELineBytes = 4 << 20

// isEventStream reports whether a content-type header value is text/event-stream.
func isEventStream(contentType string) bool {
	mediaType, _, _ := strings.Cut(contentType, ";")
	return strings.EqualFold(strings.TrimSpace(mediaType), fwkrequest.MediaTypeEventStream)
}

// completeSSELines returns only whole lines, carrying a partial last line to the next chunk.
// A line longer than maxSSELineBytes is dropped up to its newline.
func completeSSELines(ctx context.Context, reqCtx *RequestContext, chunk []byte, endOfStream bool) []byte {
	if reqCtx.sseLineDiscarding {
		lineEnd := bytes.IndexByte(chunk, '\n')
		if lineEnd < 0 {
			return nil
		}
		reqCtx.sseLineDiscarding = false
		chunk = chunk[lineEnd+1:]
	}
	lineEnd := bytes.LastIndexByte(chunk, '\n')
	if lineEnd < 0 {
		if len(reqCtx.sseLineCarry)+len(chunk) > maxSSELineBytes {
			dropSSELine(ctx, reqCtx, endOfStream)
			return nil
		}
		// Appending to the carry grows it in place, keeping a line spread over many chunks linear.
		reqCtx.sseLineCarry = append(reqCtx.sseLineCarry, chunk...)
		if endOfStream {
			carry := reqCtx.sseLineCarry
			reqCtx.sseLineCarry = nil
			return carry
		}
		return nil
	}
	data := chunk[:lineEnd+1]
	if len(reqCtx.sseLineCarry) > 0 {
		data = append(reqCtx.sseLineCarry, data...)
	}
	reqCtx.sseLineCarry = nil
	tail := chunk[lineEnd+1:]
	switch {
	case len(tail) > maxSSELineBytes:
		dropSSELine(ctx, reqCtx, endOfStream)
	case endOfStream:
		data = append(data, tail...)
	case len(tail) > 0:
		reqCtx.sseLineCarry = bytes.Clone(tail)
	}
	return dropOversizedLines(ctx, data)
}

// dropOversizedLines removes the lines of data longer than maxSSELineBytes, copying only when one is dropped.
func dropOversizedLines(ctx context.Context, data []byte) []byte {
	if len(data) <= maxSSELineBytes {
		return data
	}
	var kept []byte
	dropped := false
	for start := 0; start < len(data); {
		end := len(data)
		if i := bytes.IndexByte(data[start:], '\n'); i >= 0 {
			end = start + i + 1
		}
		line := data[start:end]
		if len(bytes.TrimSuffix(line, []byte("\n"))) > maxSSELineBytes {
			if !dropped {
				logSSELineDrop(ctx)
				kept = append(kept, data[:start]...)
				dropped = true
			}
		} else if dropped {
			kept = append(kept, line...)
		}
		start = end
	}
	if !dropped {
		return data
	}
	return kept
}

// dropSSELine drops the line being read and, mid-stream, skips the rest of it up to its newline.
func dropSSELine(ctx context.Context, reqCtx *RequestContext, endOfStream bool) {
	logSSELineDrop(ctx)
	reqCtx.sseLineCarry = nil
	reqCtx.sseLineDiscarding = !endOfStream
}

func logSSELineDrop(ctx context.Context) {
	log.FromContext(ctx).V(logutil.DEFAULT).Info("dropping oversized SSE line from response parsing", "limit", maxSSELineBytes)
}

// mergeUsage folds a parsed usage block into the usage accumulated for the request.
// Anthropic streaming reports usage on message_start and again on message_delta, and a
// block may carry only some of the fields, so a zero is read as "not reported" and
// leaves the accumulated value alone. Parsers that emit usage once with every field
// populated are unaffected.
func mergeUsage(dst *fwkrh.Usage, src fwkrh.Usage) {
	if src.PromptTokens != 0 {
		dst.PromptTokens = src.PromptTokens
	}
	if src.CompletionTokens != 0 {
		dst.CompletionTokens = src.CompletionTokens
	}
	if src.PromptTokenDetails != nil {
		dst.PromptTokenDetails = src.PromptTokenDetails
	}
	// A block reporting both halves of the usage owns the total it came with; a partial
	// block carries a total covering only its own fields, so derive it from the merge.
	if src.PromptTokens != 0 && src.CompletionTokens != 0 && src.TotalTokens != 0 {
		dst.TotalTokens = src.TotalTokens
		return
	}
	dst.TotalTokens = dst.PromptTokens + dst.CompletionTokens
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
