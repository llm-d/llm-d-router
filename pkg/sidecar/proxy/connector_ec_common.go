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

// This file holds the encoder fan-out scaffolding shared by every EC
// connector: multimodal-item extraction and the parallel
// per-item encoder dispatch loop. Each EC connector
// (ec-example via fanoutEncoderPrimer, ec-nixl via fanoutEncoderCollect,
// ec-mooncake via fanoutEncoderMooncake) supplies its own per-response perItem
// callback and otherwise reuses these helpers verbatim.

package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
	"golang.org/x/sync/errgroup"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// requestInput returns the request's Responses input items. A bare JSON string
// (a single text turn), an explicit null and an absent field all yield a nil
// slice and no error. Nothing writes a decoded slice back under input, so the
// raw form is the only one that arrives; requestMessages accepts a decoded
// slice because chunked decode writes one back.
func requestInput(req map[string]any) ([]json.RawMessage, error) {
	switch v := req[reqcommon.FieldInput].(type) {
	case nil:
		return nil, nil
	case json.RawMessage:
		var items []json.RawMessage
		if err := json.Unmarshal(v, &items); err != nil {
			var s string
			if json.Unmarshal(v, &s) == nil {
				return nil, nil
			}
			return nil, err
		}
		return items, nil
	default:
		return nil, fmt.Errorf("input is %T, want a JSON array or string", v)
	}
}

// extractMMItems extracts all multimodal content parts from the request: the
// messages array of chat completions or Messages, or a Responses input array.
// Which field to walk is gated on apiType rather than presence, since a client
// could send a stray field the other format does not use. The part arrays
// within a turn come from reqcommon.ItemPartArrays.
//
// One item is returned per content part, repeats included. A part's modality and
// its client-supplied uuid both move the serving engine's multimodal hash, so
// collapsing two parts that share a URL can leave the prefiller looking up a
// hash nothing primed.
func extractMMItems(logger logr.Logger, requestData map[string]any, apiType reqcommon.APIType) []map[string]any {
	var items []map[string]any

	var wrapped []json.RawMessage
	var err error
	switch apiType {
	case reqcommon.APITypeResponses:
		wrapped, err = requestInput(requestData)
	default:
		wrapped, err = requestMessages(requestData)
	}
	if err != nil {
		logger.V(logging.DEBUG).Info("cannot read request content for multimodal extraction", "error", jsonErrorForLog(err))
		return items
	}

	// A dropped part never enters items, so fanoutEncoderCollect's
	// contributed/total pair cannot surface it and these counts are the only
	// signal. A part of a type the encoder never primes goes uncounted, since
	// counting it would fire the line on ordinary text traffic.
	droppedParts := 0
	// A turn that does not decode carries an unknown number of parts, so it is
	// reported on its own line rather than added to droppedParts. Both a chat
	// messages entry and a Responses input item land here, so the line names
	// neither field and carries apiType instead.
	droppedTurns := 0
	collect := func(parts []any) {
		for _, part := range parts {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}
			partType, ok := partMap[reqcommon.FieldType].(string)
			if !ok {
				continue
			}

			if apiType == reqcommon.APITypeMessages {
				// Anthropic image blocks are not a chat part. The encoder is
				// posted at /v1/chat/completions, which only primes image_url,
				// so reshape before the part is forwarded as-is. ItemPartArrays
				// returns only the arrays vLLM loads images from: a turn's
				// content and, on a user turn, each tool_result's content one
				// level deep. Which blocks name media is
				// reqcommon.MessagesBlockModality's rule, shared with the
				// coordinator steps. A chat part type fails /v1/messages input
				// validation, so that request never reaches a worker.
				_, isBlockMedia := reqcommon.MessagesBlockModality(partType)
				_, isChatMedia := reqcommon.PartModality(partType)
				switch {
				case isBlockMedia:
					converted, ok := anthropicImagePart(partMap)
					if !ok {
						logger.V(logging.DEBUG).Info("skipping anthropic image the encoder cannot be primed with")
						droppedParts++
						continue
					}
					items = append(items, converted)
				case isChatMedia:
					logger.V(logging.DEBUG).Info("skipping content part the Messages API does not define", "type", partType, "apiType", apiType)
					droppedParts++
				}
				continue
			}

			// reqcommon.PartModality's rule, shared with the coordinator steps
			// so the two cannot disagree about which parts exist. A part naming
			// no media is not counted as a drop: every text and tool part lands
			// here, so counting them would fire droppedParts on ordinary traffic.
			if _, isMedia := reqcommon.PartModality(partType); !isMedia {
				continue
			}

			// An input_image carries its media behind a URL only, so one without
			// a readable URL has nothing to prime an encoder with. Specific to
			// it: an input_audio holds its bytes inline and reports no URL, so a
			// general check would drop audio the encoder can be primed with.
			if partType == reqcommon.PartTypeInputImage && reqcommon.MediaPartURL(partMap) == "" {
				logger.V(logging.DEBUG).Info("skipping input_image with no fetchable URL")
				droppedParts++
				continue
			}

			items = append(items, partMap)
		}
	}

	for _, raw := range wrapped {
		var turn map[string]any
		// UseNumber keeps a number outside float64 range readable: Python's json
		// parses it, so vLLM serves a body the default decoder would reject, and
		// dropping the turn would leave its images unprimed. json.Number also
		// re-marshals as the literal the client sent.
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.UseNumber()
		if err := dec.Decode(&turn); err != nil {
			logger.V(logging.DEBUG).Info("skipping turn that does not decode as an object", "error", jsonErrorForLog(err))
			droppedTurns++
			continue
		}
		for _, array := range reqcommon.ItemPartArrays(turn, apiType) {
			collect(array.Parts)
		}
	}

	if droppedParts > 0 {
		logger.Info("skipped multimodal content parts the encoder cannot be primed with",
			"count", droppedParts, "extracted", len(items), "apiType", apiType)
	}
	if droppedTurns > 0 {
		logger.Info("skipped turns that do not decode as an object",
			"count", droppedTurns, "extracted", len(items), "apiType", apiType)
	}

	return items
}

// mmItemsForFanout extracts the fanout items for one request, tagging the
// extraction logs with requestID.
func (s *Server) mmItemsForFanout(ctx context.Context, originalRequest map[string]any, requestID string, apiType reqcommon.APIType) []map[string]any {
	return extractMMItems(log.FromContext(ctx).WithValues("requestID", requestID), originalRequest, apiType)
}

// anthropicImagePart turns a Messages API image block into the image_url
// part the chat-completions encoder primes.
func anthropicImagePart(part map[string]any) (map[string]any, bool) {
	imageURL := reqcommon.MediaPartURL(part)
	if imageURL == "" {
		return nil, false
	}
	return map[string]any{
		"type":      reqcommon.PartTypeImageURL,
		"image_url": map[string]any{"url": imageURL},
	}, true
}

// fanoutEncoder fans out one encoder request per item, in parallel, with
// round-robin over encoderHostPorts. perItem is invoked once per item AFTER
// the encoder has returned a 2xx response; it receives the item's
// positional index and the buffered encoder response. The
// callback may return an error to fail the whole fan-out, or nil to
// accept. perItem may be nil for fire-and-forget primer-style usage.
// prepare, when non-nil, may add fields to an item's encoder request body
// before it is sent.
//
// The first goroutine to fail cancels the group context so sibling encoder
// requests are aborted at the transport layer. Every failure is logged before
// propagating; grp.Wait returns the first non-nil error.
func (s *Server) fanoutEncoder(
	ctx context.Context,
	originalRequest map[string]any,
	items []map[string]any,
	encoderHostPorts []string,
	requestID string,
	apiType reqcommon.APIType,
	prepare func(idx int, body map[string]any),
	perItem func(idx int, pw *bufferedResponseWriter) error,
) error {
	if len(encoderHostPorts) == 0 {
		return fmt.Errorf("fanoutEncoder: no encoder hostPorts provided (requestID=%s)", requestID)
	}

	logger := log.FromContext(ctx)
	encoderPath := reqcommon.PathChatCompletions
	if apiType == reqcommon.APITypeResponses {
		encoderPath = reqcommon.PathResponses
	}

	logger.Info("processing multimodal items", "count", len(items), "requestID", requestID, "encoderHostPorts", encoderHostPorts)

	grp, gctx := errgroup.WithContext(ctx)
	for idx, mmItem := range items {
		hostPort := encoderHostPorts[idx%len(encoderHostPorts)]
		grp.Go(func() (err error) {
			logger := log.FromContext(gctx)
			defer func() {
				if rec := recover(); rec != nil {
					if rec != http.ErrAbortHandler {
						panic(rec)
					}
					err = fmt.Errorf("encoder response interrupted for item %d: %w", idx, http.ErrAbortHandler)
					logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				}
			}()

			// A Messages request gets NewEncoderPrimingBody's Messages body at
			// the chat path; see its doc for why that holds.
			encoderRequest := reqcommon.NewEncoderPrimingBody(originalRequest, mmItem, apiType)
			if prepare != nil {
				prepare(idx, encoderRequest)
			}

			body, err := json.Marshal(encoderRequest)
			if err != nil {
				err = fmt.Errorf("failed to marshal encoder request for item %d: %w", idx, err)
				logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}

			encoderHandler, err := s.encoderProxyHandler(hostPort)
			if err != nil {
				err = fmt.Errorf("failed to get encoder proxy handler for %s: %w", hostPort, err)
				logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}

			req, err := http.NewRequestWithContext(gctx, "POST", encoderPath, bytes.NewReader(body))
			if err != nil {
				err = fmt.Errorf("failed to create encoder request for item %d: %w", idx, err)
				logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(reqcommon.RequestIDHeaderKey, fmt.Sprintf("%s-enc-%d", requestID, idx))

			logger.V(logging.DEBUG).Info("sending encoder request", "item", idx, "to", hostPort, "requestID", requestID)

			pw := &bufferedResponseWriter{}
			encoderHandler.ServeHTTP(pw, req)

			// bufferedResponseWriter.Write back-fills 200, so statusCode 0 means
			// the handler wrote nothing at all. isHTTPError treats it as an error.
			if isHTTPError(pw.statusCode) {
				err := fmt.Errorf("encoder request failed for item %d with status %d", idx, pw.statusCode)
				logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}

			if perItem != nil {
				if err := perItem(idx, pw); err != nil {
					logger.Error(jsonErrorForLog(err), "encoder fanout perItem", "item", idx, "requestID", requestID)
					return err
				}
			}

			logger.V(logging.DEBUG).Info("encoder request completed", "item", idx, "requestID", requestID)
			return nil
		})
	}
	return grp.Wait()
}

// runPDPipeline finalizes the post-encoder request and dispatches it to the
// configured P/D connector or directly to the decoder. The caller has already
// generated requestID and merged any encoder-side metadata into
// body. On JSON-marshal failure, runPDPipeline writes the error
// response itself (matching the existing handler pattern) and returns.
func (s *Server) runPDPipeline(
	w http.ResponseWriter,
	r *http.Request,
	body map[string]any,
	prefillEndPoint string,
	kvCacheSource string,
	requestID string,
	apiType reqcommon.APIType,
) {
	logger := log.FromContext(r.Context())
	// Skip decode-first; the encoder has run and prefill must execute.
	body[reqcommon.FieldCacheHitThreshold] = 0

	modifiedBody, err := json.Marshal(body)
	if err != nil {
		logger.Error(err, "failed to marshal request after encoder", "requestID", requestID)
		if writeErr := errorJSONInvalid(err, w); writeErr != nil {
			logger.Error(writeErr, "failed to send error response to client", "requestID", requestID)
		}
		return
	}

	pdRequest := cloneRequestWithBody(r.Context(), r, modifiedBody)
	pdRequest.Header.Add(reqcommon.RequestIDHeaderKey, requestID)

	destination := "decoder"
	if len(prefillEndPoint) > 0 {
		destination = "prefiller"
	}

	// Don't log the full body. Inline base64 images can be MB each.
	if v := logger.V(logging.DEBUG); v.Enabled() {
		kv := []any{
			"requestID", requestID,
			"destination", destination,
			"prefiller", prefillEndPoint,
			logBodyBytesKey, len(modifiedBody),
		}
		v.Info("forwarding request after encoder", kv...)
	}

	if len(prefillEndPoint) > 0 {
		logger.V(logging.DEBUG).Info("using P/D protocol after encoder", "prefiller", prefillEndPoint)
		s.handlePDConnector(w, pdRequest, prefillEndPoint, kvCacheSource, apiType)
		return
	}

	logger.V(logging.DEBUG).Info("no prefiller configured, going directly to decoder after encoder")
	toolFields, _ := captureInferenceToolCallingRequest(pdRequest, apiType, modifiedBody)
	defer toolFields.report(r.Context())
	decodeStart := time.Now()
	decodeWriter, decodeStatus := captureResponseStatus(w)
	decodeReturned := false
	defer recordDecodeAbort(&decodeReturned, decodeStart)
	if !s.forwardDataParallel || !s.dataParallelHandler(decodeWriter, pdRequest, func() {
		_, _ = toolFields.compareValidatedDispatched(modifiedBody)
	}) {
		if kvCacheSource != "" {
			s.decodeWithP2PSource(decodeWriter, pdRequest, kvCacheSource)
		} else {
			_, _ = toolFields.compareValidatedDispatched(modifiedBody)
			s.decoderProxy.ServeHTTP(decodeWriter, pdRequest)
		}
	}
	decodeReturned = true
	metrics.RecordDecodeDuration(time.Since(decodeStart))
	if decodeStatus.failed() {
		metrics.RecordError(metrics.StageDecode)
	}
}
