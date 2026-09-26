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
// (ec-example via fanoutEncoderPrimer, ec-nixl via fanoutEncoderCollect)
// supplies its own per-response perItem callback and otherwise reuses
// these helpers verbatim.

package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"golang.org/x/sync/errgroup"
)

// Multimodal content types that need encoder processing.
var mmTypes = map[string]bool{
	reqcommon.PartTypeImageURL:   true,
	reqcommon.PartTypeAudioURL:   true,
	reqcommon.PartTypeVideoURL:   true,
	reqcommon.PartTypeInputAudio: true,
	reqcommon.PartTypeInputImage: true,
}

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

// truncateLongStrings recursively shortens long string values for logging.
func truncateLongStrings(v any, maxLen int) any {
	switch x := v.(type) {
	case string:
		if len(x) > maxLen {
			return fmt.Sprintf("%s...(%d bytes)", x[:maxLen], len(x))
		}
		return x
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, vv := range x {
			out[k] = truncateLongStrings(vv, maxLen)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, vv := range x {
			out[i] = truncateLongStrings(vv, maxLen)
		}
		return out
	default:
		return v
	}
}

// extractMMItems extracts all multimodal content parts from the request:
// chat-completions' messages array, or a Responses input array. Which field to
// walk is gated on apiType rather than presence, since a client could send a
// stray field the other format does not use. Within a turn the parts sit under
// content, and for Responses under an input item's output as well.
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
		logger.V(logging.DEBUG).Info("cannot read request content for multimodal extraction", "error", err)
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

			// vLLM's chat-completions parser primes input_image and image_url
			// through the same map, so an input_image on a chat request is
			// extracted. A chat part type on a Responses request fails the model
			// server's input validation, so the request never reaches a worker
			// and priming it would only fail the fanout first.
			switch partType {
			case reqcommon.PartTypeInputImage:
				if url := reqcommon.MediaPartURL(partMap); url == "" {
					logger.V(logging.DEBUG).Info("skipping input_image with no fetchable URL")
					droppedParts++
					continue
				}
			case reqcommon.PartTypeImageURL, reqcommon.PartTypeAudioURL, reqcommon.PartTypeVideoURL, reqcommon.PartTypeInputAudio:
				if apiType == reqcommon.APITypeResponses {
					logger.V(logging.DEBUG).Info("skipping content part the Responses input union does not define", "type", partType, "apiType", apiType)
					droppedParts++
					continue
				}
			}

			if mmTypes[partType] {
				items = append(items, partMap)
			}
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
			logger.V(logging.DEBUG).Info("skipping turn that does not decode as an object", "error", err)
			droppedTurns++
			continue
		}
		if content, ok := turn[reqcommon.FieldContent].([]any); ok {
			collect(content)
		}
		// A Responses function_call_output carries its parts under output rather
		// than content, and vLLM forwards that array as a tool message's content,
		// so media in it reaches the model like any other part and has to be
		// primed. A computer_call_output's output is an object rather than an
		// array and names no part type the encoder primes.
		if apiType == reqcommon.APITypeResponses {
			if output, ok := turn[reqcommon.FieldOutput].([]any); ok {
				collect(output)
			}
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
func (s *Server) mmItemsForFanout(originalRequest map[string]any, requestID string, apiType reqcommon.APIType) []map[string]any {
	return extractMMItems(s.logger.WithValues("requestID", requestID), originalRequest, apiType)
}

// fanoutEncoder fans out one encoder request per item, in parallel, with
// round-robin over encoderHostPorts. perItem is invoked once per item AFTER
// the encoder has returned a 2xx response; it receives the item's
// positional index and the buffered encoder response. The
// callback may return an error to fail the whole fan-out, or nil to
// accept. perItem may be nil for fire-and-forget primer-style usage.
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
	perItem func(idx int, pw *bufferedResponseWriter) error,
) error {
	if len(encoderHostPorts) == 0 {
		return fmt.Errorf("fanoutEncoder: no encoder hostPorts provided (requestID=%s)", requestID)
	}

	encoderPath := reqcommon.PathChatCompletions
	if apiType == reqcommon.APITypeResponses {
		encoderPath = reqcommon.PathResponses
	}

	s.logger.Info("processing multimodal items", "count", len(items), "requestID", requestID, "encoderHostPorts", encoderHostPorts)

	grp, gctx := errgroup.WithContext(ctx)
	for idx, mmItem := range items {
		hostPort := encoderHostPorts[idx%len(encoderHostPorts)]
		grp.Go(func() error {
			encoderRequest := reqcommon.NewEncoderPrimingBody(originalRequest, mmItem, apiType)

			body, err := json.Marshal(encoderRequest)
			if err != nil {
				err = fmt.Errorf("failed to marshal encoder request for item %d: %w", idx, err)
				s.logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}

			encoderHandler, err := s.encoderProxyHandler(hostPort)
			if err != nil {
				err = fmt.Errorf("failed to get encoder proxy handler for %s: %w", hostPort, err)
				s.logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}

			req, err := http.NewRequestWithContext(gctx, "POST", encoderPath, bytes.NewReader(body))
			if err != nil {
				err = fmt.Errorf("failed to create encoder request for item %d: %w", idx, err)
				s.logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(requestHeaderRequestID, fmt.Sprintf("%s-enc-%d", requestID, idx))

			s.logger.V(logging.DEBUG).Info("sending encoder request", "item", idx, "to", hostPort, "requestID", requestID)

			pw := &bufferedResponseWriter{}
			encoderHandler.ServeHTTP(pw, req)

			// bufferedResponseWriter.Write back-fills 200, so statusCode 0 means
			// the handler wrote nothing at all. isHTTPError treats it as an error.
			if isHTTPError(pw.statusCode) {
				err := fmt.Errorf("encoder request failed for item %d with status %d: %s", idx, pw.statusCode, pw.buffer.String())
				s.logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}

			if perItem != nil {
				if err := perItem(idx, pw); err != nil {
					s.logger.Error(err, "encoder fanout perItem", "item", idx, "requestID", requestID)
					return err
				}
			}

			s.logger.V(logging.DEBUG).Info("encoder request completed", "item", idx, "requestID", requestID)
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
	requestID string,
	apiType reqcommon.APIType,
) {
	// Skip decode-first; the encoder has run and prefill must execute.
	body[requestFieldCacheHitThreshold] = 0

	modifiedBody, err := json.Marshal(body)
	if err != nil {
		s.logger.Error(err, "failed to marshal request after encoder", "requestID", requestID)
		if writeErr := errorJSONInvalid(err, w); writeErr != nil {
			s.logger.Error(writeErr, "failed to send error response to client", "requestID", requestID)
		}
		return
	}

	pdRequest := cloneRequestWithBody(r.Context(), r, modifiedBody)
	pdRequest.Header.Add(requestHeaderRequestID, requestID)

	destination := "decoder"
	if len(prefillEndPoint) > 0 {
		destination = "prefiller"
	}

	// Don't log the full body. Inline base64 images can be MB each.
	if v := s.logger.V(logging.DEBUG); v.Enabled() {
		kv := []any{
			"requestID", requestID,
			"destination", destination,
			"prefiller", prefillEndPoint,
			"bodyBytes", len(modifiedBody),
		}
		if ec, ok := body[requestFieldECTransferParams]; ok {
			kv = append(kv, requestFieldECTransferParams, truncateLongStrings(ec, 64))
		}
		v.Info("forwarding request after encoder", kv...)
	}

	if len(prefillEndPoint) > 0 {
		s.logger.V(logging.DEBUG).Info("using P/D protocol after encoder", "prefiller", prefillEndPoint)
		// The encoder path does not carry a KV cache source: the P2P prefix pull
		// is not wired through encoder disaggregation. The empty source skips the
		// p2p injection regardless of --enable-p2p-pull.
		s.handlePDConnector(w, pdRequest, prefillEndPoint, "", apiType)
		return
	}

	s.logger.V(logging.DEBUG).Info("no prefiller configured, going directly to decoder after encoder")
	if !s.forwardDataParallel || !s.dataParallelHandler(w, pdRequest) {
		s.decoderProxy.ServeHTTP(w, pdRequest)
	}
}
