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

// This file implements the proxy side of vLLM's ECMooncakeConnector, following
// examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py. The encoder
// pushes each embedding to the consumer named by consumer_zmq under a
// proxy-assigned transfer_id. The consumer is the prefiller, or the local
// decoder when the request has no prefiller.

package proxy

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

// ec_transfer_params fields of vLLM's ECMooncakeConnector.
const (
	ecMooncakeFieldConsumerZMQ = "consumer_zmq"
	ecMooncakeFieldItems       = "ec_items"
	ecMooncakeFieldTransferID  = "transfer_id"
	ecMooncakeFieldMMHash      = "mm_hash"
	ecMooncakeFieldMetadata    = "metadata"
)

// partFieldUUID is the content part field vLLM keys a multimodal item's hash on.
const partFieldUUID = "uuid"

// ecMooncakeEmbedsTypes maps each chat media part type to the metadata-only
// embeds part it is rewritten to once the encoder has pushed its embedding.
var ecMooncakeEmbedsTypes = map[string]string{
	reqcommon.PartTypeImageURL:   "image_embeds",
	reqcommon.PartTypeAudioURL:   "audio_embeds",
	reqcommon.PartTypeInputAudio: "audio_embeds",
	reqcommon.PartTypeVideoURL:   "video_embeds",
}

// ecMooncakeItem is one multimodal item's state across the encoder fan-out.
type ecMooncakeItem struct {
	partType   string
	transferID string
	// uuid is set only for an item whose part can be rewritten to embeds. It is
	// sent to the encoder so both sides key the item the same way. Other items
	// are sent without one so the consumer's content hash matches the encoder's.
	uuid string
	// ecMMHash and reported are the encoder's ec_transfer_params entry for the
	// item; ecMMHash is empty when the encoder reported none.
	ecMMHash string
	reported map[string]any
}

func (it *ecMooncakeItem) metadata() map[string]any {
	md, _ := it.reported[ecMooncakeFieldMetadata].(map[string]any)
	return md
}

func (it *ecMooncakeItem) transferItem() map[string]any {
	return map[string]any{ecMooncakeFieldMMHash: it.ecMMHash, ecMooncakeFieldTransferID: it.transferID}
}

// ecMooncakeConsumerZMQ returns the TP-rank-0 control address of the EC
// consumer for a request routed to prefillEndPoint.
func (s *Server) ecMooncakeConsumerZMQ(prefillEndPoint string) (string, error) {
	host := s.config.ECMooncakePodIP
	if prefillEndPoint != "" {
		host = extractHost(prefillEndPoint)
	}
	if host == "" {
		return "", errors.New("no ec-mooncake consumer address: request has no prefiller and --ec-mooncake-pod-ip is unset")
	}
	return "tcp://" + net.JoinHostPort(host, strconv.Itoa(s.config.ECMooncakePort)), nil
}

// ecMooncakeContentUUID derives an item's cache key from its content, so a
// repeat of the same media across requests still hits the consumer's cache.
func ecMooncakeContentUUID(part map[string]any) string {
	payload := []byte(reqcommon.MediaPartURL(part))
	if len(payload) == 0 {
		payload, _ = json.Marshal(part) // decoded from JSON, so it re-encodes
	}
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])
}

// fanoutEncoderMooncake sends one encoder request per multimodal item, naming
// the consumer for prefillEndPoint as the push destination, and records what
// each encoder reported.
func (s *Server) fanoutEncoderMooncake(
	ctx context.Context,
	originalRequest map[string]any,
	encoderHostPorts []string,
	requestID string,
	apiType reqcommon.APIType,
	prefillEndPoint string,
) ([]ecMooncakeItem, error) {
	logger := log.FromContext(ctx)
	parts := s.mmItemsForFanout(ctx, originalRequest, requestID, apiType)
	if len(parts) == 0 {
		logger.V(logging.DEBUG).Info("no multimodal items, skipping encoder", "requestID", requestID)
		return nil, nil
	}

	consumerZMQ, err := s.ecMooncakeConsumerZMQ(prefillEndPoint)
	if err != nil {
		return nil, err
	}
	logger.V(logging.DEBUG).Info("ec-mooncake consumer", "requestID", requestID, "consumer", consumerZMQ)

	items := make([]ecMooncakeItem, len(parts))
	for idx, part := range parts {
		partType, _ := part[reqcommon.FieldType].(string)
		items[idx].partType = partType
		items[idx].transferID = newUUID()
		if _, ok := ecMooncakeEmbedsTypes[partType]; !ok || apiType != reqcommon.APITypeChatCompletions {
			continue
		}
		id, _ := part[partFieldUUID].(string)
		if id == "" {
			id = ecMooncakeContentUUID(part)
			part[partFieldUUID] = id
		}
		items[idx].uuid = id
	}

	prepare := func(idx int, body map[string]any) {
		body[reqcommon.FieldECTransferParams] = map[string]any{
			ecMooncakeFieldConsumerZMQ: consumerZMQ,
			ecMooncakeFieldItems: []map[string]any{
				{ecMooncakeFieldTransferID: items[idx].transferID},
			},
		}
	}

	// Each goroutine writes only items[idx], so no lock is needed.
	perItem := func(idx int, pw *bufferedResponseWriter) error {
		var resp struct {
			ECTransferParams map[string]any `json:"ec_transfer_params"`
		}
		dec := json.NewDecoder(bytes.NewReader(pw.bodyBytes()))
		dec.UseNumber()
		if err := dec.Decode(&resp); err != nil {
			return fmt.Errorf("failed to parse encoder response for item %d: %w", idx, err)
		}
		params := resp.ECTransferParams
		if len(params) == 0 {
			logger.V(logging.DEBUG).Info("encoder response carries no ec_transfer_params",
				"item", idx, "requestID", requestID)
			return nil
		}
		key := items[idx].uuid
		if _, ok := params[key]; !ok || key == "" {
			if len(params) != 1 {
				return fmt.Errorf("encoder metadata for item %d cannot be matched: %d ec_transfer_params entries", idx, len(params))
			}
			for k := range params {
				key = k
			}
		}
		reported, ok := params[key].(map[string]any)
		if !ok {
			return fmt.Errorf("encoder ec_transfer_params entry for item %d is %T, want an object", idx, params[key])
		}
		items[idx].ecMMHash = key
		items[idx].reported = reported
		return nil
	}

	if err := s.fanoutEncoder(ctx, originalRequest, parts, encoderHostPorts, requestID, apiType, prepare, perItem); err != nil {
		return nil, err
	}
	return items, nil
}

// applyECMooncakeItems writes the encoder results into the consumer request:
// each rewritable media part becomes a metadata-only embeds part, and
// ec_transfer_params carries every reported entry plus the ec_items the
// consumer matches pushes by. It returns the number of rewritten parts and
// of ec_items entries.
func applyECMooncakeItems(body map[string]any, items []ecMooncakeItem, apiType reqcommon.APIType) (int, int, error) {
	params := make(map[string]any)
	for i := range items {
		if items[i].ecMMHash != "" {
			params[items[i].ecMMHash] = items[i].reported
		}
	}

	var transferItems []map[string]any
	rewritten := 0
	if apiType == reqcommon.APITypeChatCompletions {
		var err error
		transferItems, rewritten, err = rewriteECMooncakeMessages(body, items)
		if err != nil {
			return 0, 0, err
		}
	} else {
		for i := range items {
			if items[i].ecMMHash != "" {
				transferItems = append(transferItems, items[i].transferItem())
			}
		}
	}

	if len(transferItems) > 0 {
		params[ecMooncakeFieldItems] = transferItems
	}
	if len(params) > 0 {
		body[reqcommon.FieldECTransferParams] = params
	}
	return rewritten, len(transferItems), nil
}

// rewriteECMooncakeMessages replaces each media part of a chat request with an
// embeds part sized by the metadata its encoder reported. A part with no
// metadata is left as sent and gets no ec_items entry, so the consumer
// processes its media itself. Audio processors may consume placeholders from
// other modalities, so when an audio part is present and any part lacks
// metadata, every part stays raw and is tagged with its uuid instead.
func rewriteECMooncakeMessages(body map[string]any, items []ecMooncakeItem) ([]map[string]any, int, error) {
	hasAudio, anyMissing := false, false
	for i := range items {
		if modality, _ := reqcommon.PartModality(items[i].partType); modality == reqcommon.ModalityAudio {
			hasAudio = true
		}
		if items[i].ecMMHash == "" || len(items[i].metadata()) == 0 {
			anyMissing = true
		}
	}
	rawAudio := hasAudio && anyMissing

	messages, err := requestMessages(body)
	if err != nil {
		return nil, 0, err
	}

	var transferItems []map[string]any
	rewritten, idx := 0, 0
	changed := false
	out := make([]json.RawMessage, len(messages))
	for m, raw := range messages {
		out[m] = raw
		var turn map[string]any
		// Same decoding as extractMMItems, so the walk visits the parts it
		// extracted, in the same order.
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.UseNumber()
		if dec.Decode(&turn) != nil {
			continue
		}
		turnChanged := false
		for _, array := range reqcommon.ItemPartArrays(turn, reqcommon.APITypeChatCompletions) {
			for j, part := range array.Parts {
				partMap, ok := part.(map[string]any)
				if !ok || !isFanoutMediaPart(partMap) {
					continue
				}
				if idx >= len(items) {
					return nil, 0, fmt.Errorf("request has more media parts than the %d encoded items", len(items))
				}
				it := &items[idx]
				idx++
				if it.ecMMHash == "" || it.uuid == "" {
					if it.ecMMHash != "" {
						transferItems = append(transferItems, it.transferItem())
					}
					continue
				}
				if rawAudio {
					partMap[partFieldUUID] = it.uuid
					transferItems = append(transferItems, it.transferItem())
					turnChanged = true
					continue
				}
				md := it.metadata()
				if len(md) == 0 {
					continue
				}
				embedsType := ecMooncakeEmbedsTypes[it.partType]
				array.Parts[j] = map[string]any{
					reqcommon.FieldType: embedsType,
					embedsType:          flattenECMetadata(md),
					partFieldUUID:       it.uuid,
				}
				transferItems = append(transferItems, it.transferItem())
				rewritten++
				turnChanged = true
			}
		}
		if turnChanged {
			b, err := json.Marshal(turn)
			if err != nil {
				return nil, 0, err
			}
			out[m] = b
			changed = true
		}
	}
	if changed {
		body[reqcommon.FieldMessages] = out
	}
	return transferItems, rewritten, nil
}

// isFanoutMediaPart applies extractMMItems' chat-completions rule for which
// parts become encoder items.
func isFanoutMediaPart(part map[string]any) bool {
	partType, ok := part[reqcommon.FieldType].(string)
	if !ok {
		return false
	}
	if _, isMedia := reqcommon.PartModality(partType); !isMedia {
		return false
	}
	return partType != reqcommon.PartTypeInputImage || reqcommon.MediaPartURL(part) != ""
}

// flattenECMetadata flattens each reported per-item list one level, keeping
// the per-item vector shape the consumer stacks.
func flattenECMetadata(md map[string]any) map[string]any {
	out := make(map[string]any, len(md))
	for k, v := range md {
		list, ok := v.([]any)
		if !ok {
			out[k] = v
			continue
		}
		flat := make([]any, 0, len(list))
		for _, x := range list {
			if inner, ok := x.([]any); ok {
				flat = append(flat, inner...)
			} else {
				flat = append(flat, x)
			}
		}
		out[k] = flat
	}
	return out
}

// handleECMooncake fans out per-item encoder requests that push embeddings to
// the consumer over Mooncake, rewrites the request to reference them, and
// hands off to the configured P/D connector.
func (s *Server) handleECMooncake(w http.ResponseWriter, r *http.Request, prefillEndPoint string, encodeEndPoints []string, kvCacheSource string, apiType reqcommon.APIType) {
	logger := log.FromContext(r.Context())
	logger.V(logging.DEBUG).Info("running EC-Mooncake protocol", "prefiller", prefillEndPoint, "encoderCount", len(encodeEndPoints))

	_, body, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}

	reqUUID, err := uuid.NewUUID()
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}
	requestID := reqUUID.String()

	if len(encodeEndPoints) > 0 {
		fail := func(err error) {
			metrics.RecordError(metrics.StageEncode)
			logger.Error(err, "encoder processing failed", "requestID", requestID)
			if err := errorBadGateway(err, w); err != nil {
				logger.Error(err, "failed to send error response to client")
			}
		}

		encodeStart := time.Now()
		items, err := s.fanoutEncoderMooncake(r.Context(), body, encodeEndPoints, requestID, apiType, prefillEndPoint)
		if err != nil {
			fail(err)
			return
		}
		// No items means there was no multimodal input, so no encoder ran.
		if len(items) > 0 {
			metrics.RecordEncodeDuration(time.Since(encodeStart))
			rewritten, transfers, err := applyECMooncakeItems(body, items, apiType)
			if err != nil {
				fail(fmt.Errorf("failed to rewrite request after encoder: %w", err))
				return
			}
			if transfers < len(items) {
				logger.Info("warning: some multimodal items have no ec-mooncake transfer; the consumer processes them itself",
					"requestID", requestID, "transfers", transfers, "items", len(items))
			}
			logger.V(logging.DEBUG).Info("ec-mooncake encoder stage complete",
				"requestID", requestID, "items", len(items),
				"rewritten", rewritten, "transfers", transfers)
		}
	}

	s.runPDPipeline(w, r, body, prefillEndPoint, kvCacheSource, requestID, apiType)
}
