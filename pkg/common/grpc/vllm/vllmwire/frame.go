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

package vllmwire

import (
	"fmt"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/protoscan"
)

// MediaMode selects which media bytes a leg frame carries.
type MediaMode uint8

const (
	// MediaAll forwards every media item verbatim. Use it for the encode
	// leg, or for prefill when no encode leg ran.
	MediaAll MediaMode = iota
	// MediaNone drops every media item. Use it for decode and for the
	// conditional-decode probe.
	MediaNone
	// MediaMetadataOnly keeps uuid, modality, mime_type, and features
	// {identifier, offset, length, mm_hash, is_embed} and drops raw_bytes,
	// data_uri, url, and features.kwargs. Use it for prefill after encode.
	MediaMetadataOnly
)

// LegOptions configures per-stage overrides when framing a GenerateRequest.
type LegOptions struct {
	// RequestID replaces GenerateRequest.request_id when non-empty. vLLM's
	// gRPC server reads the ID from the message, not from x-request-id
	// metadata, so a leg that must run under the sidecar's ID sets it here.
	RequestID string
	// CapSingleToken sets stopping.max_new_tokens to 1 and clears
	// stopping.min_new_tokens (encode and prefill legs). When false, the
	// client's stopping field is forwarded verbatim.
	CapSingleToken bool
	// Media selects which media bytes the frame carries.
	Media MediaMode
	// KVTransferParams sets kv.kv_transfer_params when non-empty.
	KVTransferParams map[string]any
	// ECTransferParams sets kv.ec_transfer_params when non-empty.
	ECTransferParams map[string]any
}

// legPlan holds the sizes FrameSegments computes before it allocates.
type legPlan struct {
	requestIDLen int // encoded override field, 0 when the client's field is reused
	stopBodyLen  int
	emitStop     bool
	kvBodyLen    int
	kvStructLen  int
	ecStructLen  int
	emitKV       bool
	mediaCopy    bool // media bytes are written into the scratch buffer
	mediaLen     int  // total media bytes in the payload
	tailLen      int
	payloadLen   int
}

func (r *Request) plan(opts LegOptions) (legPlan, error) {
	var p legPlan
	var err error

	if opts.RequestID != "" {
		p.requestIDLen = protowire.SizeTag(reqRequestID) + protowire.SizeBytes(len(opts.RequestID))
	}

	p.stopBodyLen, p.emitStop = r.stoppingBodySize(opts.CapSingleToken)
	if p.emitStop {
		p.tailLen += protowire.SizeTag(reqStopping) + protowire.SizeBytes(p.stopBodyLen)
	}

	p.kvBodyLen, p.kvStructLen, p.ecStructLen, err = r.kvBodySize(opts.KVTransferParams, opts.ECTransferParams)
	if err != nil {
		return legPlan{}, err
	}
	p.emitKV = r.hasKV || p.kvBodyLen > 0
	if p.emitKV {
		p.tailLen += protowire.SizeTag(reqKV) + protowire.SizeBytes(p.kvBodyLen)
	}

	switch opts.Media {
	case MediaNone:
	case MediaAll:
		if r.mediaBodies == nil {
			p.mediaLen = r.media.bytes
		} else {
			p.mediaCopy = true
			for _, body := range r.mediaBodies {
				p.mediaLen += mediaFieldSize(len(body))
			}
		}
	case MediaMetadataOnly:
		p.mediaCopy = true
		for i := 0; i < r.media.count; i++ {
			p.mediaLen += mediaFieldSize(mediaMetadataSize(r.mediaBody(i)))
		}
	default:
		return legPlan{}, fmt.Errorf("vllmwire: unknown MediaMode %d", opts.Media)
	}

	ridLen := p.requestIDLen
	if ridLen == 0 {
		ridLen = len(r.requestIDField)
	}
	p.payloadLen = ridLen + r.base.bytes + p.mediaLen + p.tailLen
	if p.payloadLen > grpcframe.MaxPayloadLen {
		return legPlan{}, fmt.Errorf("vllmwire: GenerateRequest: %w", grpcframe.ErrTooLarge)
	}
	return p, nil
}

func mediaFieldSize(bodyLen int) int {
	return protowire.SizeTag(reqMedia) + protowire.SizeBytes(bodyLen)
}

// mediaBody returns the body of media item i, honoring SetMediaBodies.
func (r *Request) mediaBody(i int) []byte {
	if r.mediaBodies != nil {
		return r.mediaBodies[i]
	}
	field := r.media.at(i)
	_, _, n := protowire.ConsumeTag(field)
	body, _ := protowire.ConsumeBytes(field[n:])
	return body
}

// SetMediaBodies replaces the body of every media item in the frames r
// builds, in order. The sidecar uses it after it assigns content UUIDs to
// raw media so prefill and decode see the same uuid as the encoder.
// bodies must have MediaCount entries. Pass nil to restore the client bytes.
func (r *Request) SetMediaBodies(bodies [][]byte) error {
	if bodies != nil && len(bodies) != r.media.count {
		return fmt.Errorf("vllmwire: SetMediaBodies: got %d bodies for %d media items", len(bodies), r.media.count)
	}
	r.mediaBodies = bodies
	return nil
}

// FrameSegments builds the gRPC frame for one worker leg as a list of
// segments. token_ids and text are sub-slices of the scanned payload, never
// copied. With MediaAll and no SetMediaBodies override, media bytes are too.
// Callers pass the segments to a writer that accepts a vectored body, or
// call Frame for one buffer.
func (r *Request) FrameSegments(opts LegOptions) ([][]byte, error) {
	p, err := r.plan(opts)
	if err != nil {
		return nil, err
	}

	scratchLen := grpcframe.HeaderLen + p.requestIDLen + p.tailLen
	if p.mediaCopy {
		scratchLen += p.mediaLen
	}
	buf := make([]byte, grpcframe.HeaderLen, scratchLen)
	if err := grpcframe.PutHeader(buf, p.payloadLen); err != nil {
		return nil, err
	}

	// header, request_id, base spans, media, tail. The segments of the
	// scratch buffer are capped at their length so a caller append cannot
	// write into the next one.
	segCap := 3 + r.base.count
	if p.mediaCopy {
		segCap++
	} else {
		segCap += r.media.count
	}
	segs := make([][]byte, 0, segCap)
	segs = append(segs, buf[:grpcframe.HeaderLen:grpcframe.HeaderLen])

	if p.requestIDLen > 0 {
		mark := len(buf)
		buf = protowire.AppendTag(buf, reqRequestID, protowire.BytesType)
		buf = protowire.AppendString(buf, opts.RequestID)
		segs = append(segs, buf[mark:len(buf):len(buf)])
	} else if len(r.requestIDField) > 0 {
		segs = append(segs, r.requestIDField)
	}

	for i := 0; i < r.base.count; i++ {
		segs = append(segs, r.base.at(i))
	}

	switch {
	case opts.Media == MediaNone:
	case !p.mediaCopy:
		for i := 0; i < r.media.count; i++ {
			segs = append(segs, r.media.at(i))
		}
	default:
		mark := len(buf)
		buf = r.appendMedia(buf, opts.Media)
		if len(buf) > mark {
			segs = append(segs, buf[mark:len(buf):len(buf)])
		}
	}

	mark := len(buf)
	buf = r.appendTail(buf, opts, &p)
	if len(buf) > mark {
		segs = append(segs, buf[mark:len(buf):len(buf)])
	}
	return segs, nil
}

// Frame builds a single contiguous gRPC frame for one worker leg.
func (r *Request) Frame(opts LegOptions) ([]byte, error) {
	segs, err := r.FrameSegments(opts)
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, totalLen(segs))
	for _, seg := range segs {
		out = append(out, seg...)
	}
	return out, nil
}

func (r *Request) appendMedia(dst []byte, mode MediaMode) []byte {
	for i := 0; i < r.media.count; i++ {
		body := r.mediaBody(i)
		dst = protowire.AppendTag(dst, reqMedia, protowire.BytesType)
		if mode == MediaMetadataOnly {
			dst = protoscan.AppendLen(dst, mediaMetadataSize(body))
			dst = appendMediaMetadata(dst, body)
		} else {
			dst = protowire.AppendBytes(dst, body)
		}
	}
	return dst
}

func (r *Request) appendTail(dst []byte, opts LegOptions, p *legPlan) []byte {
	if p.emitStop {
		dst = protowire.AppendTag(dst, reqStopping, protowire.BytesType)
		dst = protoscan.AppendLen(dst, p.stopBodyLen)
		dst = r.appendStoppingBody(dst, opts.CapSingleToken)
	}
	if p.emitKV {
		dst = protowire.AppendTag(dst, reqKV, protowire.BytesType)
		dst = protoscan.AppendLen(dst, p.kvBodyLen)
		dst = r.appendKVBody(dst, opts.KVTransferParams, p.kvStructLen, opts.ECTransferParams, p.ecStructLen)
	}
	return dst
}

func (r *Request) stoppingBodySize(capSingleToken bool) (int, bool) {
	if !capSingleToken {
		return len(r.stoppingBody), r.hasStopping
	}
	kept := 0
	_ = protoscan.ForEachField(r.stoppingBody, func(num protowire.Number, _ protowire.Type, _, field []byte) error {
		if num != stopMaxNewTokens && num != stopMinNewTokens {
			kept += len(field)
		}
		return nil
	})
	return kept + protowire.SizeTag(stopMaxNewTokens) + protowire.SizeVarint(1), true
}

func (r *Request) appendStoppingBody(dst []byte, capSingleToken bool) []byte {
	if !capSingleToken {
		return append(dst, r.stoppingBody...)
	}
	dst, _ = protoscan.AppendWithout(dst, r.stoppingBody, stopMaxNewTokens, stopMinNewTokens)
	dst = protowire.AppendTag(dst, stopMaxNewTokens, protowire.VarintType)
	return protowire.AppendVarint(dst, 1)
}

func (r *Request) kvBodySize(kvParams, ecParams map[string]any) (bodyLen, kvStructLen, ecStructLen int, err error) {
	bodyLen = len(r.kvBody)
	if len(kvParams) > 0 {
		kvStructLen, err = protoscan.SizeStruct(kvParams)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("vllmwire: encode kv_transfer_params: %w", err)
		}
		bodyLen += protowire.SizeTag(kvTransferParams) + protowire.SizeBytes(kvStructLen)
	}
	if len(ecParams) > 0 {
		ecStructLen, err = protoscan.SizeStruct(ecParams)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("vllmwire: encode ec_transfer_params: %w", err)
		}
		bodyLen += protowire.SizeTag(ecTransferParams) + protowire.SizeBytes(ecStructLen)
	}
	return bodyLen, kvStructLen, ecStructLen, nil
}

func (r *Request) appendKVBody(dst []byte, kvParams map[string]any, kvStructLen int, ecParams map[string]any, ecStructLen int) []byte {
	dst = append(dst, r.kvBody...)
	if len(kvParams) > 0 {
		dst = protowire.AppendTag(dst, kvTransferParams, protowire.BytesType)
		dst = protoscan.AppendLen(dst, kvStructLen)
		dst = protoscan.AppendStruct(dst, kvParams)
	}
	if len(ecParams) > 0 {
		dst = protowire.AppendTag(dst, ecTransferParams, protowire.BytesType)
		dst = protoscan.AppendLen(dst, ecStructLen)
		dst = protoscan.AppendStruct(dst, ecParams)
	}
	return dst
}

// EncoderSubFrameOptions configures EncoderSubFrame.
type EncoderSubFrameOptions struct {
	// RequestID sets GenerateRequest.request_id (omitted when empty).
	RequestID string
	// TokenIDs is the sub-prompt for this media item.
	TokenIDs []uint32
	// Item is the media item to encode. Its Body may come from MediaItems
	// or from an earlier RewrittenBody call.
	Item MediaItem
	// UUIDOverride replaces MediaItem.uuid when non-empty.
	UUIDOverride string
	// FeaturesOffset, when set, replaces features.offset.
	FeaturesOffset *uint64
	// ECTransferParams sets kv.ec_transfer_params when non-empty.
	ECTransferParams map[string]any
}

// EncoderSubFrame builds a standalone unary GenerateRequest frame for one
// media item. EC disaggregation fans out one such request per item. The
// frame keeps every client field except request_id, prompt, media,
// stopping, and kv. It sets stopping.max_new_tokens to 1 and keeps the
// sanitized client kv fields (cache_salt, bypass_prefix_cache).
func (r *Request) EncoderSubFrame(opts EncoderSubFrameOptions) ([]byte, error) {
	keptLen := 0
	for i := 0; i < r.base.count; i++ {
		n, err := protoscan.SizeWithout(r.base.at(i), reqText, reqTokenIDs)
		if err != nil {
			return nil, err
		}
		keptLen += n
	}

	requestIDLen := 0
	if opts.RequestID != "" {
		requestIDLen = protowire.SizeTag(reqRequestID) + protowire.SizeBytes(len(opts.RequestID))
	}

	packedLen := 0
	for _, id := range opts.TokenIDs {
		packedLen += protowire.SizeVarint(uint64(id))
	}
	tokenIDsBodyLen := 0
	if packedLen > 0 {
		tokenIDsBodyLen = protowire.SizeTag(tokenIDsIDs) + protowire.SizeBytes(packedLen)
	}
	tokenIDsLen := protowire.SizeTag(reqTokenIDs) + protowire.SizeBytes(tokenIDsBodyLen)

	mediaBody := opts.Item.RewrittenBody(opts.UUIDOverride, opts.FeaturesOffset)
	mediaLen := mediaFieldSize(len(mediaBody))

	stopBodyLen, _ := r.stoppingBodySize(true)
	stopLen := protowire.SizeTag(reqStopping) + protowire.SizeBytes(stopBodyLen)

	kvBodyLen, _, ecStructLen, err := r.kvBodySize(nil, opts.ECTransferParams)
	if err != nil {
		return nil, err
	}
	kvLen := 0
	emitKV := r.hasKV || kvBodyLen > 0
	if emitKV {
		kvLen = protowire.SizeTag(reqKV) + protowire.SizeBytes(kvBodyLen)
	}

	payloadLen := requestIDLen + keptLen + tokenIDsLen + mediaLen + stopLen + kvLen
	if payloadLen > grpcframe.MaxPayloadLen {
		return nil, fmt.Errorf("vllmwire: encoder sub-request: %w", grpcframe.ErrTooLarge)
	}

	frame := make([]byte, grpcframe.HeaderLen, grpcframe.HeaderLen+payloadLen)
	if requestIDLen > 0 {
		frame = protowire.AppendTag(frame, reqRequestID, protowire.BytesType)
		frame = protowire.AppendString(frame, opts.RequestID)
	}
	for i := 0; i < r.base.count; i++ {
		frame, _ = protoscan.AppendWithout(frame, r.base.at(i), reqText, reqTokenIDs)
	}
	frame = protowire.AppendTag(frame, reqTokenIDs, protowire.BytesType)
	frame = protoscan.AppendLen(frame, tokenIDsBodyLen)
	if packedLen > 0 {
		frame = protowire.AppendTag(frame, tokenIDsIDs, protowire.BytesType)
		frame = protoscan.AppendLen(frame, packedLen)
		for _, id := range opts.TokenIDs {
			frame = protowire.AppendVarint(frame, uint64(id))
		}
	}
	frame = protowire.AppendTag(frame, reqMedia, protowire.BytesType)
	frame = protowire.AppendBytes(frame, mediaBody)
	frame = protowire.AppendTag(frame, reqStopping, protowire.BytesType)
	frame = protoscan.AppendLen(frame, stopBodyLen)
	frame = r.appendStoppingBody(frame, true)
	if emitKV {
		frame = protowire.AppendTag(frame, reqKV, protowire.BytesType)
		frame = protoscan.AppendLen(frame, kvBodyLen)
		frame = r.appendKVBody(frame, nil, 0, opts.ECTransferParams, ecStructLen)
	}
	return grpcframe.FinishFrame(frame)
}
