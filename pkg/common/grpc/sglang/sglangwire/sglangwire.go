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

// Package sglangwire reads and rewrites sglang.api.v1 Generate messages at
// the protobuf wire level. The coordinator and EPP hot paths use it instead of
// proto.Unmarshal, proto.Clone, and proto.Marshal, which decode and copy every
// token ID, media blob, and logprob of a request or streamed token frame.
//
// Measured on an 8k-token request with a 1 MiB image, a full decode takes
// 318 us and 1.16 MB per request, whereas the wire scan takes 66 us and 49 KB.
// A streamed token frame costs 908 ns and 9 allocations with a full decode,
// versus zero allocations in ScanStreamItem.
//
// Field numbers come from the generated protobuf descriptors at init, so a
// renumbered field fails tests instead of reading the wrong field.
package sglangwire

import (
	"errors"
	"fmt"
	"slices"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/protoscan"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
)

// ErrMalformed reports bytes that are not valid protobuf wire data.
var ErrMalformed = protoscan.ErrMalformed

// Field numbers come from the generated descriptors, so a renumbered field
// fails the tests instead of silently corrupting messages.
var (
	reqRid            = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "rid")
	reqText           = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "text")
	reqInputIDs       = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "input_ids")
	reqStream         = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "stream")
	reqSamplingParams = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "sampling_params")
	reqImageData      = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "image_data")
	reqMMHashes       = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "mm_hashes")
	reqVideoData      = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "video_data")
	reqAudioData      = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "audio_data")
	reqExtraKey       = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "extra_key")
	reqBootstrapHost  = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "bootstrap_host")
	reqBootstrapPort  = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "bootstrap_port")
	reqBootstrapRoom  = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "bootstrap_room")
	reqPriority       = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "priority")
	reqCacheSalt      = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "cache_salt")

	optStringOrOne = protoscan.FieldNum(&sglangapiv1.OptionalStringOrList{}, "one")
	optStringValue = protoscan.FieldNum(&sglangapiv1.OptionalString{}, "value")
	optInt64OrOne  = protoscan.FieldNum(&sglangapiv1.OptionalInt64OrList{}, "one")
	optInt64Value  = protoscan.FieldNum(&sglangapiv1.OptionalInt64{}, "value")

	// Every *OrList wrapper encodes its single value as field "one" and its
	// batch as field "many".
	orListOne  = protoscan.FieldNum(&sglangapiv1.StringOrList{}, "one")
	orListMany = protoscan.FieldNum(&sglangapiv1.StringOrList{}, "many")

	mediaItem      = protoscan.FieldNum(&sglangapiv1.MediaInput{}, "item")
	mediaMany      = protoscan.FieldNum(&sglangapiv1.MediaInput{}, "many")
	mediaNested    = protoscan.FieldNum(&sglangapiv1.MediaInput{}, "nested")
	mediaItemsItem = protoscan.FieldNum(&sglangapiv1.MediaItems{}, "items")

	tokenIDsIDs         = protoscan.FieldNum(&sglangapiv1.TokenIds{}, "ids")
	stringListItems     = protoscan.FieldNum(&sglangapiv1.StringList{}, "items")
	samplingMaxNewToken = protoscan.FieldNum(&sglangapiv1.SamplingParams{}, "max_new_tokens")

	itemFrame        = protoscan.FieldNum(&sglangapiv1.GenerateStreamItem{}, "frame")
	itemError        = protoscan.FieldNum(&sglangapiv1.GenerateStreamItem{}, "error")
	respMetaInfo     = protoscan.FieldNum(&sglangapiv1.GenerateResponse{}, "meta_info")
	respIndex        = protoscan.FieldNum(&sglangapiv1.GenerateResponse{}, "index")
	metaPromptTokens = protoscan.FieldNum(&sglangapiv1.GenerateMetaInfo{}, "prompt_tokens")
	metaComplTokens  = protoscan.FieldNum(&sglangapiv1.GenerateMetaInfo{}, "completion_tokens")
	metaFinishReason = protoscan.FieldNum(&sglangapiv1.GenerateMetaInfo{}, "finish_reason")
	metaCachedTokens = protoscan.FieldNum(&sglangapiv1.GenerateMetaInfo{}, "cached_tokens")
)

func init() {
	// The scan assumes every *OrList wrapper uses the same two field numbers.
	for _, m := range []proto.Message{
		&sglangapiv1.TokenIdsOrList{}, &sglangapiv1.StringListOrList{}, &sglangapiv1.SamplingParamsOrList{},
	} {
		if protoscan.FieldNum(m, "one") != orListOne || protoscan.FieldNum(m, "many") != orListMany {
			panic("sglangwire: OrList wrappers disagree on field numbers")
		}
	}
}

// Request holds what the coordinator reads from a GenerateRequest payload.
type Request struct {
	// Stream is the stream field.
	Stream bool
	// Payload is the request payload without any bootstrap_host,
	// bootstrap_port, or bootstrap_room field, which a client may not set.
	// When the client sent none, Payload aliases the scanned payload with no
	// copy. Both legs of a request share it. See LegEnvelope.
	Payload []byte
}

// ScanGenerateRequest checks that payload is well-formed GenerateRequest wire
// data at the top level, reads the stream field, and drops client bootstrap
// fields. Nested messages are left for the SGLang worker to validate.
func ScanGenerateRequest(payload []byte) (Request, error) {
	req := Request{Payload: payload}
	var stripped []byte // set once the first bootstrap field is dropped
	for pos := 0; pos < len(payload); {
		num, wireType, n := protowire.ConsumeTag(payload[pos:])
		if n < 0 {
			return Request{}, ErrMalformed
		}
		m := protowire.ConsumeFieldValue(num, wireType, payload[pos+n:])
		if m < 0 {
			return Request{}, ErrMalformed
		}
		end := pos + n + m
		if num == reqBootstrapHost || num == reqBootstrapPort || num == reqBootstrapRoom {
			if stripped == nil {
				stripped = make([]byte, 0, len(payload)-(n+m))
				stripped = append(stripped, payload[:pos]...)
			}
			pos = end
			continue
		}
		if num == reqStream && wireType == protowire.VarintType {
			v, _ := protowire.ConsumeVarint(payload[pos+n:])
			req.Stream = v != 0
		}
		if stripped != nil {
			stripped = append(stripped, payload[pos:end]...)
		}
		pos = end
	}
	if stripped != nil {
		req.Payload = stripped
	}
	return req, nil
}

// Bootstrap holds the SGLang disaggregation fields to set on a leg request.
type Bootstrap struct {
	// Host is the prefill bootstrap host. An empty Host omits bootstrap_host.
	Host string
	Port int64
	Room int64
}

// LegEnvelope returns the gRPC frame header and the bootstrap suffix that
// make one leg request out of a bootstrap-free payload of payloadLen bytes:
// send head, then the payload, then tail. The payload bytes are not copied,
// so the prefill and decode legs share them. When b is nil, tail is empty.
// head and tail come from one allocation.
func LegEnvelope(payloadLen int, b *Bootstrap) (head, tail []byte, err error) {
	tailLen := 0
	if b != nil {
		tailLen = bootstrapSize(b)
	}
	if payloadLen < 0 || payloadLen > grpcframe.MaxPayloadLen-tailLen {
		return nil, nil, errors.New("sglangwire: GenerateRequest too large for a gRPC frame")
	}
	buf := make([]byte, grpcframe.HeaderLen, grpcframe.HeaderLen+tailLen)
	if b != nil {
		buf = appendBootstrap(buf, b)
	}
	if err := grpcframe.PutHeader(buf, payloadLen+tailLen); err != nil {
		return nil, nil, err
	}
	return buf[:grpcframe.HeaderLen:grpcframe.HeaderLen], buf[grpcframe.HeaderLen:], nil
}

// optionalStringSize returns the encoded size of an OptionalStringOrList
// {one: {value: s}} message body, plus the inner OptionalString body size.
func optionalStringSize(s string) (outer, inner int) {
	inner = protowire.SizeTag(optStringValue) + protowire.SizeBytes(len(s))
	return protowire.SizeTag(optStringOrOne) + protowire.SizeBytes(inner), inner
}

// optionalInt64Size is optionalStringSize for OptionalInt64OrList.
func optionalInt64Size(v int64) (outer, inner int) {
	inner = protowire.SizeTag(optInt64Value) + protowire.SizeVarint(int64Varint(v))
	return protowire.SizeTag(optInt64OrOne) + protowire.SizeBytes(inner), inner
}

func bootstrapSize(b *Bootstrap) int {
	size := optionalInt64FieldSize(reqBootstrapPort, b.Port) + optionalInt64FieldSize(reqBootstrapRoom, b.Room)
	if b.Host != "" {
		outer, _ := optionalStringSize(b.Host)
		size += protowire.SizeTag(reqBootstrapHost) + protowire.SizeBytes(outer)
	}
	return size
}

func optionalInt64FieldSize(num protowire.Number, v int64) int {
	outer, _ := optionalInt64Size(v)
	return protowire.SizeTag(num) + protowire.SizeBytes(outer)
}

func appendBootstrap(out []byte, b *Bootstrap) []byte {
	if b.Host != "" {
		outer, inner := optionalStringSize(b.Host)
		out = protowire.AppendTag(out, reqBootstrapHost, protowire.BytesType)
		out = protoscan.AppendLen(out, outer)
		out = protowire.AppendTag(out, optStringOrOne, protowire.BytesType)
		out = protoscan.AppendLen(out, inner)
		out = protowire.AppendTag(out, optStringValue, protowire.BytesType)
		out = protowire.AppendString(out, b.Host)
	}
	out = appendOptionalInt64(out, reqBootstrapPort, b.Port)
	return appendOptionalInt64(out, reqBootstrapRoom, b.Room)
}

func appendOptionalInt64(out []byte, num protowire.Number, v int64) []byte {
	outer, inner := optionalInt64Size(v)
	out = protowire.AppendTag(out, num, protowire.BytesType)
	out = protoscan.AppendLen(out, outer)
	out = protowire.AppendTag(out, optInt64OrOne, protowire.BytesType)
	out = protoscan.AppendLen(out, inner)
	out = protowire.AppendTag(out, optInt64Value, protowire.VarintType)
	return protowire.AppendVarint(out, int64Varint(v))
}

// int64Varint returns the varint value of a protobuf int64 field, which
// encodes negative values as their two's complement.
func int64Varint(v int64) uint64 {
	return uint64(v) //#nosec G115 -- two's complement is the protobuf int64 encoding
}

// OrList is the wire view of a *OrList wrapper: which oneof member is set and
// the wire bytes of "one". A wrapper with neither member is treated as absent,
// matching GetValue() == nil.
type OrList struct {
	Present bool
	Many    bool
	One     []byte
}

// MediaScan holds what the EPP reads from a MediaInput field.
type MediaScan struct {
	// Present reports a single-request media value: an item, or a flat list
	// (many) with at least one element. An empty flat list (JSON []) is not
	// present.
	Present bool
	// Nested reports the batched MediaItemsList form.
	Nested bool
}

// RequestFields holds the GenerateRequest fields that the EPP reads.
type RequestFields struct {
	Rid, Text, InputIDs, SamplingParams, MMHashes, ExtraKey, CacheSalt OrList
	ImageData, VideoData, AudioData                                    MediaScan
	Stream                                                             bool
}

// ScanRequestFields reads the EPP's GenerateRequest fields at the wire level.
// Byte slices alias payload except when a submessage *OrList field appears more
// than once with "one" set, where its wire bytes are concatenated to preserve
// protobuf merge semantics.
func ScanRequestFields(payload []byte) (RequestFields, error) {
	var f RequestFields
	var imageMany, videoMany, audioMany bool
	err := protoscan.ForEachField(payload, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		if wireType == protowire.VarintType {
			if num == reqStream {
				v, _ := protowire.ConsumeVarint(value)
				f.Stream = v != 0
			}
			return nil
		}
		if wireType != protowire.BytesType {
			return nil
		}
		switch num {
		case reqRid:
			return mergeScalarOrList(&f.Rid, value)
		case reqText:
			return mergeScalarOrList(&f.Text, value)
		case reqExtraKey:
			return mergeScalarOrList(&f.ExtraKey, value)
		case reqCacheSalt:
			return mergeScalarOrList(&f.CacheSalt, value)
		case reqInputIDs:
			return mergeMessageOrList(&f.InputIDs, value)
		case reqSamplingParams:
			return mergeMessageOrList(&f.SamplingParams, value)
		case reqMMHashes:
			return mergeMessageOrList(&f.MMHashes, value)
		case reqImageData:
			return mergeMedia(&f.ImageData, &imageMany, value)
		case reqVideoData:
			return mergeMedia(&f.VideoData, &videoMany, value)
		case reqAudioData:
			return mergeMedia(&f.AudioData, &audioMany, value)
		}
		return nil
	})
	return f, err
}

func mergeScalarOrList(dst *OrList, b []byte) error {
	return protoscan.ForEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		if wireType != protowire.BytesType {
			return nil
		}
		switch num {
		case orListOne:
			*dst = OrList{Present: true, One: value}
		case orListMany:
			*dst = OrList{Present: true, Many: true}
		}
		return nil
	})
}

func mergeMessageOrList(dst *OrList, b []byte) error {
	return protoscan.ForEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		if wireType != protowire.BytesType {
			return nil
		}
		switch num {
		case orListOne:
			switch {
			case !dst.Present || dst.Many || len(dst.One) == 0:
				*dst = OrList{Present: true, One: value}
			case len(value) > 0:
				dst.One = slices.Concat(dst.One, value)
			}
		case orListMany:
			*dst = OrList{Present: true, Many: true}
		}
		return nil
	})
}

func mergeMedia(dst *MediaScan, inMany *bool, b []byte) error {
	return protoscan.ForEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		if wireType != protowire.BytesType {
			return nil
		}
		switch num {
		case mediaItem:
			*dst = MediaScan{Present: true}
			*inMany = false
		case mediaNested:
			*dst = MediaScan{Nested: true}
			*inMany = false
		case mediaMany:
			if !*inMany {
				*dst = MediaScan{}
				*inMany = true
			}
			hasItems, err := scanMediaItems(value)
			if err != nil {
				return err
			}
			if hasItems {
				dst.Present = true
			}
		}
		return nil
	})
}

func scanMediaItems(b []byte) (bool, error) {
	var hasItems bool
	err := protoscan.ForEachField(b, func(num protowire.Number, wireType protowire.Type, _, _ []byte) error {
		if num == mediaItemsItem && wireType == protowire.BytesType {
			hasItems = true
		}
		return nil
	})
	return hasItems, err
}

// ScanTokenIDs decodes a TokenIds message into []uint32 in one allocation. It
// accepts packed and unpacked encodings. An ID outside [0, MaxUint32] fails
// with its position.
func ScanTokenIDs(b []byte) ([]uint32, error) {
	count := 0
	if err := protoscan.ForEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		if num != tokenIDsIDs {
			return nil
		}
		switch wireType {
		case protowire.BytesType:
			count += protoscan.CountVarints(value)
		case protowire.VarintType:
			count++
		}
		return nil
	}); err != nil {
		return nil, err
	}
	ids := make([]uint32, 0, count)
	err := protoscan.ForEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		if num != tokenIDsIDs || (wireType != protowire.BytesType && wireType != protowire.VarintType) {
			return nil
		}
		var err error
		ids, err = protoscan.AppendPackedUint32(ids, wireType, value)
		if err != nil {
			return fmt.Errorf("input_ids: %w", err)
		}
		return nil
	})
	return ids, err
}

// ScanStringList decodes the items of a StringList message.
func ScanStringList(b []byte) ([]string, error) {
	var items []string
	err := protoscan.ForEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		if num == stringListItems && wireType == protowire.BytesType {
			items = append(items, string(value))
		}
		return nil
	})
	return items, err
}

// ScanMaxNewTokens reads SamplingParams.max_new_tokens. ok is false when the
// field is absent.
func ScanMaxNewTokens(b []byte) (v int64, ok bool, err error) {
	err = protoscan.ForEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		if num == samplingMaxNewToken && wireType == protowire.VarintType {
			u, _ := protowire.ConsumeVarint(value)
			v, ok = int64(u), true //#nosec G115 -- two's complement is the protobuf int64 encoding
		}
		return nil
	})
	return v, ok, err
}

// RewritePriority returns frame with every client priority field removed
// and priority set, in one allocation. frame is a full gRPC frame, as is the
// result. GenerateRequest.priority is an optional field, so a zero priority
// is still written.
func RewritePriority(frame []byte, priority int64) ([]byte, error) {
	return protoscan.ReplaceVarintField(frame, reqPriority, int64Varint(priority), false)
}

// StreamItem holds the GenerateStreamItem fields that the coordinator and EPP read.
type StreamItem struct {
	// HasFrame reports that the item holds a GenerateResponse frame.
	HasFrame bool
	// IsError reports that the item holds a GenerateStreamError.
	IsError bool
	// Index is the GenerateResponse index (0 when unset).
	Index uint32
	// HasMetaInfo reports that the frame holds meta_info.
	HasMetaInfo bool
	// Finished reports that meta_info holds a finish_reason, which marks the
	// terminal frame for Index.
	Finished bool
	// PromptTokens is meta_info.prompt_tokens.
	PromptTokens uint32
	// CompletionTokens is meta_info.completion_tokens.
	CompletionTokens uint64
	// CachedTokens is meta_info.cached_tokens.
	CachedTokens uint32
}

// ScanStreamItem reads the coordinator and EPP fields from a GenerateStreamItem
// payload without allocating. It follows protobuf merge rules: the last scalar
// value wins, and repeated message fields merge unless the item oneof switches
// between frame and error.
func ScanStreamItem(payload []byte) (StreamItem, error) {
	var item StreamItem
	err := protoscan.ForEachField(payload, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		if wireType != protowire.BytesType {
			return nil
		}
		// frame and error share the item oneof. Like proto.Unmarshal, a
		// switch to the other member discards the fields read so far.
		switch num {
		case itemError:
			item = StreamItem{IsError: true}
		case itemFrame:
			if item.IsError {
				item = StreamItem{}
			}
			item.HasFrame = true
			return scanResponse(value, &item)
		}
		return nil
	})
	return item, err
}

func scanResponse(b []byte, item *StreamItem) error {
	return protoscan.ForEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		switch {
		case num == respIndex && wireType == protowire.VarintType:
			v, _ := protowire.ConsumeVarint(value)
			item.Index = uint32(v) //#nosec G115 -- uint32 field, protobuf truncates the same way
		case num == respMetaInfo && wireType == protowire.BytesType:
			item.HasMetaInfo = true
			return scanMetaInfo(value, item)
		}
		return nil
	})
}

func scanMetaInfo(b []byte, item *StreamItem) error {
	return protoscan.ForEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) error {
		switch {
		case num == metaFinishReason && wireType == protowire.BytesType:
			item.Finished = true
		case num == metaPromptTokens && wireType == protowire.VarintType:
			v, _ := protowire.ConsumeVarint(value)
			item.PromptTokens = uint32(v) //#nosec G115 -- uint32 field, protobuf truncates the same way
		case num == metaComplTokens && wireType == protowire.VarintType:
			item.CompletionTokens, _ = protowire.ConsumeVarint(value)
		case num == metaCachedTokens && wireType == protowire.VarintType:
			v, _ := protowire.ConsumeVarint(value)
			item.CachedTokens = uint32(v) //#nosec G115 -- uint32 field, protobuf truncates the same way
		}
		return nil
	})
}
