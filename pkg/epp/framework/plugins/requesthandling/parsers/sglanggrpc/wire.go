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

package sglanggrpc

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
)

// This file reads sglang.api.v1 messages at the protobuf wire level. The EPP
// needs a few fields of a GenerateRequest and of every streamed
// GenerateStreamItem. A full proto.Unmarshal copies every field, media and
// token IDs included, and on the response path runs once per token.

// errMalformed reports bytes that are not valid protobuf wire data.
var errMalformed = errors.New("malformed protobuf wire data")

// Field numbers come from the generated descriptors, so a renumbered field
// fails the tests instead of silently reading the wrong field.
var (
	reqRid            = fieldNum(&sglangapiv1.GenerateRequest{}, "rid")
	reqText           = fieldNum(&sglangapiv1.GenerateRequest{}, "text")
	reqInputIDs       = fieldNum(&sglangapiv1.GenerateRequest{}, "input_ids")
	reqStream         = fieldNum(&sglangapiv1.GenerateRequest{}, "stream")
	reqSamplingParams = fieldNum(&sglangapiv1.GenerateRequest{}, "sampling_params")
	reqImageData      = fieldNum(&sglangapiv1.GenerateRequest{}, "image_data")
	reqMMHashes       = fieldNum(&sglangapiv1.GenerateRequest{}, "mm_hashes")
	reqVideoData      = fieldNum(&sglangapiv1.GenerateRequest{}, "video_data")
	reqAudioData      = fieldNum(&sglangapiv1.GenerateRequest{}, "audio_data")
	reqExtraKey       = fieldNum(&sglangapiv1.GenerateRequest{}, "extra_key")
	reqPriority       = fieldNum(&sglangapiv1.GenerateRequest{}, "priority")
	reqCacheSalt      = fieldNum(&sglangapiv1.GenerateRequest{}, "cache_salt")

	// Every *OrList wrapper encodes its single value as field "one" and its
	// batch as field "many".
	orListOne  = fieldNum(&sglangapiv1.StringOrList{}, "one")
	orListMany = fieldNum(&sglangapiv1.StringOrList{}, "many")

	tokenIDsIDs         = fieldNum(&sglangapiv1.TokenIds{}, "ids")
	stringListItems     = fieldNum(&sglangapiv1.StringList{}, "items")
	samplingMaxNewToken = fieldNum(&sglangapiv1.SamplingParams{}, "max_new_tokens")

	itemFrame        = fieldNum(&sglangapiv1.GenerateStreamItem{}, "frame")
	itemError        = fieldNum(&sglangapiv1.GenerateStreamItem{}, "error")
	respMetaInfo     = fieldNum(&sglangapiv1.GenerateResponse{}, "meta_info")
	metaPromptTokens = fieldNum(&sglangapiv1.GenerateMetaInfo{}, "prompt_tokens")
	metaComplTokens  = fieldNum(&sglangapiv1.GenerateMetaInfo{}, "completion_tokens")
	metaFinishReason = fieldNum(&sglangapiv1.GenerateMetaInfo{}, "finish_reason")
	metaCachedTokens = fieldNum(&sglangapiv1.GenerateMetaInfo{}, "cached_tokens")
)

func fieldNum(m interface{ ProtoReflect() protoreflect.Message }, name protoreflect.Name) protowire.Number {
	fd := m.ProtoReflect().Descriptor().Fields().ByName(name)
	if fd == nil {
		panic("sglanggrpc: unknown field " + string(name))
	}
	return fd.Number()
}

func init() {
	// The scan assumes every *OrList wrapper uses the same two field numbers.
	for _, m := range []interface{ ProtoReflect() protoreflect.Message }{
		&sglangapiv1.TokenIdsOrList{}, &sglangapiv1.StringListOrList{}, &sglangapiv1.SamplingParamsOrList{},
	} {
		if fieldNum(m, "one") != orListOne || fieldNum(m, "many") != orListMany {
			panic("sglanggrpc: OrList wrappers disagree on field numbers")
		}
	}
}

// forEachField calls fn for every top-level field of b with its number, wire
// type, and value bytes (the length-delimited content, or the varint bytes).
func forEachField(b []byte, fn func(num protowire.Number, wireType protowire.Type, value []byte) error) error {
	for len(b) > 0 {
		num, wireType, n := protowire.ConsumeTag(b)
		if n < 0 {
			return errMalformed
		}
		m := protowire.ConsumeFieldValue(num, wireType, b[n:])
		if m < 0 {
			return errMalformed
		}
		value := b[n : n+m]
		if wireType == protowire.BytesType {
			value, _ = protowire.ConsumeBytes(value)
		}
		if err := fn(num, wireType, value); err != nil {
			return err
		}
		b = b[n+m:]
	}
	return nil
}

// orListScan is the wire view of a *OrList wrapper: which member is set and
// the bytes of "one". A wrapper with neither member is treated as absent, as
// GetValue() == nil is.
type orListScan struct {
	present bool
	many    bool
	one     []byte
}

func scanOrList(b []byte) (orListScan, error) {
	var s orListScan
	err := forEachField(b, func(num protowire.Number, wireType protowire.Type, value []byte) error {
		switch {
		case num == orListOne && wireType == protowire.BytesType:
			s.present, s.many, s.one = true, false, value
		case num == orListMany && wireType == protowire.BytesType:
			s.present, s.many = true, true
		}
		return nil
	})
	return s, err
}

// generateRequestScan holds the GenerateRequest fields the EPP reads.
type generateRequestScan struct {
	rid, text, inputIDs, samplingParams, mmHashes, extraKey, cacheSalt orListScan
	stream                                                             bool
	hasImage, hasVideo, hasAudio                                       bool
}

// scanGenerateRequest reads payload at the top level and one level into the
// *OrList wrappers. It copies nothing: every byte slice aliases payload.
func scanGenerateRequest(payload []byte) (generateRequestScan, error) {
	var s generateRequestScan
	var scanErr error
	scan := func(dst *orListScan, value []byte) {
		if v, err := scanOrList(value); err != nil {
			scanErr = err
		} else {
			*dst = v
		}
	}
	err := forEachField(payload, func(num protowire.Number, wireType protowire.Type, value []byte) error {
		if wireType == protowire.VarintType {
			if num == reqStream {
				v, _ := protowire.ConsumeVarint(value)
				s.stream = v != 0
			}
			return nil
		}
		if wireType != protowire.BytesType {
			return nil
		}
		switch num {
		case reqRid:
			scan(&s.rid, value)
		case reqText:
			scan(&s.text, value)
		case reqInputIDs:
			scan(&s.inputIDs, value)
		case reqSamplingParams:
			scan(&s.samplingParams, value)
		case reqMMHashes:
			scan(&s.mmHashes, value)
		case reqExtraKey:
			scan(&s.extraKey, value)
		case reqCacheSalt:
			scan(&s.cacheSalt, value)
		case reqImageData:
			s.hasImage = true
		case reqVideoData:
			s.hasVideo = true
		case reqAudioData:
			s.hasAudio = true
		}
		return scanErr
	})
	if err == nil {
		err = scanErr
	}
	return s, err
}

// scanTokenIDs decodes a TokenIds message into []uint32 in one allocation.
// It accepts packed and unpacked encodings. An ID outside [0, MaxUint32]
// fails with its position, as the typed conversion did.
func scanTokenIDs(b []byte) ([]uint32, error) {
	count := 0
	if err := forEachField(b, func(num protowire.Number, wireType protowire.Type, value []byte) error {
		if num != tokenIDsIDs {
			return nil
		}
		switch wireType {
		case protowire.BytesType:
			count += countVarints(value)
		case protowire.VarintType:
			count++
		}
		return nil
	}); err != nil {
		return nil, err
	}
	ids := make([]uint32, 0, count)
	add := func(v uint64) error {
		if v > math.MaxUint32 {
			return fmt.Errorf("input_ids[%d] out of uint32 range: %d", len(ids), int64(v)) //#nosec G115 -- reports the int64 the client sent
		}
		ids = append(ids, uint32(v))
		return nil
	}
	err := forEachField(b, func(num protowire.Number, wireType protowire.Type, value []byte) error {
		if num != tokenIDsIDs {
			return nil
		}
		switch wireType {
		case protowire.BytesType:
			for len(value) > 0 {
				v, n := protowire.ConsumeVarint(value)
				if n < 0 {
					return errMalformed
				}
				if err := add(v); err != nil {
					return err
				}
				value = value[n:]
			}
		case protowire.VarintType:
			v, _ := protowire.ConsumeVarint(value)
			return add(v)
		}
		return nil
	})
	return ids, err
}

// countVarints counts the varints in a packed field: one per byte with the
// continuation bit clear.
func countVarints(b []byte) int {
	n := 0
	for _, c := range b {
		if c&0x80 == 0 {
			n++
		}
	}
	return n
}

// scanStringList decodes a StringList message's items.
func scanStringList(b []byte) ([]string, error) {
	var items []string
	err := forEachField(b, func(num protowire.Number, wireType protowire.Type, value []byte) error {
		if num == stringListItems && wireType == protowire.BytesType {
			items = append(items, string(value))
		}
		return nil
	})
	return items, err
}

// scanMaxNewTokens reads SamplingParams.max_new_tokens. ok is false when the
// field is absent.
func scanMaxNewTokens(b []byte) (v int64, ok bool, err error) {
	err = forEachField(b, func(num protowire.Number, wireType protowire.Type, value []byte) error {
		if num == samplingMaxNewToken && wireType == protowire.VarintType {
			u, _ := protowire.ConsumeVarint(value)
			v, ok = int64(u), true //#nosec G115 -- two's complement is the protobuf int64 encoding
		}
		return nil
	})
	return v, ok, err
}

// framePriority returns a gRPC frame of payload with every priority field
// removed and priority appended, in one allocation.
func framePriority(payload []byte, priority int64) ([]byte, error) {
	value := uint64(priority) //#nosec G115 -- two's complement is the protobuf int64 encoding
	suffixLen := protowire.SizeTag(reqPriority) + protowire.SizeVarint(value)
	out := make([]byte, grpcframe.HeaderLen, grpcframe.HeaderLen+len(payload)+suffixLen)
	for b := payload; len(b) > 0; {
		num, wireType, n := protowire.ConsumeTag(b)
		if n < 0 {
			return nil, errMalformed
		}
		m := protowire.ConsumeFieldValue(num, wireType, b[n:])
		if m < 0 {
			return nil, errMalformed
		}
		if num != reqPriority {
			out = append(out, b[:n+m]...)
		}
		b = b[n+m:]
	}
	out = protowire.AppendTag(out, reqPriority, protowire.VarintType)
	out = protowire.AppendVarint(out, value)
	msgLen := len(out) - grpcframe.HeaderLen
	const maxPayload = min(math.MaxUint32, math.MaxInt-grpcframe.HeaderLen)
	if msgLen > maxPayload {
		return nil, errors.New("GenerateRequest too large for a gRPC frame")
	}
	binary.BigEndian.PutUint32(out[1:grpcframe.HeaderLen], uint32(msgLen)) //#nosec G115 -- guarded by maxPayload check above
	return out, nil
}

// streamItemScan holds the GenerateStreamItem fields the EPP reads.
type streamItemScan struct {
	// hasFrame reports a frame item; false for an error item.
	hasFrame bool
	// finished reports that meta_info holds a finish_reason: the terminal
	// frame of its index.
	finished                   bool
	promptTokens, cachedTokens uint32
	completionTokens           uint64
}

// scanStreamItem reads a GenerateStreamItem payload without allocating. It
// follows protobuf merge rules: the last member of the item oneof wins and
// discards the fields of the other.
func scanStreamItem(payload []byte) (streamItemScan, error) {
	var s streamItemScan
	var scanErr error
	err := forEachField(payload, func(num protowire.Number, wireType protowire.Type, value []byte) error {
		if wireType != protowire.BytesType {
			return nil
		}
		switch num {
		case itemError:
			s = streamItemScan{}
		case itemFrame:
			s = streamItemScan{hasFrame: true}
			scanErr = forEachField(value, func(num protowire.Number, wireType protowire.Type, value []byte) error {
				if num == respMetaInfo && wireType == protowire.BytesType {
					return scanMetaInfo(value, &s)
				}
				return nil
			})
		}
		return scanErr
	})
	if err == nil {
		err = scanErr
	}
	return s, err
}

func scanMetaInfo(b []byte, s *streamItemScan) error {
	return forEachField(b, func(num protowire.Number, wireType protowire.Type, value []byte) error {
		switch {
		case num == metaFinishReason && wireType == protowire.BytesType:
			s.finished = true
		case num == metaPromptTokens && wireType == protowire.VarintType:
			v, _ := protowire.ConsumeVarint(value)
			s.promptTokens = uint32(v) //#nosec G115 -- uint32 field; protobuf truncates the same way
		case num == metaComplTokens && wireType == protowire.VarintType:
			s.completionTokens, _ = protowire.ConsumeVarint(value)
		case num == metaCachedTokens && wireType == protowire.VarintType:
			v, _ := protowire.ConsumeVarint(value)
			s.cachedTokens = uint32(v) //#nosec G115 -- uint32 field; protobuf truncates the same way
		}
		return nil
	})
}
