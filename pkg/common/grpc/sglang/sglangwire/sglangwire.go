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
// the protobuf wire level. The coordinator hot paths use it instead of a full
// proto.Unmarshal, proto.Clone, and proto.Marshal, which copy every token ID
// and media byte of a request and every field of a stream frame.
package sglangwire

import (
	"encoding/binary"
	"errors"
	"math"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
)

// ErrMalformed reports bytes that are not valid protobuf wire data.
var ErrMalformed = errors.New("malformed protobuf wire data")

// Field numbers come from the generated descriptors, so a renumbered field
// fails the tests instead of silently corrupting messages.
var (
	reqStream        = fieldNum(&sglangapiv1.GenerateRequest{}, "stream")
	reqBootstrapHost = fieldNum(&sglangapiv1.GenerateRequest{}, "bootstrap_host")
	reqBootstrapPort = fieldNum(&sglangapiv1.GenerateRequest{}, "bootstrap_port")
	reqBootstrapRoom = fieldNum(&sglangapiv1.GenerateRequest{}, "bootstrap_room")
	optStringOrOne   = fieldNum(&sglangapiv1.OptionalStringOrList{}, "one")
	optStringValue   = fieldNum(&sglangapiv1.OptionalString{}, "value")
	optInt64OrOne    = fieldNum(&sglangapiv1.OptionalInt64OrList{}, "one")
	optInt64Value    = fieldNum(&sglangapiv1.OptionalInt64{}, "value")

	itemFrame        = fieldNum(&sglangapiv1.GenerateStreamItem{}, "frame")
	itemError        = fieldNum(&sglangapiv1.GenerateStreamItem{}, "error")
	respMetaInfo     = fieldNum(&sglangapiv1.GenerateResponse{}, "meta_info")
	respIndex        = fieldNum(&sglangapiv1.GenerateResponse{}, "index")
	metaFinishReason = fieldNum(&sglangapiv1.GenerateMetaInfo{}, "finish_reason")
	metaCachedTokens = fieldNum(&sglangapiv1.GenerateMetaInfo{}, "cached_tokens")
)

func fieldNum(m interface {
	ProtoReflect() protoreflect.Message
}, name protoreflect.Name) protowire.Number {
	fd := m.ProtoReflect().Descriptor().Fields().ByName(name)
	if fd == nil {
		panic("sglangwire: unknown field " + string(name))
	}
	return fd.Number()
}

// forEachField calls fn for every top-level field of b with the field's
// number, wire type, value bytes, and full encoding (tag and value).
func forEachField(b []byte, fn func(num protowire.Number, wireType protowire.Type, value, field []byte)) error {
	for len(b) > 0 {
		num, wireType, n := protowire.ConsumeTag(b)
		if n < 0 {
			return ErrMalformed
		}
		m := protowire.ConsumeFieldValue(num, wireType, b[n:])
		if m < 0 {
			return ErrMalformed
		}
		fn(num, wireType, b[n:n+m], b[:n+m])
		b = b[n+m:]
	}
	return nil
}

// ScanGenerateRequest checks that payload is well-formed GenerateRequest wire
// data at the top level and returns the stream field. Nested messages are
// left for the SGLang worker to validate.
func ScanGenerateRequest(payload []byte) (stream bool, err error) {
	err = forEachField(payload, func(num protowire.Number, wireType protowire.Type, value, _ []byte) {
		if num == reqStream && wireType == protowire.VarintType {
			v, _ := protowire.ConsumeVarint(value)
			stream = v != 0
		}
	})
	return stream, err
}

// Bootstrap holds the SGLang disaggregation fields to set on a leg request.
type Bootstrap struct {
	// Host is the prefill bootstrap host. An empty Host omits bootstrap_host.
	Host string
	Port int64
	Room int64
}

// FrameGenerateRequest returns payload as a gRPC frame, with every
// bootstrap_host, bootstrap_port, and bootstrap_room field removed. When b is
// not nil, it then appends b's fields. All other fields are copied byte for
// byte. The result takes one allocation.
func FrameGenerateRequest(payload []byte, b *Bootstrap) ([]byte, error) {
	size := grpcframe.HeaderLen + len(payload)
	if b != nil {
		size += bootstrapSize(b)
	}
	out := make([]byte, grpcframe.HeaderLen, size)
	err := forEachField(payload, func(num protowire.Number, _ protowire.Type, _, field []byte) {
		if num != reqBootstrapHost && num != reqBootstrapPort && num != reqBootstrapRoom {
			out = append(out, field...)
		}
	})
	if err != nil {
		return nil, err
	}
	if b != nil {
		out = appendBootstrap(out, b)
	}
	const maxPayload = min(math.MaxUint32, math.MaxInt-grpcframe.HeaderLen)
	msgLen := len(out) - grpcframe.HeaderLen
	if msgLen > maxPayload {
		return nil, errors.New("sglangwire: GenerateRequest too large for a gRPC frame")
	}
	binary.BigEndian.PutUint32(out[1:grpcframe.HeaderLen], uint32(msgLen)) //#nosec G115 -- guarded by MaxUint32 check above
	return out, nil
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
		out = appendLen(out, outer)
		out = protowire.AppendTag(out, optStringOrOne, protowire.BytesType)
		out = appendLen(out, inner)
		out = protowire.AppendTag(out, optStringValue, protowire.BytesType)
		out = protowire.AppendString(out, b.Host)
	}
	out = appendOptionalInt64(out, reqBootstrapPort, b.Port)
	return appendOptionalInt64(out, reqBootstrapRoom, b.Room)
}

func appendOptionalInt64(out []byte, num protowire.Number, v int64) []byte {
	outer, inner := optionalInt64Size(v)
	out = protowire.AppendTag(out, num, protowire.BytesType)
	out = appendLen(out, outer)
	out = protowire.AppendTag(out, optInt64OrOne, protowire.BytesType)
	out = appendLen(out, inner)
	out = protowire.AppendTag(out, optInt64Value, protowire.VarintType)
	return protowire.AppendVarint(out, int64Varint(v))
}

// appendLen appends a length prefix. n is a non-negative encoded size.
func appendLen(out []byte, n int) []byte {
	return protowire.AppendVarint(out, uint64(n)) //#nosec G115 -- n is a non-negative size
}

// int64Varint returns the varint value of a protobuf int64 field, which
// encodes negative values as their two's complement.
func int64Varint(v int64) uint64 {
	return uint64(v) //#nosec G115 -- two's complement is the protobuf int64 encoding
}

// StreamItem holds the GenerateStreamItem fields the coordinator reads.
type StreamItem struct {
	// IsError reports that the item holds a GenerateStreamError.
	IsError bool
	// Index is the GenerateResponse index (0 when unset).
	Index uint32
	// HasMetaInfo reports that the frame holds meta_info.
	HasMetaInfo bool
	// Finished reports that meta_info holds a finish_reason, which marks the
	// terminal frame for Index.
	Finished bool
	// CachedTokens is meta_info.cached_tokens.
	CachedTokens uint32
}

// ScanStreamItem reads the coordinator's fields from a GenerateStreamItem
// payload without allocating. It follows protobuf merge rules: the last
// scalar value wins, and repeated message fields merge.
func ScanStreamItem(payload []byte) (StreamItem, error) {
	var item StreamItem
	var scanErr error
	err := forEachField(payload, func(num protowire.Number, wireType protowire.Type, value, _ []byte) {
		if wireType != protowire.BytesType {
			return
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
			body, _ := protowire.ConsumeBytes(value)
			if err := scanResponse(body, &item); err != nil {
				scanErr = err
			}
		}
	})
	if err == nil {
		err = scanErr
	}
	return item, err
}

func scanResponse(b []byte, item *StreamItem) error {
	var scanErr error
	err := forEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) {
		switch {
		case num == respIndex && wireType == protowire.VarintType:
			v, _ := protowire.ConsumeVarint(value)
			item.Index = uint32(v) //#nosec G115 -- uint32 field, protobuf truncates the same way
		case num == respMetaInfo && wireType == protowire.BytesType:
			item.HasMetaInfo = true
			body, _ := protowire.ConsumeBytes(value)
			if err := scanMetaInfo(body, item); err != nil {
				scanErr = err
			}
		}
	})
	if err == nil {
		err = scanErr
	}
	return err
}

func scanMetaInfo(b []byte, item *StreamItem) error {
	return forEachField(b, func(num protowire.Number, wireType protowire.Type, value, _ []byte) {
		switch {
		case num == metaFinishReason && wireType == protowire.BytesType:
			item.Finished = true
		case num == metaCachedTokens && wireType == protowire.VarintType:
			v, _ := protowire.ConsumeVarint(value)
			item.CachedTokens = uint32(v) //#nosec G115 -- uint32 field, protobuf truncates the same way
		}
	})
}
