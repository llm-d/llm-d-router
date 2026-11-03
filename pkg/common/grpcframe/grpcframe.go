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

package grpcframe

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"

	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

const (
	// HeaderLen is the length in bytes of a gRPC length-prefixed message header:
	// a 1-byte compression flag followed by a 4-byte big-endian message length.
	HeaderLen = 5
	// MediaType is the base HTTP Content-Type media type for gRPC over HTTP/2.
	MediaType = "application/grpc"
	// HeaderStatus names the gRPC status code header/trailer.
	HeaderStatus = "grpc-status"
	// HeaderMessage names the percent-encoded gRPC error message header/trailer.
	HeaderMessage = "grpc-message"
	// HeaderStatusDetailsBin names the base64-encoded google.rpc.Status trailer.
	HeaderStatusDetailsBin = "grpc-status-details-bin"
	// StatusOK is the wire representation of codes.OK ("0").
	StatusOK = "0"
)

var trailerHeaders = [...]string{
	HeaderStatus,
	HeaderMessage,
	HeaderStatusDetailsBin,
}

// IsContentType reports whether contentType is a gRPC media type
// ("application/grpc" or "application/grpc+<subtype>").
func IsContentType(contentType string) bool {
	mediaType, _, _ := strings.Cut(contentType, ";")
	mediaType = strings.ToLower(strings.TrimSpace(mediaType))
	return mediaType == MediaType || strings.HasPrefix(mediaType, MediaType+"+")
}

// Status extracts the gRPC status code string from an HTTP header or trailer
// map, including unannounced HTTP/2 trailers prefixed with http.TrailerPrefix
// by httputil.ReverseProxy.
func Status(h http.Header) (string, bool) {
	if len(h) == 0 {
		return "", false
	}
	if v := firstNonEmpty(h.Values(HeaderStatus)); v != "" {
		return v, true
	}
	if v := firstNonEmpty(h.Values(http.TrailerPrefix + HeaderStatus)); v != "" {
		return v, true
	}
	for k, vv := range h {
		trimmed := strings.TrimPrefix(k, http.TrailerPrefix)
		if strings.EqualFold(trimmed, HeaderStatus) {
			if v := firstNonEmpty(vv); v != "" {
				return v, true
			}
		}
	}
	return "", false
}

// HasErrorStatus reports whether h carries a non-OK gRPC status code.
func HasErrorStatus(h http.Header) bool {
	code, ok := Status(h)
	return ok && code != StatusOK
}

// CopyTrailers copies gRPC status trailers from src onto dst using
// http.TrailerPrefix so net/http emits them as HTTP/2 trailers after body writes.
func CopyTrailers(dst, src http.Header) {
	for _, key := range trailerHeaders {
		for _, val := range src.Values(key) {
			dst.Add(http.TrailerPrefix+key, val)
		}
		for _, val := range src.Values(http.TrailerPrefix + key) {
			dst.Add(http.TrailerPrefix+key, val)
		}
	}
}

// WriteError writes a trailers-only gRPC error response to w.
func WriteError(w http.ResponseWriter, code codes.Code, msg string) {
	h := w.Header()
	h.Set(reqcommon.HeaderContentType, MediaType)
	h.Set(HeaderStatus, strconv.FormatUint(uint64(code), 10))
	if msg != "" {
		h.Set(HeaderMessage, encodeStatusMessage(msg))
	}
	w.WriteHeader(http.StatusOK)
}

func firstNonEmpty(vals []string) string {
	for _, v := range vals {
		if trimmed := strings.TrimSpace(v); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

// encodeStatusMessage percent-encodes bytes outside unescaped printable ASCII
// per the gRPC over HTTP/2 specification.
func encodeStatusMessage(msg string) string {
	const hexUpper = "0123456789ABCDEF"
	var b strings.Builder
	b.Grow(len(msg))
	for i := 0; i < len(msg); i++ {
		c := msg[i]
		if c >= 0x20 && c <= 0x7E && c != '%' {
			b.WriteByte(c)
			continue
		}
		b.WriteByte('%')
		b.WriteByte(hexUpper[c>>4])
		b.WriteByte(hexUpper[c&0x0F])
	}
	return b.String()
}

// ReadFrame reads a single uncompressed gRPC frame from r and returns its
// payload bytes. It returns io.EOF when r ends cleanly on a frame boundary.
// When maxPayloadBytes > 0, frames declaring a larger payload are rejected
// before allocating the payload buffer.
func ReadFrame(r io.Reader, maxPayloadBytes int) ([]byte, error) {
	var header [HeaderLen]byte
	_, err := io.ReadFull(r, header[:])
	if errors.Is(err, io.EOF) {
		return nil, io.EOF
	}
	if err != nil {
		return nil, fmt.Errorf("read gRPC frame header: %w", err)
	}
	if header[0] != 0 {
		return nil, errors.New("compressed gRPC payload is not supported")
	}
	msgLen := binary.BigEndian.Uint32(header[1:HeaderLen])
	if maxPayloadBytes > 0 && uint64(msgLen) > uint64(maxPayloadBytes) {
		return nil, fmt.Errorf("gRPC frame payload %d bytes exceeds limit %d", msgLen, maxPayloadBytes)
	}
	if uint64(msgLen) > uint64(math.MaxInt) {
		return nil, fmt.Errorf("gRPC frame payload %d bytes overflows int", msgLen)
	}
	payload := make([]byte, int(msgLen)) //#nosec G115 -- guarded by MaxInt check above
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, fmt.Errorf("read gRPC frame payload: %w", err)
	}
	return payload, nil
}

// ParsePayload extracts the message payload from a single uncompressed gRPC frame.
func ParsePayload(data []byte) ([]byte, error) {
	if len(data) < HeaderLen {
		return nil, fmt.Errorf("invalid gRPC frame: expected at least %d bytes for header, got %d", HeaderLen, len(data))
	}

	if data[0] != 0 {
		// TODO(#895): handle compressed payload.
		return nil, errors.New("compressed gRPC payload is not supported")
	}
	msgLen := binary.BigEndian.Uint32(data[1:HeaderLen])

	// Compare in uint64 so HeaderLen+msgLen cannot overflow when msgLen is near
	// math.MaxUint32, which would wrap the check and let the slice below panic.
	if uint64(len(data)) < uint64(HeaderLen)+uint64(msgLen) {
		return nil, fmt.Errorf("incomplete gRPC payload: header indicates %d bytes, but only %d bytes are available", msgLen, len(data)-HeaderLen)
	}
	return data[HeaderLen : HeaderLen+msgLen], nil
}

// ParsePayloads extracts the message payload of every gRPC frame in data.
// When a frame cannot be parsed, the payloads of the frames before it are
// returned together with the error so callers can still inspect earlier frames.
func ParsePayloads(data []byte) ([][]byte, error) {
	var payloads [][]byte
	for len(data) > 0 {
		payload, err := ParsePayload(data)
		if err != nil {
			return payloads, err
		}
		payloads = append(payloads, payload)
		data = data[HeaderLen+len(payload):]
	}
	return payloads, nil
}

// SplitCompleteFrames splits data into contiguous whole uncompressed gRPC frames
// and any trailing partial frame bytes that require further stream chunks.
// It returns an error if a frame specifies a non-zero compression flag.
func SplitCompleteFrames(data []byte) (complete, rest []byte, err error) {
	offset := 0
	for offset < len(data) {
		rem := data[offset:]
		if len(rem) < HeaderLen {
			break
		}
		if rem[0] != 0 {
			return data[:offset], nil, errors.New("compressed gRPC payload is not supported")
		}
		msgLen := binary.BigEndian.Uint32(rem[1:HeaderLen])
		if uint64(len(rem)) < uint64(HeaderLen)+uint64(msgLen) {
			break
		}
		payload := rem[HeaderLen : HeaderLen+msgLen]
		offset += HeaderLen + len(payload)
	}
	return data[:offset], data[offset:], nil
}

// FramePayload wraps payload in an uncompressed 5-byte gRPC frame.
func FramePayload(payload []byte) ([]byte, error) {
	if uint64(len(payload)) > math.MaxUint32 {
		return nil, fmt.Errorf("gRPC payload length %d exceeds uint32 maximum", len(payload))
	}
	framed := make([]byte, HeaderLen+len(payload))
	binary.BigEndian.PutUint32(framed[1:HeaderLen], uint32(len(payload))) //#nosec G115 -- guarded by MaxUint32 check above
	copy(framed[HeaderLen:], payload)
	return framed, nil
}

// MarshalProto serializes msg and wraps it in an uncompressed 5-byte gRPC frame.
func MarshalProto(msg proto.Message) ([]byte, error) {
	if msg == nil {
		return nil, errors.New("nil protobuf message")
	}
	raw, err := proto.Marshal(msg)
	if err != nil {
		return nil, err
	}
	return FramePayload(raw)
}
