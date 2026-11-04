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
	"slices"
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

var (
	// ErrCompressed reports a gRPC frame with the compression flag set.
	ErrCompressed = errors.New("compressed gRPC payload is not supported")
	// ErrTooLarge reports a gRPC request body over the size limit.
	ErrTooLarge = errors.New("gRPC request body too large")
	// ErrMalformed reports a truncated gRPC frame or data after the frame.
	ErrMalformed = errors.New("invalid gRPC frame")

	canonicalStatus = http.CanonicalHeaderKey(HeaderStatus)
	trailerHeaders  = [...]string{
		HeaderStatus,
		HeaderMessage,
		HeaderStatusDetailsBin,
	}
)

// IsContentType reports whether contentType is a gRPC media type
// ("application/grpc" or "application/grpc+<subtype>").
func IsContentType(contentType string) bool {
	mediaType, _, _ := strings.Cut(contentType, ";")
	mediaType = strings.ToLower(strings.TrimSpace(mediaType))
	return mediaType == MediaType || strings.HasPrefix(mediaType, MediaType+"+")
}

// Status extracts the gRPC status code string from an HTTP header or trailer
// map, including unannounced HTTP/2 trailers prefixed with http.TrailerPrefix.
func Status(h http.Header) (string, bool) {
	// http.CanonicalHeaderKey leaves keys containing ':' unchanged, so
	// httputil.ReverseProxy (TrailerPrefix + "Grpc-Status") and direct callers
	// (TrailerPrefix + HeaderStatus) store distinct map keys.
	for _, key := range [...]string{
		canonicalStatus,
		http.TrailerPrefix + canonicalStatus,
		http.TrailerPrefix + HeaderStatus,
	} {
		if v := firstNonEmpty(h[key]); v != "" {
			return v, true
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
		canonical := http.CanonicalHeaderKey(key)
		for _, srcKey := range [...]string{canonical, http.TrailerPrefix + canonical, http.TrailerPrefix + key} {
			for _, val := range src[srcKey] {
				dst.Add(http.TrailerPrefix+key, val)
			}
		}
	}
}

// WriteError writes a trailers-only gRPC error response to w.
func WriteError(w http.ResponseWriter, code codes.Code, msg string) {
	h := w.Header()
	h.Set(reqcommon.HeaderContentType, MediaType)
	h.Set(HeaderStatus, strconv.FormatUint(uint64(code), 10))
	if msg != "" {
		h.Set(HeaderMessage, EncodeStatusMessage(msg))
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

// EncodeStatusMessage percent-encodes bytes outside unescaped printable ASCII
// per the gRPC over HTTP/2 specification.
func EncodeStatusMessage(msg string) string {
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

func parseFrameHeader(header []byte) (uint32, error) {
	if len(header) < HeaderLen {
		return 0, fmt.Errorf("invalid gRPC frame: expected at least %d bytes for header, got %d", HeaderLen, len(header))
	}
	if header[0] != 0 {
		// TODO(#895): handle compressed payload.
		return 0, ErrCompressed
	}
	return binary.BigEndian.Uint32(header[1:HeaderLen]), nil
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
	msgLen, err := checkFrameHeader(header[:], maxPayloadBytes)
	if err != nil {
		return nil, err
	}
	payload := make([]byte, msgLen)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, fmt.Errorf("read gRPC frame payload: %w", err)
	}
	return payload, nil
}

// checkFrameHeader parses header and checks the declared payload length
// against maxPayloadBytes (when > 0) and math.MaxInt-HeaderLen.
func checkFrameHeader(header []byte, maxPayloadBytes int) (int, error) {
	msgLen, err := parseFrameHeader(header)
	if err != nil {
		return 0, err
	}
	if maxPayloadBytes > 0 && uint64(msgLen) > uint64(maxPayloadBytes) {
		return 0, fmt.Errorf("gRPC frame payload %d bytes exceeds limit %d", msgLen, maxPayloadBytes)
	}
	if uint64(msgLen) > uint64(math.MaxInt-HeaderLen) {
		return 0, fmt.Errorf("gRPC frame payload %d bytes overflows int", msgLen)
	}
	return int(msgLen), nil //#nosec G115 -- guarded by MaxInt check above
}

// Reader reads uncompressed gRPC frames from a stream into one reused
// buffer, so a long server stream does not allocate per frame.
type Reader struct {
	r               io.Reader
	maxPayloadBytes int
	buf             []byte
}

// NewReader returns a Reader for r. When maxPayloadBytes > 0, frames that
// declare a larger payload are rejected before the buffer grows.
func NewReader(r io.Reader, maxPayloadBytes int) *Reader {
	return &Reader{r: r, maxPayloadBytes: maxPayloadBytes, buf: make([]byte, HeaderLen, 4096)}
}

// Next reads the next frame and returns it with its 5-byte header, ready to
// write to another gRPC stream as is. Use frame[HeaderLen:] for the payload.
// The returned slice is valid only until the next call to Next. Next returns
// io.EOF when the stream ends cleanly on a frame boundary.
func (fr *Reader) Next() ([]byte, error) {
	header := fr.buf[:HeaderLen]
	_, err := io.ReadFull(fr.r, header)
	if errors.Is(err, io.EOF) {
		return nil, io.EOF
	}
	if err != nil {
		return nil, fmt.Errorf("read gRPC frame header: %w", err)
	}
	msgLen, err := checkFrameHeader(header, fr.maxPayloadBytes)
	if err != nil {
		return nil, err
	}
	frame := slices.Grow(header, msgLen)[:HeaderLen+msgLen]
	if _, err := io.ReadFull(fr.r, frame[HeaderLen:]); err != nil {
		return nil, fmt.Errorf("read gRPC frame payload: %w", err)
	}
	fr.buf = frame
	return frame, nil
}

// ReadMessage reads a request body that must hold exactly one uncompressed
// gRPC frame of at most maxFrameBytes bytes, header included. It allocates
// the frame once at its exact size, from the length prefix. When the body
// is larger than maxFrameBytes it returns ErrTooLarge. Otherwise it returns
// ErrCompressed for a compressed frame and ErrMalformed for a truncated
// frame or trailing data. Other errors come from r.
func ReadMessage(r io.Reader, maxFrameBytes int64) ([]byte, error) {
	var header [HeaderLen]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, malformedOrReadError(err)
	}
	msgLen, err := parseFrameHeader(header[:])
	if err != nil || int64(HeaderLen)+int64(msgLen) > maxFrameBytes {
		// Only an error path drains the body: a body over the limit reports
		// ErrTooLarge first, whatever its header declares.
		if err == nil {
			err = ErrMalformed
		}
		return nil, classifyRemainder(r, HeaderLen, maxFrameBytes, err)
	}
	frame := make([]byte, HeaderLen+int(msgLen))
	copy(frame, header[:])
	if _, err := io.ReadFull(r, frame[HeaderLen:]); err != nil {
		return nil, malformedOrReadError(err)
	}
	if _, err := io.ReadFull(r, header[:1]); !errors.Is(err, io.EOF) {
		if err != nil {
			return nil, fmt.Errorf("read gRPC request body: %w", err)
		}
		return nil, classifyRemainder(r, int64(len(frame))+1, maxFrameBytes, ErrMalformed)
	}
	return frame, nil
}

func malformedOrReadError(err error) error {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return ErrMalformed
	}
	return fmt.Errorf("read gRPC request body: %w", err)
}

// classifyRemainder drains at most maxFrameBytes+1 body bytes in total and
// returns ErrTooLarge when the body exceeds maxFrameBytes, else fallback.
func classifyRemainder(r io.Reader, consumed, maxFrameBytes int64, fallback error) error {
	if consumed > maxFrameBytes {
		return ErrTooLarge
	}
	n, err := io.Copy(io.Discard, io.LimitReader(r, maxFrameBytes+1-consumed))
	if err != nil {
		return fmt.Errorf("read gRPC request body: %w", err)
	}
	if consumed+n > maxFrameBytes {
		return ErrTooLarge
	}
	return fallback
}

// ParsePayload extracts the message payload from a single uncompressed gRPC frame.
func ParsePayload(data []byte) ([]byte, error) {
	msgLen, err := parseFrameHeader(data)
	if err != nil {
		return nil, err
	}
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

// FramePayload wraps payload in an uncompressed 5-byte gRPC frame.
func FramePayload(payload []byte) ([]byte, error) {
	const maxPayload = min(math.MaxUint32, math.MaxInt-HeaderLen)
	if len(payload) > maxPayload {
		return nil, fmt.Errorf("gRPC payload length %d exceeds maximum %d", len(payload), maxPayload)
	}
	framed := make([]byte, HeaderLen+len(payload))
	binary.BigEndian.PutUint32(framed[1:HeaderLen], uint32(len(payload))) //#nosec G115 -- guarded by maxPayload check above
	copy(framed[HeaderLen:], payload)
	return framed, nil
}

// MarshalProto serializes msg and wraps it in an uncompressed 5-byte gRPC
// frame. It encodes directly after the reserved header, so it allocates once.
func MarshalProto(msg proto.Message) ([]byte, error) {
	if msg == nil {
		return nil, errors.New("nil protobuf message")
	}
	size := proto.Size(msg)
	if uint64(size) > math.MaxUint32 {
		return nil, fmt.Errorf("gRPC payload length %d exceeds maximum %d", size, uint64(math.MaxUint32))
	}
	// UseCachedSize reuses the sizes that proto.Size just computed.
	framed, err := proto.MarshalOptions{UseCachedSize: true}.MarshalAppend(make([]byte, HeaderLen, HeaderLen+size), msg)
	if err != nil {
		return nil, err
	}
	binary.BigEndian.PutUint32(framed[1:HeaderLen], uint32(len(framed)-HeaderLen)) //#nosec G115 -- guarded by MaxUint32 check above
	return framed, nil
}
