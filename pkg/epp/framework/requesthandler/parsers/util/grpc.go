/*
Copyright 2026 The Kubernetes Authors.

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

package grpcutil

import (
	"encoding/binary"
	"errors"
	"fmt"
)

const (
	gRPCPayloadHeaderLen = 5
)

// ParseGrpcPayload extracts the message payload from a gRPC frame.
// A standard gRPC frame consists of a 1-byte compression flag, a 4-byte message length,
// and the actual message payload.
// It returns an error if the payload is compressed.
func ParseGrpcPayload(data []byte) ([]byte, error) {
	if len(data) < gRPCPayloadHeaderLen {
		return nil, fmt.Errorf("invalid gRPC frame: expected at least %d bytes for header, got %d", gRPCPayloadHeaderLen, len(data))
	}

	isCompressed := data[0] == 1
	if isCompressed {
		// TODO(#895): handle compressed payload.
		return nil, errors.New("compressed gRPC payload is not supported")
	}
	msgLen := binary.BigEndian.Uint32(data[1:5])

	if uint32(len(data)) < gRPCPayloadHeaderLen+msgLen {
		return nil, fmt.Errorf("incomplete gRPC payload: header indicates %d bytes, but only %d bytes are available", msgLen, uint32(len(data))-gRPCPayloadHeaderLen)
	}
	return data[gRPCPayloadHeaderLen : gRPCPayloadHeaderLen+msgLen], nil
}
