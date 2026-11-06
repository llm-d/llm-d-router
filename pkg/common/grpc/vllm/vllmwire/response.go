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
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/protoscan"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
)

// ResponseInfo holds the fields scanned from a GenerateResponse message.
// ScanGenerateResponse fills it without allocating on canonical frames. It
// holds sub-slices of the scanned payload, so call KVTransferParams and
// ECTransferParams while the payload is alive and unchanged.
type ResponseInfo struct {
	HasPromptInfo   bool
	NumPromptTokens uint32

	HasOutputs bool

	HasFinishInfo   bool
	NumOutputTokens uint32
	HasCachedTokens bool
	NumCachedTokens uint32

	hasKVTransferParams bool
	kvTransferRaw       []byte
	hasECTransferParams bool
	ecTransferRaw       []byte

	outputsCount    int
	finishInfoCount int
	kvCopied        bool
	ecCopied        bool
}

// KVTransferParams decodes finish_info.kv_transfer_params. It returns nil
// when the frame carries none.
func (info *ResponseInfo) KVTransferParams() (map[string]any, error) {
	if !info.hasKVTransferParams {
		return nil, nil //nolint:nilnil // a nil map reports absence without an allocation
	}
	return protoscan.DecodeStruct(info.kvTransferRaw)
}

// ECTransferParams decodes finish_info.ec_transfer_params. It returns nil
// when the frame carries none.
func (info *ResponseInfo) ECTransferParams() (map[string]any, error) {
	if !info.hasECTransferParams {
		return nil, nil //nolint:nilnil // a nil map reports absence without an allocation
	}
	return protoscan.DecodeStruct(info.ecTransferRaw)
}

// ScanGenerateResponse reads the routing-relevant fields from a
// GenerateResponse payload without a full decode.
func ScanGenerateResponse(payload []byte) (ResponseInfo, error) {
	var info ResponseInfo
	err := protoscan.ForEachField(payload, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		switch num {
		case respPromptInfo:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			info.HasPromptInfo = true
			return scanPromptInfo(value, &info)
		case respOutputs:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			info.HasOutputs = true
			info.outputsCount++
			return scanSequenceOutput(value, &info)
		default:
			return nil
		}
	})
	return info, err
}

func scanPromptInfo(b []byte, info *ResponseInfo) error {
	return protoscan.ForEachField(b, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		if num == promptNumTokens {
			if wt != protowire.VarintType {
				return ErrMalformed
			}
			v, _ := protowire.ConsumeVarint(value)
			info.NumPromptTokens = uint32(v) //#nosec G115 -- uint32 field, protobuf truncates the same way
		}
		return nil
	})
}

func scanSequenceOutput(b []byte, info *ResponseInfo) error {
	return protoscan.ForEachField(b, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		if num != seqFinishInfo {
			return nil
		}
		if wt != protowire.BytesType {
			return ErrMalformed
		}
		info.HasFinishInfo = true
		info.finishInfoCount++
		return scanFinishInfo(value, info)
	})
}

func scanFinishInfo(b []byte, info *ResponseInfo) error {
	return protoscan.ForEachField(b, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		switch num {
		case finishNumOutputTokens:
			if wt != protowire.VarintType {
				return ErrMalformed
			}
			v, _ := protowire.ConsumeVarint(value)
			info.NumOutputTokens = uint32(v) //#nosec G115 -- uint32 field, protobuf truncates the same way
		case finishNumCachedTokens:
			if wt != protowire.VarintType {
				return ErrMalformed
			}
			v, _ := protowire.ConsumeVarint(value)
			info.HasCachedTokens = true
			info.NumCachedTokens = uint32(v) //#nosec G115 -- uint32 field, protobuf truncates the same way
		case finishKVTransferParams:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			info.kvTransferRaw = mergeRaw(info.kvTransferRaw, value, info.hasKVTransferParams, &info.kvCopied)
			info.hasKVTransferParams = true
		case finishECTransferParams:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			info.ecTransferRaw = mergeRaw(info.ecTransferRaw, value, info.hasECTransferParams, &info.ecCopied)
			info.hasECTransferParams = true
		}
		return nil
	})
}

// mergeRaw aliases body on the first occurrence and concatenates on repeats,
// which is the protobuf merge rule for an embedded message.
func mergeRaw(cur, body []byte, seen bool, copied *bool) []byte {
	if !seen {
		return body
	}
	if !*copied {
		cur = append(make([]byte, 0, len(cur)+len(body)), cur...)
		*copied = true
	}
	return append(cur, body...)
}

// RewriteTerminalCachedTokens substitutes cachedTokens onto
// outputs.finish_info.num_cached_tokens when frame is a terminal
// GenerateResponse frame whose cached token count differs from cachedTokens.
//
// Non-terminal frames (no FinishInfo) and terminal frames that already carry
// cachedTokens return frame itself with zero allocations.
func RewriteTerminalCachedTokens(frame []byte, cachedTokens uint32) ([]byte, error) {
	payload, err := grpcframe.ParsePayload(frame)
	if err != nil {
		return nil, err
	}
	info, err := ScanGenerateResponse(payload)
	if err != nil {
		return nil, err
	}
	if !info.HasFinishInfo {
		return frame, nil
	}
	if info.HasCachedTokens && info.NumCachedTokens == cachedTokens {
		return frame, nil
	}
	if info.outputsCount == 1 && info.finishInfoCount == 1 {
		if rewritten, ok := rewriteSingleTerminalFrame(payload, cachedTokens); ok {
			return rewritten, nil
		}
	}
	return rewriteFallback(frame, payload, cachedTokens)
}

// rewriteFallback handles wire inputs with repeated outputs or finish_info
// messages through a full decode.
func rewriteFallback(frame, payload []byte, cachedTokens uint32) ([]byte, error) {
	var resp vllmpb.GenerateResponse
	if err := proto.Unmarshal(payload, &resp); err != nil {
		return nil, err
	}
	if resp.GetOutputs().GetFinishInfo() == nil {
		return frame, nil
	}
	resp.Outputs.FinishInfo.NumCachedTokens = proto.Uint32(cachedTokens)
	return grpcframe.MarshalProto(&resp)
}

// fieldSpan locates the single field num in b and returns its start, end,
// and (for BytesType) its body.
func fieldSpan(b []byte, want protowire.Number) (start, end int, body []byte, ok bool) {
	pos := 0
	for pos < len(b) {
		num, wt, n := protowire.ConsumeTag(b[pos:])
		if n < 0 {
			return 0, 0, nil, false
		}
		m := protowire.ConsumeFieldValue(num, wt, b[pos+n:])
		if m < 0 {
			return 0, 0, nil, false
		}
		if num == want {
			if wt != protowire.BytesType {
				return 0, 0, nil, false
			}
			body, _ = protowire.ConsumeBytes(b[pos+n : pos+n+m])
			return pos, pos + n + m, body, true
		}
		pos += n + m
	}
	return 0, 0, nil, false
}

func rewriteSingleTerminalFrame(payload []byte, cachedTokens uint32) ([]byte, bool) {
	seqStart, seqEnd, seqBody, ok := fieldSpan(payload, respOutputs)
	if !ok {
		return nil, false
	}
	finStart, finEnd, finBody, ok := fieldSpan(seqBody, seqFinishInfo)
	if !ok {
		return nil, false
	}

	keptFinLen, _ := protoscan.SizeWithout(finBody, finishNumCachedTokens)
	newFinBodyLen := keptFinLen + protowire.SizeTag(finishNumCachedTokens) + protowire.SizeVarint(uint64(cachedTokens))
	newFinFieldLen := protowire.SizeTag(seqFinishInfo) + protowire.SizeBytes(newFinBodyLen)
	newSeqBodyLen := finStart + newFinFieldLen + (len(seqBody) - finEnd)
	newSeqFieldLen := protowire.SizeTag(respOutputs) + protowire.SizeBytes(newSeqBodyLen)
	newPayloadLen := seqStart + newSeqFieldLen + (len(payload) - seqEnd)
	if newPayloadLen > grpcframe.MaxPayloadLen {
		return nil, false
	}

	out := make([]byte, grpcframe.HeaderLen, grpcframe.HeaderLen+newPayloadLen)
	out = append(out, payload[:seqStart]...)
	out = protowire.AppendTag(out, respOutputs, protowire.BytesType)
	out = protoscan.AppendLen(out, newSeqBodyLen)
	out = append(out, seqBody[:finStart]...)
	out = protowire.AppendTag(out, seqFinishInfo, protowire.BytesType)
	out = protoscan.AppendLen(out, newFinBodyLen)
	out, _ = protoscan.AppendWithout(out, finBody, finishNumCachedTokens)
	out = protowire.AppendTag(out, finishNumCachedTokens, protowire.VarintType)
	out = protowire.AppendVarint(out, uint64(cachedTokens))
	out = append(out, seqBody[finEnd:]...)
	out = append(out, payload[seqEnd:]...)
	out, err := grpcframe.FinishFrame(out)
	return out, err == nil
}
