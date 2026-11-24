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

// Package vllmwire reads and rewrites vllm.Inference GenerateRequest and
// GenerateResponse messages at the protobuf wire level. The EPP parser, the
// PD sidecar, and the coordinator use it instead of proto.Unmarshal,
// proto.Clone, and proto.Marshal so token IDs, media bytes, and non-terminal
// stream frames are forwarded without a decode or a copy.
package vllmwire

import (
	"fmt"
	"slices"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/protoscan"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
)

// ErrMalformed reports bytes that are not valid protobuf wire data. It is
// the same value as protoscan.ErrMalformed.
var ErrMalformed = protoscan.ErrMalformed

// PromptKind describes which branch of GenerateRequest.prompt is set.
type PromptKind uint8

const (
	// PromptNone indicates that neither text nor token_ids was set.
	PromptNone PromptKind = iota
	// PromptText indicates that GenerateRequest.text was set.
	PromptText
	// PromptTokenIDs indicates that GenerateRequest.token_ids was set.
	PromptTokenIDs
)

// Field numbers come from the generated descriptors, so a renumbered field
// fails at init instead of silently corrupting messages.
var (
	reqRequestID = protoscan.FieldNum(&vllmpb.GenerateRequest{}, "request_id")
	reqModel     = protoscan.FieldNum(&vllmpb.GenerateRequest{}, "model")
	reqText      = protoscan.FieldNum(&vllmpb.GenerateRequest{}, "text")
	reqTokenIDs  = protoscan.FieldNum(&vllmpb.GenerateRequest{}, "token_ids")
	reqSampling  = protoscan.FieldNum(&vllmpb.GenerateRequest{}, "sampling")
	reqStopping  = protoscan.FieldNum(&vllmpb.GenerateRequest{}, "stopping")
	reqKV        = protoscan.FieldNum(&vllmpb.GenerateRequest{}, "kv")
	reqPriority  = protoscan.FieldNum(&vllmpb.GenerateRequest{}, "priority")
	reqMedia     = protoscan.FieldNum(&vllmpb.GenerateRequest{}, "media")
	reqLoraName  = protoscan.FieldNum(&vllmpb.GenerateRequest{}, "lora_name")

	tokenIDsIDs = protoscan.FieldNum(&vllmpb.TokenIds{}, "ids")

	samplingNumSequences = protoscan.FieldNum(&vllmpb.RandomSampling{}, "num_sequences")

	stopMaxNewTokens = protoscan.FieldNum(&vllmpb.StoppingCriteria{}, "max_new_tokens")
	stopMinNewTokens = protoscan.FieldNum(&vllmpb.StoppingCriteria{}, "min_new_tokens")

	kvBypassPrefixCache = protoscan.FieldNum(&vllmpb.KVCacheParameters{}, "bypass_prefix_cache")
	kvCacheSalt         = protoscan.FieldNum(&vllmpb.KVCacheParameters{}, "cache_salt")
	kvTransferParams    = protoscan.FieldNum(&vllmpb.KVCacheParameters{}, "kv_transfer_params")
	ecTransferParams    = protoscan.FieldNum(&vllmpb.KVCacheParameters{}, "ec_transfer_params")

	mediaModality = protoscan.FieldNum(&vllmpb.MediaItem{}, "modality")
	mediaURL      = protoscan.FieldNum(&vllmpb.MediaItem{}, "url")
	mediaDataURI  = protoscan.FieldNum(&vllmpb.MediaItem{}, "data_uri")
	mediaRawBytes = protoscan.FieldNum(&vllmpb.MediaItem{}, "raw_bytes")
	mediaUUID     = protoscan.FieldNum(&vllmpb.MediaItem{}, "uuid")
	mediaFeatures = protoscan.FieldNum(&vllmpb.MediaItem{}, "features")

	featKwargs     = protoscan.FieldNum(&vllmpb.PreprocessedMediaFeatures{}, "kwargs")
	featIdentifier = protoscan.FieldNum(&vllmpb.PreprocessedMediaFeatures{}, "identifier")
	featOffset     = protoscan.FieldNum(&vllmpb.PreprocessedMediaFeatures{}, "offset")
	featLength     = protoscan.FieldNum(&vllmpb.PreprocessedMediaFeatures{}, "length")
	featMMHash     = protoscan.FieldNum(&vllmpb.PreprocessedMediaFeatures{}, "mm_hash")

	respPromptInfo = protoscan.FieldNum(&vllmpb.GenerateResponse{}, "prompt_info")
	respOutputs    = protoscan.FieldNum(&vllmpb.GenerateResponse{}, "outputs")

	promptNumTokens = protoscan.FieldNum(&vllmpb.PromptInfo{}, "num_prompt_tokens")

	seqFinishInfo = protoscan.FieldNum(&vllmpb.SequenceOutput{}, "finish_info")

	finishNumOutputTokens  = protoscan.FieldNum(&vllmpb.FinishInfo{}, "num_output_tokens")
	finishKVTransferParams = protoscan.FieldNum(&vllmpb.FinishInfo{}, "kv_transfer_params")
	finishECTransferParams = protoscan.FieldNum(&vllmpb.FinishInfo{}, "ec_transfer_params")
	finishNumCachedTokens  = protoscan.FieldNum(&vllmpb.FinishInfo{}, "num_cached_tokens")
)

// inlineSpans is the number of byte spans a Request stores without a heap
// allocation. A canonical GenerateRequest has at most four base spans
// (between request_id, stopping, kv, and media) and few media items.
const inlineSpans = 4

// spanList stores byte slices that alias the scanned payload. The first
// inlineSpans entries live in the struct so a canonical request scans with
// zero allocations.
type spanList struct {
	inline [inlineSpans][]byte
	count  int
	extra  [][]byte
	bytes  int
}

func (l *spanList) add(span []byte) {
	l.bytes += len(span)
	if l.count < inlineSpans {
		l.inline[l.count] = span
		l.count++
		return
	}
	l.extra = append(l.extra, span)
	l.count++
}

func (l *spanList) at(i int) []byte {
	if i < inlineSpans {
		return l.inline[i]
	}
	return l.extra[i-inlineSpans]
}

// Request holds the scanned view of a GenerateRequest. Large fields
// (token_ids, text, media) stay in place as sub-slices of the original
// payload. The payload must stay alive and unchanged while the Request is in
// use.
//
// The frame builders and the accessors only read r, so goroutines may share
// one Request once TokenIDs and SetMediaBodies, which write to it, have run.
type Request struct {
	// Prompt reports which branch of GenerateRequest.prompt was set.
	Prompt PromptKind

	requestIDField []byte // full request_id field (tag, length, value)
	requestID      []byte
	model          []byte
	loraName       []byte
	text           []byte

	tokenIDsFirst  []byte // first TokenIds message body
	tokenIDsMore   [][]byte
	tokenIDs       []uint32
	tokenIDsCached bool

	hasNumSequences bool
	numSequences    uint32
	hasMaxNewTokens bool
	maxNewTokens    uint32
	hasPriority     bool
	priority        int32

	bypassPrefixCache bool
	hasCacheSalt      bool
	cacheSalt         []byte

	clientKVFirst     []byte // client kv_transfer_params Struct bodies
	clientKVMore      [][]byte
	hadClientTransfer bool

	// base holds every top-level field except request_id, stopping, kv, and
	// media. FrameSegments forwards these spans without a copy.
	base spanList
	// media holds one full field (tag, length, body) per media item.
	media spanList
	// mediaBodies, when set by SetMediaBodies, replaces the media item
	// bodies in every frame this Request builds.
	mediaBodies [][]byte

	hasStopping    bool
	stoppingBody   []byte
	stoppingCopied bool

	hasKV    bool
	kvBody   []byte // client kv body without kv_transfer_params and ec_transfer_params
	kvCopied bool
}

// RequestID returns GenerateRequest.request_id (empty when omitted).
func (r *Request) RequestID() string { return string(r.requestID) }

// Model returns GenerateRequest.model (empty when omitted).
func (r *Request) Model() string { return string(r.model) }

// LoraName returns GenerateRequest.lora_name (empty when omitted).
func (r *Request) LoraName() string { return string(r.loraName) }

// Text returns the prompt.text bytes. It aliases the payload and is nil
// when the prompt is token_ids or absent.
func (r *Request) Text() []byte {
	if r.Prompt != PromptText {
		return nil
	}
	return r.text
}

// TokenIDs decodes prompt.token_ids.ids once and caches the result on r.
// Merged TokenIds messages concatenate in order. It returns nil when the
// prompt is not token_ids. The returned slice is shared across calls and
// must not be modified.
func (r *Request) TokenIDs() ([]uint32, error) {
	if r.Prompt != PromptTokenIDs {
		return nil, nil
	}
	if r.tokenIDsCached {
		return r.tokenIDs, nil
	}
	ids := make([]uint32, 0, r.tokenCount())
	var err error
	if ids, err = appendTokenIDs(ids, r.tokenIDsFirst); err != nil {
		return nil, err
	}
	for _, body := range r.tokenIDsMore {
		if ids, err = appendTokenIDs(ids, body); err != nil {
			return nil, err
		}
	}
	r.tokenIDs = ids
	r.tokenIDsCached = true
	return ids, nil
}

func appendTokenIDs(dst []uint32, body []byte) ([]uint32, error) {
	var err error
	walkErr := protoscan.ForEachField(body, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		// A fixed-width occurrence is an unknown field to proto.Unmarshal,
		// so it is skipped here and in the count, as sglangwire does.
		if num != tokenIDsIDs || (wt != protowire.BytesType && wt != protowire.VarintType) {
			return nil
		}
		dst, err = protoscan.AppendPackedUint32(dst, wt, value)
		return err
	})
	return dst, walkErr
}

// tokenCount returns the number of prompt token IDs without decoding them.
// It returns 0 when the prompt is not token_ids.
func (r *Request) tokenCount() int {
	if r.Prompt != PromptTokenIDs {
		return 0
	}
	n := countTokenIDs(r.tokenIDsFirst)
	for _, body := range r.tokenIDsMore {
		n += countTokenIDs(body)
	}
	return n
}

func countTokenIDs(body []byte) int {
	n := 0
	_ = protoscan.ForEachField(body, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		if num != tokenIDsIDs {
			return nil
		}
		switch wt {
		case protowire.BytesType:
			n += protoscan.CountVarints(value)
		case protowire.VarintType:
			n++
		}
		return nil
	})
	return n
}

// CacheSalt returns kv.cache_salt and whether the client set it. The slice
// aliases the payload.
func (r *Request) CacheSalt() ([]byte, bool) { return r.cacheSalt, r.hasCacheSalt }

// MaxNewTokens returns stopping.max_new_tokens and whether the client set it.
func (r *Request) MaxNewTokens() (uint32, bool) { return r.maxNewTokens, r.hasMaxNewTokens }

// NumSequences returns sampling.num_sequences and whether the client set it.
func (r *Request) NumSequences() (uint32, bool) { return r.numSequences, r.hasNumSequences }

// Priority returns GenerateRequest.priority and whether the client set it.
func (r *Request) Priority() (int32, bool) { return r.priority, r.hasPriority }

// HadClientTransferParams reports whether the client sent kv.kv_transfer_params
// or kv.ec_transfer_params. Frames built from r never carry them.
func (r *Request) HadClientTransferParams() bool { return r.hadClientTransfer }

// ClientKVTransferParams decodes the client's kv.kv_transfer_params. The
// sidecar reads it only when the EPP sets the P2P KV source header. ok is
// false when the client sent none.
func (r *Request) ClientKVTransferParams() (params map[string]any, ok bool, err error) {
	if r.clientKVFirst == nil {
		return nil, false, nil
	}
	raw := r.clientKVFirst
	if len(r.clientKVMore) > 0 {
		merged := make([]byte, 0, len(raw)+totalLen(r.clientKVMore))
		merged = append(merged, raw...)
		for _, more := range r.clientKVMore {
			merged = append(merged, more...)
		}
		raw = merged
	}
	m, err := protoscan.DecodeStruct(raw)
	if err != nil {
		return nil, false, fmt.Errorf("vllmwire: client kv_transfer_params: %w", err)
	}
	return m, true, nil
}

func totalLen(bufs [][]byte) int {
	n := 0
	for _, b := range bufs {
		n += len(b)
	}
	return n
}

// HasMedia reports whether the request carries at least one media item.
func (r *Request) HasMedia() bool { return r.media.count > 0 }

// ScanGenerateRequest validates payload at the top level (plus the stopping,
// kv, token_ids, media, and media features submessages) and indexes the fields that the
// EPP, sidecar, and coordinator read or replace. It never copies token_ids,
// text, or media bytes. Scalar fields follow protobuf merge rules: the last
// value wins. Repeated stopping and kv messages merge.
func ScanGenerateRequest(payload []byte) (Request, error) {
	var req Request
	spanStart := 0
	for pos := 0; pos < len(payload); {
		num, wt, n := protowire.ConsumeTag(payload[pos:])
		if n < 0 {
			return Request{}, ErrMalformed
		}
		m := protowire.ConsumeFieldValue(num, wt, payload[pos+n:])
		if m < 0 {
			return Request{}, ErrMalformed
		}
		end := pos + n + m
		field := payload[pos:end]
		value := payload[pos+n : end]
		if wt == protowire.BytesType {
			value, _ = protowire.ConsumeBytes(value)
		}

		carve := false
		switch num {
		case reqRequestID:
			if wt != protowire.BytesType {
				return Request{}, ErrMalformed
			}
			carve = true
			req.requestIDField = field
			req.requestID = value
		case reqStopping:
			if wt != protowire.BytesType {
				return Request{}, ErrMalformed
			}
			carve = true
			if err := req.scanStopping(value); err != nil {
				return Request{}, err
			}
		case reqKV:
			if wt != protowire.BytesType {
				return Request{}, ErrMalformed
			}
			carve = true
			if err := req.scanKV(value); err != nil {
				return Request{}, err
			}
		case reqMedia:
			if wt != protowire.BytesType {
				return Request{}, ErrMalformed
			}
			if err := validateMediaItem(value); err != nil {
				return Request{}, err
			}
			carve = true
			req.media.add(field)
		default:
			if err := req.scanBaseField(num, wt, value); err != nil {
				return Request{}, err
			}
		}

		if carve {
			req.addBase(payload[spanStart:pos])
			spanStart = end
		}
		pos = end
	}
	req.addBase(payload[spanStart:])
	return req, nil
}

func (r *Request) addBase(span []byte) {
	if len(span) == 0 {
		return
	}
	r.base.add(span)
}

// scanBaseField records the fields that stay in the base spans.
func (r *Request) scanBaseField(num protowire.Number, wt protowire.Type, value []byte) error {
	switch num {
	case reqModel:
		if wt != protowire.BytesType {
			return ErrMalformed
		}
		r.model = value
	case reqLoraName:
		if wt != protowire.BytesType {
			return ErrMalformed
		}
		r.loraName = value
	case reqText:
		if wt != protowire.BytesType {
			return ErrMalformed
		}
		// prompt is a oneof: a later text clears earlier token_ids.
		r.Prompt = PromptText
		r.text = value
		r.tokenIDsFirst = nil
		r.tokenIDsMore = nil
	case reqTokenIDs:
		if wt != protowire.BytesType {
			return ErrMalformed
		}
		if err := protoscan.Validate(value); err != nil {
			return err
		}
		if r.Prompt != PromptTokenIDs {
			r.Prompt = PromptTokenIDs
			r.text = nil
			r.tokenIDsFirst = value
			r.tokenIDsMore = nil
		} else {
			r.tokenIDsMore = append(r.tokenIDsMore, value)
		}
	case reqSampling:
		if wt != protowire.BytesType {
			return ErrMalformed
		}
		return protoscan.ForEachField(value, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
			if num != samplingNumSequences {
				return nil
			}
			if wt != protowire.VarintType {
				return ErrMalformed
			}
			v, _ := protowire.ConsumeVarint(value)
			r.hasNumSequences = true
			r.numSequences = uint32(v) //#nosec G115 -- uint32 field, protobuf truncates the same way
			return nil
		})
	case reqPriority:
		if wt != protowire.VarintType {
			return ErrMalformed
		}
		v, _ := protowire.ConsumeVarint(value)
		r.hasPriority = true
		r.priority = int32(uint32(v)) //#nosec G115 -- int32 field, protobuf truncates the same way
	}
	return nil
}

func (r *Request) scanStopping(body []byte) error {
	err := protoscan.ForEachField(body, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		if num != stopMaxNewTokens && num != stopMinNewTokens {
			return nil
		}
		if wt != protowire.VarintType {
			return ErrMalformed
		}
		if num == stopMaxNewTokens {
			v, _ := protowire.ConsumeVarint(value)
			r.hasMaxNewTokens = true
			r.maxNewTokens = uint32(v) //#nosec G115 -- uint32 field, protobuf truncates the same way
		}
		return nil
	})
	if err != nil {
		return err
	}
	if !r.hasStopping {
		r.hasStopping = true
		r.stoppingBody = body
		return nil
	}
	r.stoppingBody = appendDetached(r.stoppingBody, body, &r.stoppingCopied)
	return nil
}

// appendDetached returns cur with body appended. When copied is false, cur
// aliases the scanned payload, so the first append moves it to a buffer of
// its own and sets copied.
func appendDetached(cur, body []byte, copied *bool) []byte {
	if !*copied {
		*copied = true
		return slices.Concat(cur, body)
	}
	return append(cur, body...)
}

// scanKV validates a KVCacheParameters body, records cache_salt,
// bypass_prefix_cache, and the client's kv_transfer_params, and keeps a
// sanitized copy of the body without kv_transfer_params and
// ec_transfer_params.
func (r *Request) scanKV(body []byte) error {
	hasTransfer := false
	err := protoscan.ForEachField(body, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		switch num {
		case kvBypassPrefixCache:
			if wt != protowire.VarintType {
				return ErrMalformed
			}
			v, _ := protowire.ConsumeVarint(value)
			r.bypassPrefixCache = v != 0
		case kvCacheSalt:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			r.hasCacheSalt = true
			r.cacheSalt = value
		case kvTransferParams:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			if err := protoscan.Validate(value); err != nil {
				return err
			}
			hasTransfer = true
			if r.clientKVFirst == nil {
				r.clientKVFirst = value
			} else {
				r.clientKVMore = append(r.clientKVMore, value)
			}
		case ecTransferParams:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			if err := protoscan.Validate(value); err != nil {
				return err
			}
			hasTransfer = true
		}
		return nil
	})
	if err != nil {
		return err
	}

	if !hasTransfer {
		r.hasKV = true
		if !r.kvCopied && len(r.kvBody) == 0 {
			r.kvBody = body
			return nil
		}
		r.copyKVBody(len(body))
		r.kvBody = append(r.kvBody, body...)
		return nil
	}

	r.hadClientTransfer = true
	r.copyKVBody(len(body))
	err = protoscan.ForEachField(body, func(num protowire.Number, _ protowire.Type, _, field []byte) error {
		if num == kvTransferParams || num == ecTransferParams {
			return nil
		}
		r.hasKV = true
		r.kvBody = append(r.kvBody, field...)
		return nil
	})
	return err
}

// copyKVBody moves kvBody, which aliases the scanned payload, to a buffer
// of its own with room for extra more bytes.
func (r *Request) copyKVBody(extra int) {
	if r.kvCopied {
		return
	}
	r.kvBody = slices.Grow(r.kvBody[:len(r.kvBody):len(r.kvBody)], extra)
	r.kvCopied = true
}

// validateMediaItem checks a MediaItem body and its features submessage, so
// that the MediaMetadataOnly rewrite, which re-encodes features, never meets
// malformed bytes.
func validateMediaItem(body []byte) error {
	return protoscan.ForEachField(body, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		if num == mediaFeatures && wt == protowire.BytesType {
			return protoscan.Validate(value)
		}
		return nil
	})
}
