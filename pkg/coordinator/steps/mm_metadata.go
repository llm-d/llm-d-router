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

package steps

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"sort"

	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// vllmRawViewExt is the msgpack extension type vLLM's MsgpackEncoder uses for
// the inline bytes of a small tensor (CUSTOM_TYPE_RAW_VIEW in
// vllm/v1/serial_utils.py).
const vllmRawViewExt = 3

// mmMetadataFromEC builds the mm_metadata item vLLM's render would return for
// one entry, from the "metadata" object an encode response publishes in its
// ec_transfer_params descriptor. vLLM ships the same fields both ways: the
// placeholder-metadata fields the model's parser declares.
//
// The item is a vLLM MultiModalKwargsItem in the msgpack layout of
// vllm/v1/serial_utils.py: {field: {"data": [dtype, shape, raw bytes],
// "field": ["batched", {"keep_on_cpu": true}]}}. Only a list of integers on a
// batched keep_on_cpu field is encoded (image_grid_thw and video_grid_thw on
// the Qwen-VL family); any other value returns ok false, so the caller sends
// the media and lets the worker preprocess it.
func mmMetadataFromEC(metadata map[string]any) (item string, ok bool) {
	if len(metadata) == 0 {
		return "", false
	}
	keys := make([]string, 0, len(metadata))
	for k := range metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var buf bytes.Buffer
	enc := msgpack.NewEncoder(&buf)
	if err := enc.EncodeMapLen(len(keys)); err != nil {
		return "", false
	}
	for _, key := range keys {
		values, ok := int64Values(metadata[key])
		if !ok {
			return "", false
		}
		if err := encodeBatchedInt64Field(enc, key, values); err != nil {
			return "", false
		}
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), true
}

// entriesWithECMetadata returns a copy of entries with MMMetadata rebuilt from
// each entry's ec_transfer_params descriptor. ok is false when any entry has no
// descriptor or no metadata mmMetadataFromEC can encode: vLLM's prefill needs
// either the tensors or the metadata for every item, and the request has
// neither tensors nor a way to send part of its media.
func entriesWithECMetadata(entries []pipeline.MultimodalEntry, ecParams map[string]any) ([]pipeline.MultimodalEntry, bool) {
	out := make([]pipeline.MultimodalEntry, len(entries))
	for i, entry := range entries {
		desc, _ := ecParams[entry.Hash].(map[string]any)
		metadata, _ := desc["metadata"].(map[string]any)
		item, ok := mmMetadataFromEC(metadata)
		if !ok {
			return nil, false
		}
		entry.MMMetadata = item
		out[i] = entry
	}
	return out, true
}

// int64Values reads a JSON array of integers, as decoded into []any.
func int64Values(v any) ([]int64, bool) {
	list, ok := v.([]any)
	if !ok || len(list) == 0 {
		return nil, false
	}
	out := make([]int64, len(list))
	for i, elem := range list {
		f, ok := elem.(float64)
		if !ok || f != math.Trunc(f) || f < math.MinInt64 || f > math.MaxInt64 {
			return nil, false
		}
		out[i] = int64(f)
	}
	return out, true
}

func encodeBatchedInt64Field(enc *msgpack.Encoder, key string, values []int64) error {
	raw := make([]byte, 8*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint64(raw[8*i:], uint64(v))
	}
	steps := []func() error{
		func() error { return enc.EncodeString(key) },
		func() error { return enc.EncodeMapLen(2) },
		func() error { return enc.EncodeString("data") },
		func() error { return enc.EncodeArrayLen(3) },
		func() error { return enc.EncodeString("int64") },
		func() error { return enc.EncodeArrayLen(1) },
		func() error { return enc.EncodeInt(int64(len(values))) },
		func() error { return enc.EncodeExtHeader(vllmRawViewExt, len(raw)) },
		func() error { _, err := enc.Writer().Write(raw); return err },
		func() error { return enc.EncodeString("field") },
		func() error { return enc.EncodeArrayLen(2) },
		func() error { return enc.EncodeString("batched") },
		func() error { return enc.EncodeMapLen(1) },
		func() error { return enc.EncodeString("keep_on_cpu") },
		func() error { return enc.EncodeBool(true) },
	}
	for _, step := range steps {
		if err := step(); err != nil {
			return fmt.Errorf("encoding %s: %w", key, err)
		}
	}
	return nil
}
