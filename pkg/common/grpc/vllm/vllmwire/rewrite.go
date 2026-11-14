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
	"github.com/llm-d/llm-d-router/pkg/common/grpc/protoscan"
)

// RewritePriority returns a new frame with GenerateRequest.priority set to
// priority, dropping any client value. When the client sent no priority and
// priority is 0 it returns frame itself without a copy. frame is a full
// 5-byte-prefixed gRPC frame, as is the result.
func RewritePriority(frame []byte, priority int32) ([]byte, error) {
	// Proto3 int32 encodes negative values as 10-byte sign-extended varints.
	value := uint64(int64(priority)) //#nosec G115 -- two's complement is the protobuf int32 encoding
	return protoscan.ReplaceVarintField(frame, reqPriority, value, true)
}

// RewriteModel returns a new frame with GenerateRequest.model replaced by
// model. An empty model drops the field.
func RewriteModel(frame []byte, model string) ([]byte, error) {
	return protoscan.ReplaceBytesField(frame, reqModel, model)
}

// RewriteLoraName returns a new frame with GenerateRequest.lora_name
// replaced by name. An empty name drops the field.
func RewriteLoraName(frame []byte, name string) ([]byte, error) {
	return protoscan.ReplaceBytesField(frame, reqLoraName, name)
}

// RewriteKVHints returns a new frame with GenerateRequest.kv_hints set to
// envelope, a serialized KVHintsEnvelope, dropping every client occurrence of
// the field. An empty envelope drops the field. When frame has no kv_hints and
// envelope is empty it returns frame itself without a copy.
func RewriteKVHints(frame, envelope []byte) ([]byte, error) {
	return protoscan.ReplaceBytesField(frame, reqKVHints, envelope)
}
