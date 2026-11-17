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

package p2psource

import (
	"net"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/protoscan"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

// KV hint protocol values KVCR accepts for a kv.fetch action.
const (
	kvHintProtocolVersion = "0.1"
	kvFetchActionType     = "kv.fetch"
	kvFetchActionVersion  = "1.0"
	kvFetchActionID       = "fetch-0"
	kvFetchSourceField    = "source_control_endpoint"
	kvFetchHashesField    = "block_hashes"

	// placeholderBlockHash fills kv.fetch's required, non-empty block_hashes.
	// vLLM's KVCR tier enables opportunistic_query by default, and then KVCR
	// treats every block of the request as fetchable from the named source
	// without reading the hashes. Real engine hashes would not survive the
	// trip anyway: vLLM's gRPC server converts a protobuf Struct number to an
	// integer only up to 2^53 - 1, and KVCR rejects a hint whose hash is not
	// an integer.
	placeholderBlockHash = 0
)

// kvHintsFieldNumber is GenerateRequest.kv_hints in the generated vLLM proto.
var kvHintsFieldNumber = (&vllmpb.GenerateRequest{}).ProtoReflect().Descriptor().Fields().ByName("kv_hints").Number()

// isVLLMGenerate reports whether path is the vllm.Inference Generate or
// GenerateStream method, with or without its leading slash, as the
// vllminference parser accepts it.
func isVLLMGenerate(path string) bool {
	path = "/" + strings.TrimPrefix(path, "/")
	return path == vllmpb.Inference_Generate_FullMethodName || path == vllmpb.Inference_GenerateStream_FullMethodName
}

// kvcrControlEndpoint is the KVCR control endpoint a destination dials to
// fetch from the source at address on port.
func kvcrControlEndpoint(address string, port int) string {
	return "tcp://" + net.JoinHostPort(address, strconv.Itoa(port))
}

// setKVFetchHint writes a kv.fetch hint naming sourceEndpoint into a
// vllm.Inference GenerateRequest frame body, replacing any kv_hints it holds,
// and reports whether it did. A body that is not a frame is left as is.
func setKVFetchHint(body *fwkrequest.InferenceRequestBody, sourceEndpoint string) (bool, error) {
	if body == nil {
		return false, nil
	}
	frame, ok := body.Payload.(fwkrequest.PayloadBytes)
	if !ok {
		return false, nil
	}
	payload, err := structpb.NewStruct(map[string]any{
		kvFetchSourceField: sourceEndpoint,
		kvFetchHashesField: []any{placeholderBlockHash},
	})
	if err != nil {
		return false, err
	}
	envelope, err := proto.Marshal(&vllmpb.KVHintsEnvelope{
		ProtocolVersion: kvHintProtocolVersion,
		MessageId:       uuid.NewString(),
		Actions: []*vllmpb.KVHintAction{{
			ActionId:      kvFetchActionID,
			ActionType:    kvFetchActionType,
			ActionVersion: kvFetchActionVersion,
			Payload:       payload,
		}},
	})
	if err != nil {
		return false, err
	}
	rewritten, err := protoscan.ReplaceBytesField(frame, kvHintsFieldNumber, string(envelope))
	if err != nil {
		return false, err
	}
	body.Payload = fwkrequest.PayloadBytes(rewritten)
	body.Mutated = true
	return true, nil
}
