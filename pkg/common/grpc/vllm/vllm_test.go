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

package vllm_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
)

func TestFullMethodNames(t *testing.T) {
	t.Parallel()

	cases := map[string]string{
		"Inference.Generate":       vllmpb.Inference_Generate_FullMethodName,
		"Inference.GenerateStream": vllmpb.Inference_GenerateStream_FullMethodName,
		"Control.GetServerInfo":    vllmpb.Control_GetServerInfo_FullMethodName,
		"Control.GetModelInfo":     vllmpb.Control_GetModelInfo_FullMethodName,
		"Control.Abort":            vllmpb.Control_Abort_FullMethodName,
	}
	want := map[string]string{
		"Inference.Generate":       "/vllm.Inference/Generate",
		"Inference.GenerateStream": "/vllm.Inference/GenerateStream",
		"Control.GetServerInfo":    "/vllm.Control/GetServerInfo",
		"Control.GetModelInfo":     "/vllm.Control/GetModelInfo",
		"Control.Abort":            "/vllm.Control/Abort",
	}
	if diff := cmp.Diff(want, cases); diff != "" {
		t.Fatalf("method names mismatch (-want +got):\n%s", diff)
	}
}

func TestGenerateRoundTripFrame(t *testing.T) {
	t.Parallel()

	kvParams, err := structpb.NewStruct(map[string]any{
		"do_remote_prefill": true,
		"remote_host":       "10.0.0.4",
		"remote_port":       float64(5557),
	})
	if err != nil {
		t.Fatalf("structpb.NewStruct: %v", err)
	}

	req := &vllmpb.GenerateRequest{
		RequestId: "req-1",
		Model:     "meta-llama/Llama-3.1-8B-Instruct",
		Prompt: &vllmpb.GenerateRequest_TokenIds{
			TokenIds: &vllmpb.TokenIds{Ids: []uint32{128000, 9906, 1917}},
		},
		Stopping: &vllmpb.StoppingCriteria{MaxNewTokens: 64},
		Kv: &vllmpb.KVCacheParameters{
			CacheSalt:        "salt-1",
			KvTransferParams: kvParams,
		},
		Priority: 3,
		LoraName: "adapter-a",
	}

	framed, err := grpcframe.MarshalProto(req)
	if err != nil {
		t.Fatalf("MarshalProto: %v", err)
	}
	payload, err := grpcframe.ParsePayload(framed)
	if err != nil {
		t.Fatalf("ParsePayload: %v", err)
	}
	var got vllmpb.GenerateRequest
	if err := proto.Unmarshal(payload, &got); err != nil {
		t.Fatalf("proto.Unmarshal: %v", err)
	}
	if diff := cmp.Diff(req, &got, protocmp.Transform()); diff != "" {
		t.Fatalf("GenerateRequest round-trip mismatch (-want +got):\n%s", diff)
	}
}
