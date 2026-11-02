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

package routing

import (
	"testing"

	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestHasPDRoute(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		{path: reqcommon.PathChatCompletions, want: true},
		{path: reqcommon.PathCompletions, want: true},
		{path: reqcommon.PathResponses, want: true},
		{path: reqcommon.PathMessages, want: true},
		{path: reqcommon.PathVLLMGenerate, want: true},
		{path: reqcommon.PathSGLangGenerate, want: true},
		{path: reqcommon.PathChatCompletions + "/", want: true},
		{path: reqcommon.PathChatCompletions + "?api-version=1", want: true},
		// A gateway can strip a provider prefix before the request reaches the sidecar.
		{path: "/openai" + reqcommon.PathChatCompletions, want: true},
		{path: vllmpb.Inference_Generate_FullMethodName, want: true},
		{path: vllmpb.Inference_GenerateStream_FullMethodName, want: true},
		{path: vllmpb.Inference_Generate_FullMethodName + "/", want: true},
		{path: sglangapiv1.SglangService_Generate_FullMethodName, want: true},

		{path: reqcommon.PathEmbeddings},
		{path: reqcommon.PathImagesGenerations},
		{path: reqcommon.PathImagesEdits},
		{path: reqcommon.PathAudioSpeech},
		{path: reqcommon.PathAudioGenerate},
		{path: reqcommon.PathVideos},
		{path: reqcommon.PathConversations},
		{path: reqcommon.PathModels},
		{path: reqcommon.PathMessages + "/count_tokens"},
		{path: reqcommon.PathChatCompletions + "/render"},
		{path: sglangapiv1.SglangService_HealthCheck_FullMethodName},
		{path: "/vllm.grpc.engine.VllmEngine/Generate"},
		{path: "/grpc.health.v1.Health/Check"},
		{path: ""},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			if got := HasPDRoute(tt.path); got != tt.want {
				t.Errorf("HasPDRoute(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}
