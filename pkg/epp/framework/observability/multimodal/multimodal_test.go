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

package multimodal

import (
	"testing"

	"github.com/stretchr/testify/assert"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestSummary(t *testing.T) {
	tests := []struct {
		name          string
		req           *scheduling.InferenceRequest
		wantModality  string
		wantHashCount int
	}{
		{
			name:          "nil request",
			req:           nil,
			wantModality:  ModalityNone,
			wantHashCount: 0,
		},
		{
			name:          "nil body",
			req:           &scheduling.InferenceRequest{Body: nil},
			wantModality:  ModalityNone,
			wantHashCount: 0,
		},
		{
			name:          "no tokenized prompt",
			req:           &scheduling.InferenceRequest{Body: &fwkrh.InferenceRequestBody{}},
			wantModality:  ModalityNone,
			wantHashCount: 0,
		},
		{
			name:          "empty multimodal features",
			req:           requestWith(),
			wantModality:  ModalityNone,
			wantHashCount: 0,
		},
		{
			name: "image only",
			req: requestWith(
				fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage, Hash: "h1"},
				fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage, Hash: "h2"},
			),
			wantModality:  "image",
			wantHashCount: 2,
		},
		{
			name: "mixed modalities sorted and comma-joined",
			req: requestWith(
				fwkrh.MultiModalFeature{Modality: "video", Hash: "v1"},
				fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage, Hash: "i1"},
				fwkrh.MultiModalFeature{Modality: "audio", Hash: "a1"},
			),
			wantModality:  "audio,image,video",
			wantHashCount: 3,
		},
		{
			name: "missing modality strings still count toward hash_count",
			req: requestWith(
				fwkrh.MultiModalFeature{Hash: "h1"},
				fwkrh.MultiModalFeature{Hash: "h2"},
			),
			wantModality:  ModalityNone,
			wantHashCount: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modality, hashCount := Summary(tt.req)
			assert.Equal(t, tt.wantModality, modality)
			assert.Equal(t, tt.wantHashCount, hashCount)
		})
	}
}

func TestSpanAttributes(t *testing.T) {
	attrs := SpanAttributes(requestWith(
		fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage, Hash: "h1"},
	))
	assert.Len(t, attrs, 2)
	assert.Equal(t, "mm.modality", string(attrs[0].Key))
	assert.Equal(t, "image", attrs[0].Value.AsString())
	assert.Equal(t, "mm.hash_count", string(attrs[1].Key))
	assert.Equal(t, int64(1), attrs[1].Value.AsInt64())
}

func requestWith(features ...fwkrh.MultiModalFeature) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				MultiModalFeatures: features,
			},
		},
	}
}
