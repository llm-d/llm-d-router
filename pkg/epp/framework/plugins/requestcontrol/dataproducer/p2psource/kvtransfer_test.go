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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

func TestKVTransferParamsHelper(t *testing.T) {
	tests := []struct {
		name    string
		payload fwkrh.RequestPayload
		want    map[string]any
	}{
		{
			name:    "decoded object",
			payload: fwkrh.PayloadMap{reqcommon.FieldKVTransferParams: map[string]any{"other": true}},
			want:    map[string]any{"other": true},
		},
		{
			name:    "raw object is decoded",
			payload: fwkrh.PayloadMap{reqcommon.FieldKVTransferParams: json.RawMessage(`{"remote_port":7777}`)},
			want:    map[string]any{"remote_port": json.Number("7777")},
		},
		{
			name:    "no kv_transfer_params",
			payload: fwkrh.PayloadMap{"model": "m"},
		},
		{
			name:    "kv_transfer_params that is not an object",
			payload: fwkrh.PayloadMap{reqcommon.FieldKVTransferParams: "text"},
		},
		{
			name:    "raw payload",
			payload: fwkrh.RawPayload(`{"kv_transfer_params":{}}`),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := &fwkrh.InferenceRequestBody{Payload: tt.payload}
			assert.Equal(t, tt.want, kvTransferParams(body))
			assert.False(t, body.Mutated, "reading must not mark the body mutated")
		})
	}
	assert.Nil(t, kvTransferParams(nil))
}

func TestSetKVTransferParamsHelper(t *testing.T) {
	params := map[string]any{"other": true}

	body := &fwkrh.InferenceRequestBody{Payload: fwkrh.PayloadMap{reqcommon.FieldKVTransferParams: json.RawMessage(`{}`)}}
	assert.True(t, setKVTransferParams(body, params))
	assert.True(t, body.Mutated)
	assert.Equal(t, params, body.Payload.(fwkrh.PayloadMap)[reqcommon.FieldKVTransferParams])

	raw := &fwkrh.InferenceRequestBody{Payload: fwkrh.RawPayload(`{}`)}
	assert.False(t, setKVTransferParams(raw, params))
	assert.Equal(t, fwkrh.RawPayload(`{}`), raw.Payload, "a raw payload is left as is")
	assert.False(t, raw.Mutated)

	assert.False(t, setKVTransferParams(nil, params))
}
