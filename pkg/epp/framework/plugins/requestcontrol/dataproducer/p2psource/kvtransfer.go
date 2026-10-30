/*
Copyright 2025 The Kubernetes Authors.
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
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

// kvTransferParams returns the body's kv_transfer_params object, or nil when
// the body is not a parsed JSON object or the field is absent or not an
// object. The OpenAI parser keeps the field as json.RawMessage, so an edit
// takes effect only once setKVTransferParams stores the decoded object.
func kvTransferParams(body *fwkrh.InferenceRequestBody) map[string]any {
	if body == nil {
		return nil
	}
	payload, ok := body.Payload.(fwkrh.PayloadMap)
	if !ok {
		return nil
	}
	params, _ := parserutil.ObjectField(payload, reqcommon.FieldKVTransferParams)
	return params
}

// setKVTransferParams stores params as the body's kv_transfer_params and
// reports whether it did. A nil body, or one a parser kept as raw bytes, is
// left as is.
func setKVTransferParams(body *fwkrh.InferenceRequestBody, params map[string]any) bool {
	if body == nil {
		return false
	}
	if _, ok := body.Payload.(fwkrh.PayloadMap); !ok {
		return false
	}
	body.MutatePayloadMap(func(payload fwkrh.PayloadMap) {
		payload[reqcommon.FieldKVTransferParams] = params
	})
	return true
}
