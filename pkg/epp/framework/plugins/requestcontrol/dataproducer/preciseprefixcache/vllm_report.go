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

package preciseprefixcache

import (
	"encoding/json"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// vllmFullReport sets vllm_xargs.kv_cache_report_mode to "full". Native
// Generate requests carry vllm_xargs under sampling_params. Bodies that are
// not JSON maps, or whose argument containers are not JSON objects, cannot
// carry the argument.
func vllmFullReport(body *fwkrh.InferenceRequestBody) (func(fwkrh.PayloadMap), bool) {
	if body == nil || body.Payload == nil {
		return nil, false
	}
	payload, ok := body.Payload.AsMap()
	if !ok {
		return nil, false
	}
	parent := map[string]any(payload)
	if body.Generate != nil {
		if parent, ok = jsonObject(payload["sampling_params"]); !ok {
			return nil, false
		}
	}
	xargs, ok := jsonObject(parent["vllm_xargs"])
	if !ok {
		return nil, false
	}
	return func(payload fwkrh.PayloadMap) {
		xargs["kv_cache_report_mode"] = "full"
		parent["vllm_xargs"] = xargs
		if body.Generate != nil {
			payload["sampling_params"] = parent
		}
	}, true
}

// jsonObject returns value as a mutable JSON object, or false when value is
// not one. An absent value is an empty object. An opaque object decodes to its
// raw members, so re-marshaling preserves their encoding.
func jsonObject(value any) (map[string]any, bool) {
	switch value := value.(type) {
	case nil:
		return map[string]any{}, true
	case map[string]any:
		return value, true
	case fwkrh.PayloadMap:
		return value, true
	case json.RawMessage:
		var members map[string]json.RawMessage
		if err := json.Unmarshal(value, &members); err != nil {
			return nil, false
		}
		object := make(map[string]any, len(members)+1)
		for name, member := range members {
			object[name] = member
		}
		return object, true
	default:
		return nil, false
	}
}
