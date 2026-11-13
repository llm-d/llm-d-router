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

package request

import (
	"encoding/json"
	"errors"
)

// DecodeRequestBody parses a JSON object body. Only the top-level fields in
// inspected are decoded into Go values; every other field stays a
// json.RawMessage so its bytes are forwarded unchanged. encoding/json sorts
// map keys at every depth on Marshal, which would reorder free-form content
// such as tools[].function.parameters that chat templates render verbatim.
// Each caller passes the set of fields it reads.
func DecodeRequestBody(raw []byte, inspected map[string]struct{}) (map[string]any, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	if fields == nil {
		return nil, errors.New("request body is not a JSON object")
	}
	parsed := make(map[string]any, len(fields))
	for k, v := range fields {
		if _, ok := inspected[k]; !ok {
			parsed[k] = v
			continue
		}
		var decoded any
		if err := json.Unmarshal(v, &decoded); err != nil {
			return nil, err
		}
		parsed[k] = decoded
	}
	return parsed, nil
}
