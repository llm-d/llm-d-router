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

package server

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

// The gRPC request bodies decode with UseNumber, so the token-id and
// placeholder fields hold json.Number. An integer can be written 5, 5.0 or
// 5e0; all three name the same value and NonNegativeInt reads each.
func TestParseUint32ArrayJSONNumberForms(t *testing.T) {
	tests := []struct {
		name  string
		raw   any
		want  []uint32
		errOK bool
	}{
		{name: "integer literal", raw: []any{json.Number("5")}, want: []uint32{5}},
		{name: "decimal form", raw: []any{json.Number("5.0")}, want: []uint32{5}},
		{name: "exponent form", raw: []any{json.Number("1e3")}, want: []uint32{1000}},
		{name: "non-integral rejected", raw: []any{json.Number("5.5")}, errOK: true},
		{name: "above uint32 rejected", raw: []any{json.Number("5000000000")}, errOK: true},
		{name: "float64 decoded", raw: []any{float64(9)}, want: []uint32{9}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseUint32Array(tt.raw, "token_ids", false)
			if tt.errOK {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}
