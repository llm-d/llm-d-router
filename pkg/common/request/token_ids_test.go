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
	"math"
	"strings"
	"testing"
)

func TestParseTokenIDs(t *testing.T) {
	tests := []struct {
		name   string
		values []any
		want   []uint32
		errOK  bool
	}{
		{name: "empty", values: []any{}, want: []uint32{}},
		{name: "float64 decoded", values: []any{float64(1), float64(32000)}, want: []uint32{1, 32000}},
		{name: "json.Number decoded", values: []any{json.Number("1"), json.Number("32000")}, want: []uint32{1, 32000}},
		{name: "json.Number decimal form", values: []any{json.Number("5.0")}, want: []uint32{5}},
		{name: "in-process int", values: []any{7}, want: []uint32{7}},
		{name: "in-process int64", values: []any{int64(11)}, want: []uint32{11}},
		{name: "negative rejected", values: []any{float64(-1)}, errOK: true},
		{name: "non-integral rejected", values: []any{float64(1.5)}, errOK: true},
		{name: "NaN rejected", values: []any{math.NaN()}, errOK: true},
		{name: "+Inf rejected", values: []any{math.Inf(1)}, errOK: true},
		{name: "-Inf rejected", values: []any{math.Inf(-1)}, errOK: true},
		{name: "string rejected", values: []any{"1"}, errOK: true},
		{name: "above uint32 rejected", values: []any{float64(5000000000)}, errOK: true},
		{name: "max uint32 accepted", values: []any{float64(math.MaxUint32)}, want: []uint32{math.MaxUint32}},
		{name: "max uint32 plus one rejected", values: []any{float64(math.MaxUint32) + 1}, errOK: true},
		{name: "one bad element fails the array", values: []any{float64(1), "x"}, errOK: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseTokenIDs(tt.values)
			if tt.errOK {
				if err == nil {
					t.Fatalf("ParseTokenIDs(%v) = %v, want error", tt.values, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseTokenIDs(%v) error = %v", tt.values, err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("ParseTokenIDs(%v) = %v, want %v", tt.values, got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("ParseTokenIDs(%v) = %v, want %v", tt.values, got, tt.want)
				}
			}
		})
	}
}

func TestParseTokenIDsNamesTheOffendingIndex(t *testing.T) {
	tests := []struct {
		name   string
		values []any
	}{
		{name: "not a number", values: []any{float64(1), float64(2), "x"}},
		{name: "above uint32", values: []any{float64(1), float64(2), float64(5000000000)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseTokenIDs(tt.values)
			if err == nil {
				t.Fatal("ParseTokenIDs expected an error")
			}
			if !strings.Contains(err.Error(), "index 2") {
				t.Fatalf("error should name index 2, got: %v", err)
			}
		})
	}
}
