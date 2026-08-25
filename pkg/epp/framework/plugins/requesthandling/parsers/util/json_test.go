/*
Copyright 2026 The Kubernetes Authors.

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

package parserutil

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestUnmarshalUsesNumber(t *testing.T) {
	var got any
	if err := Unmarshal([]byte(`{"integer":9007199254740993,"nested":[1.5]}  `), &got); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	want := map[string]any{
		"integer": json.Number("9007199254740993"),
		"nested":  []any{json.Number("1.5")},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Unmarshal() mismatch (-want +got):\n%s", diff)
	}
}

func TestUnmarshalRejectsTrailingData(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantError string
	}{
		{
			name:      "second JSON value",
			input:     `{} {}`,
			wantError: ErrTrailingData.Error(),
		},
		{
			name:      "malformed trailing data",
			input:     `{} trailing`,
			wantError: "unexpected trailing data after JSON value: invalid character 'a' in literal true (expecting 'u')",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got any
			err := Unmarshal([]byte(tt.input), &got)
			if !errors.Is(err, ErrTrailingData) {
				t.Fatalf("Unmarshal() error = %v, want unexpected trailing data", err)
			}
			if err.Error() != tt.wantError {
				t.Errorf("Unmarshal() error = %q, want %q", err, tt.wantError)
			}
		})
	}
}
