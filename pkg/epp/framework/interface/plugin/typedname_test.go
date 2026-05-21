/*
Copyright 2025 The Kubernetes Authors.

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

package plugin

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTypedName_String(t *testing.T) {
	tests := []struct {
		name     string
		tn       TypedName
		expected string
	}{
		{
			name:     "type and name both set",
			tn:       TypedName{Type: "scorer", Name: "kv-cache"},
			expected: "kv-cache/scorer",
		},
		{
			name:     "empty name",
			tn:       TypedName{Type: "scorer", Name: ""},
			expected: "/scorer",
		},
		{
			name:     "empty type",
			tn:       TypedName{Type: "", Name: "instance-1"},
			expected: "instance-1/",
		},
		{
			name:     "both empty",
			tn:       TypedName{},
			expected: "/",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.tn.String())
		})
	}
}
