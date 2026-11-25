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

package metadata

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEscapeTenantID(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "no special characters", in: "tenant-1", want: "tenant-1"},
		{name: "separator", in: "tenant-1/eu", want: "tenant-1%2Feu"},
		{name: "percent sign", in: "tenant-1%2F", want: "tenant-1%252F"},
		{name: "separator and percent sign", in: "a%/b", want: "a%25%2Fb"},
		{name: "multiple separators", in: "a/b/c", want: "a%2Fb%2Fc"},
		{name: "leading separator", in: "/tenant", want: "%2Ftenant"},
		{name: "empty", in: "", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, EscapeTenantID(tt.in))
		})
	}
}
