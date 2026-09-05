/*
Copyright 2025 The llm-d Authors.

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

package collections_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/common/collections"
)

func TestSliceMap(t *testing.T) {
	cases := []struct {
		name  string
		slice []int
		fn    func(int) int
		want  []int
	}{
		{
			name:  "slice is nil",
			slice: nil,
			want:  nil,
		},
		{
			name:  "slice is empty",
			slice: []int{},
			want:  []int{},
		},
		{
			name:  "Get the power of the elements",
			slice: []int{1, 2, 3},
			fn: func(i int) int {
				return i * i
			},
			want: []int{1, 4, 9},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ans := collections.SliceMap(c.slice, c.fn)
			assert.Equal(t, c.want, ans)
		})
	}
}
