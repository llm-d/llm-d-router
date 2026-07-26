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

package datalayer

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMultiClusterAttribute(t *testing.T) {
	tests := []struct {
		name string
		mark bool
		want bool
	}{
		{name: "unmarked reads as not cross-cluster", mark: false, want: false},
		{name: "marked reads as cross-cluster", mark: true, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ep := NewEndpoint(nil, nil)
			if tt.mark {
				SetMultiCluster(ep)
			}
			assert.Equal(t, tt.want, IsMultiCluster(ep))
		})
	}
}
