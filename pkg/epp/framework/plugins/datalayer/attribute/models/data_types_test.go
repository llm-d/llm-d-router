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

package models

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestModelDataCollectionClone verifies Clone copies entries (including
// MaxModelLen) and is independent of the original.
func TestModelDataCollectionClone(t *testing.T) {
	assert.Nil(t, ModelDataCollection(nil).Clone())

	orig := ModelDataCollection{{ID: "m1", MaxModelLen: 100}, {ID: "m2", MaxModelLen: 200}}
	cloned, ok := orig.Clone().(ModelDataCollection)
	assert.True(t, ok)
	assert.Equal(t, orig, cloned)

	// Mutating the clone must not affect the original.
	cloned[0].MaxModelLen = 999
	assert.Equal(t, 100, orig[0].MaxModelLen)
}
