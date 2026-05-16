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

package metadata

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHeaderAliases(t *testing.T) {
	t.Parallel()

	assert.Equal(t, ObjectiveKey, CanonicalHeaderKey(OldObjectiveKey))
	assert.Equal(t, ObjectiveKey, CanonicalHeaderKey(ObjectiveKey))
	assert.Equal(t, ObjectiveKey, CanonicalHeaderKey("X-Gateway-Inference-Objective"))
	assert.Equal(t, []string{ObjectiveKey, OldObjectiveKey}, HeaderNames(ObjectiveKey))
	assert.Equal(t, []string{TTFTSLOHeaderKey, OldTTFTSLOHeaderKey}, HeaderNames(TTFTSLOHeaderKey))
	assert.True(t, IsManagedHeader(OldTTFTSLOHeaderKey))
	assert.False(t, IsManagedHeader("x-user-header"))
}

func TestGetHeaderPrefersCurrentName(t *testing.T) {
	t.Parallel()

	headers := map[string]string{
		OldObjectiveKey: "old-objective",
		ObjectiveKey:    "new-objective",
	}
	assert.Equal(t, "new-objective", GetHeader(headers, ObjectiveKey))

	headers = map[string]string{
		OldObjectiveKey: "old-objective",
	}
	assert.Equal(t, "old-objective", GetHeader(headers, ObjectiveKey))

	headers = map[string]string{
		"X-LLM-D-Router-Inference-Objective": "mixed-case-objective",
	}
	assert.Equal(t, "mixed-case-objective", GetHeader(headers, ObjectiveKey))
}

func TestGetValuePrefersCurrentName(t *testing.T) {
	t.Parallel()

	values := map[string]any{
		OldSubsetFilterKey: []any{"10.0.0.2:8080"},
		SubsetFilterKey:    []any{"10.0.0.1:8080"},
	}

	got, ok := GetValue(values, SubsetFilterKey)
	assert.True(t, ok)
	assert.Equal(t, []any{"10.0.0.1:8080"}, got)
}
