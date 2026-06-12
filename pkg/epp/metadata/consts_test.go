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

	assert.Equal(t, []string{ObjectiveKey, OldObjectiveKey}, HeaderNames(ObjectiveKey))
	assert.Equal(t, []string{ObjectiveKey, OldObjectiveKey}, HeaderNames("X-LLM-D-Inference-Objective"))
	assert.Equal(t, []string{TTFTSLOHeaderKey, OldTTFTSLOHeaderKey}, HeaderNames(TTFTSLOHeaderKey))
	assert.Equal(t, []string{DestinationEndpointKey}, HeaderNames(DestinationEndpointKey))
	assert.Equal(t, []string{"x-user-header"}, HeaderNames("X-User-Header"))
}

func TestGetLowerCaseHeaderValuePrefersCurrentName(t *testing.T) {
	t.Parallel()

	headers := map[string]string{
		OldObjectiveKey: "old-objective",
		ObjectiveKey:    "new-objective",
	}
	got, ok := GetLowerCaseHeaderValue(headers, ObjectiveKey)
	assert.True(t, ok)
	assert.Equal(t, "new-objective", got)

	headers = map[string]string{
		OldObjectiveKey: "old-objective",
	}
	got, ok = GetLowerCaseHeaderValue(headers, ObjectiveKey)
	assert.True(t, ok)
	assert.Equal(t, "old-objective", got)

	headers = map[string]string{
		"x-user-header": "user-value",
	}
	got, ok = GetLowerCaseHeaderValue(headers, "X-User-Header")
	assert.True(t, ok)
	assert.Equal(t, "user-value", got)
}
