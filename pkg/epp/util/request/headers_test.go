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

package request

import (
	"testing"

	"github.com/stretchr/testify/assert"

	errcommon "github.com/llm-d/llm-d-inference-scheduler/pkg/common/error"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/metadata"
)

func TestIsSystemOwnedHeaderIncludesAliases(t *testing.T) {
	t.Parallel()

	systemHeaders := []string{
		metadata.FlowFairnessIDKey,
		metadata.OldFlowFairnessIDKey,
		metadata.ObjectiveKey,
		metadata.OldObjectiveKey,
		metadata.ModelNameRewriteKey,
		metadata.OldModelNameRewriteKey,
		metadata.SubsetFilterKey,
		metadata.OldSubsetFilterKey,
		metadata.TTFTSLOHeaderKey,
		metadata.OldTTFTSLOHeaderKey,
		metadata.TPOTSLOHeaderKey,
		metadata.OldTPOTSLOHeaderKey,
		metadata.DestinationEndpointKey,
		metadata.OldDestinationEndpointKey,
		metadata.DestinationEndpointServedKey,
		metadata.OldDestinationEndpointServedKey,
		errcommon.RequestDroppedReasonHeaderKey,
		errcommon.OldRequestDroppedReasonHeaderKey,
		"Content-Length",
	}

	for _, header := range systemHeaders {
		assert.True(t, IsSystemOwnedHeader(header), "header %q should be system-owned", header)
	}
	assert.False(t, IsSystemOwnedHeader("x-user-data"))
}
