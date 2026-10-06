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

package toolcalling

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestResponseSummarySpanAttributes(t *testing.T) {
	attributes := (ResponseSummary{
		ToolCallingRequested:     true,
		UpstreamToolCallPresent:  true,
		ForwardedToolCallPresent: false,
	}).SpanAttributes()

	require.Len(t, attributes, 2)
	require.Equal(t, ResponseAttributeUpstreamToolCallPresent, string(attributes[0].Key))
	require.True(t, attributes[0].Value.AsBool())
	require.Equal(t, ResponseAttributeForwardedToolCallPresent, string(attributes[1].Key))
	require.False(t, attributes[1].Value.AsBool())
}

func TestResponseSummaryOmitsAttributesForNonToolCallingRequest(t *testing.T) {
	attributes := (ResponseSummary{}).SpanAttributes()
	require.Empty(t, attributes)
}
