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

import "go.opentelemetry.io/otel/attribute"

const (
	ResponseAttributeUpstreamToolCallPresent  = "llm_d.tool_calling.response.upstream_present"
	ResponseAttributeForwardedToolCallPresent = "llm_d.tool_calling.response.forwarded_present"
)

// ResponseSummary contains only bounded presence information. It deliberately
// does not retain response bodies, tool names, or tool arguments.
type ResponseSummary struct {
	ToolCallingRequested     bool
	UpstreamToolCallPresent  bool
	ForwardedToolCallPresent bool
}

// SpanAttributes returns response presence attributes only for requests that
// contained tool-calling fields. This keeps non-tool-calling spans unchanged.
func (summary ResponseSummary) SpanAttributes() []attribute.KeyValue {
	if !summary.ToolCallingRequested {
		return nil
	}

	return []attribute.KeyValue{
		attribute.Bool(ResponseAttributeUpstreamToolCallPresent, summary.UpstreamToolCallPresent),
		attribute.Bool(ResponseAttributeForwardedToolCallPresent, summary.ForwardedToolCallPresent),
	}
}
