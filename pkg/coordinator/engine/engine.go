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

// Package engine defines values shared by coordinator engine implementations.
package engine

import (
	"context"
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// DecodeRequest prepares an engine request using its configured transfer protocol.
type DecodeRequest func(context.Context, *pipeline.RequestContext) Request

// Request carries a protocol path and body for the HTTP transport.
type Request struct {
	Path string
	Body map[string]any
}

// Limits bounds prompt tokens and multimodal placeholder tokens. Zero disables a limit.
type Limits struct {
	MaxTotalTokens            int
	MaxTotalPlaceholderTokens int
}

// CheckTokenLimit returns an error wrapping pipeline.ErrBadRequest above the token limit.
func (l Limits) CheckTokenLimit(tokenCount int) error {
	if l.MaxTotalTokens > 0 && tokenCount > l.MaxTotalTokens {
		return fmt.Errorf("too many total tokens: got %d, max %d: %w", tokenCount, l.MaxTotalTokens, pipeline.ErrBadRequest)
	}
	return nil
}

// CheckPlaceholderLimit returns an error wrapping pipeline.ErrBadRequest when the
// sum of non-negative placeholder lengths exceeds the limit or overflows.
func (l Limits) CheckPlaceholderLimit(entries []pipeline.MultimodalEntry) error {
	if l.MaxTotalPlaceholderTokens <= 0 {
		return nil
	}
	total := 0
	for _, e := range entries {
		total += e.Placeholder.Length
		// total < 0 catches int overflow: lengths are non-negative, so a sum
		// past MaxInt wraps negative and would otherwise slip past the
		// total > max check, silently bypassing the limit. On the generate path
		// lengths come straight from the client, so this must not be evadable.
		if total < 0 || total > l.MaxTotalPlaceholderTokens {
			return fmt.Errorf("too many placeholder tokens: got %d, max %d: %w", total, l.MaxTotalPlaceholderTokens, pipeline.ErrBadRequest)
		}
	}
	return nil
}
