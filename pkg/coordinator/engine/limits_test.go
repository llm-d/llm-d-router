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

package engine

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func TestLimits_PlaceholderOverflow(t *testing.T) {
	limits := Limits{MaxTotalPlaceholderTokens: 5}

	// Two lengths whose sum overflows int and wraps negative. Without the
	// overflow guard, total > max is false and the limit is silently bypassed.
	entries := []pipeline.MultimodalEntry{
		{Placeholder: pipeline.PlaceholderRange{Length: math.MaxInt}},
		{Placeholder: pipeline.PlaceholderRange{Length: math.MaxInt}},
	}
	err := limits.CheckPlaceholderLimit(entries)
	if err == nil {
		t.Fatal("expected error for overflowing placeholder length sum")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
	if !strings.Contains(err.Error(), "too many placeholder tokens") {
		t.Fatalf("unexpected error: %v", err)
	}
}
