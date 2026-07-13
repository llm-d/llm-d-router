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

package steps

import (
	"strings"
	"testing"
)

func TestReadErrorBody_CapsOversizedBody(t *testing.T) {
	body := readErrorBody(strings.NewReader(strings.Repeat("a", maxErrorBodySize*4)))
	if len(body) != maxErrorBodySize {
		t.Fatalf("expected body capped to %d bytes, got %d", maxErrorBodySize, len(body))
	}
}

func TestReadErrorBody_ReturnsSmallBodyVerbatim(t *testing.T) {
	body := readErrorBody(strings.NewReader("overloaded"))
	if string(body) != "overloaded" {
		t.Fatalf("expected %q, got %q", "overloaded", string(body))
	}
}
