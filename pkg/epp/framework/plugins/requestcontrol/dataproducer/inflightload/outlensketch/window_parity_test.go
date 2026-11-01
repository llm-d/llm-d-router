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

package outlensketch

import (
	"encoding/json"
	"os"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// Parity against the reference window.py W1-1024 recipe. Skipped unless
// OSL_WINDOW_FIXTURE is set (a JSON dump of message lists and their reference
// windowed text), so it never runs in CI without the fixture.
func TestWindowParity(t *testing.T) {
	fixturePath := os.Getenv("OSL_WINDOW_FIXTURE")
	if fixturePath == "" {
		t.Skip("set OSL_WINDOW_FIXTURE to run window.py parity")
	}
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx struct {
		Cases []struct {
			Messages [][2]string `json:"messages"`
			Windowed string      `json:"windowed"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}

	for i, c := range fx.Cases {
		messages := make([]fwkrh.Message, len(c.Messages))
		for j, m := range c.Messages {
			messages[j] = fwkrh.Message{Role: m[0], Content: fwkrh.Content{Raw: m[1]}}
		}
		if got := applyWindow(messages); got != c.Windowed {
			t.Errorf("case %d window mismatch:\n got=%q\nwant=%q", i, got, c.Windowed)
		}
	}
}
