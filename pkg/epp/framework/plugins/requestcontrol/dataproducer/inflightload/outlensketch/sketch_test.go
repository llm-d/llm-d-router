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
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// newTestSketch builds a sketch over a synthetic 3-token model so the full pipeline
// runs without the production asset. The AdaptiveKey is left effectively dormant
// (large warm-up batch), so the key is the signal signature and the test exercises
// the window -> embed -> key -> histogram path end to end.
func newTestSketch(t *testing.T) *Sketch {
	t.Helper()
	dir := writeTestModel(t,
		[]string{"[UNK]", "solve", "problem"},
		[][]float32{{0, 0}, {1, 0}, {0, 1}},
	)
	cfg := DefaultConfig()
	cfg.ModelDir = dir
	cfg.WarmupSamples = 5
	cfg.WarmupBatch = 1_000_000 // keep the AdaptiveKey warming so the key is signal-only
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func userReq(text string) *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{
		Model:           "m",
		ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: []fwkrh.Message{msg("user", text)}},
		Payload:         fwkrh.PayloadMap{},
	}
}

func TestConfigValidation(t *testing.T) {
	if err := (Config{}).validate(); err == nil {
		t.Error("empty config (no modelDir) must fail validation")
	}
	bad := DefaultConfig()
	bad.ModelDir = "/x"
	bad.Quantile = 1.5
	if err := bad.validate(); err == nil {
		t.Error("quantile > 1 must fail validation")
	}
	ok := DefaultConfig()
	ok.ModelDir = "/x"
	if err := ok.validate(); err != nil {
		t.Errorf("default config with modelDir should validate, got %v", err)
	}
}

func TestSketchColdAbstains(t *testing.T) {
	s := newTestSketch(t)
	if s.Predict(userReq("solve problem")).OK {
		t.Error("a cold sketch must abstain")
	}
}

func TestSketchLearnsMagnitude(t *testing.T) {
	s := newTestSketch(t)
	req := userReq("solve problem")

	// Feed a consistent short workload for this signal signature.
	for i := 0; i < 10; i++ {
		s.Learn(s.Predict(req), 120) // S
	}
	pred := s.Predict(req)
	if !pred.OK {
		t.Fatal("sketch should predict after warmup observations")
	}
	if pred.Magnitude != binMidpoints[0] {
		t.Errorf("learned magnitude = %d, want %d (S midpoint)", pred.Magnitude, binMidpoints[0])
	}
}

func TestSketchNilAndEmptyInputs(t *testing.T) {
	s := newTestSketch(t)
	if s.Predict(nil).OK {
		t.Error("nil body must abstain")
	}
	// A nil-body prediction and a non-positive length are ignored (no panic, no learning).
	s.Learn(s.Predict(nil), 100)
	s.Learn(s.Predict(userReq("solve problem")), 0)
	if s.Predict(userReq("solve problem")).OK {
		t.Error("ignored observations must not warm a key")
	}
}
