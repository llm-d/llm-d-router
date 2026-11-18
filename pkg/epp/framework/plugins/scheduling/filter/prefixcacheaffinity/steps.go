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

package prefixcacheaffinity

import (
	"errors"
	"fmt"
)

// PrefillSteps is the profile of an engine that prefills in steps.
// TokensPerStep is an engine flag and the times are measured through the
// router; see README.
//
// The engine reads the whole prompt, then computes its uncached part in steps
// of up to TokensPerStep tokens. Every request pending on the engine adds
// RequestMs to the request. A prompt longer than one step shares each of its
// steps with the pending requests, up to RequestsPerStep requests in all. The
// first token follows the request's last step by HandoffMs.
type PrefillSteps struct {
	// TokensPerStep is the number of new prompt tokens one step computes
	// (vLLM --max-num-batched-tokens).
	TokensPerStep int64 `json:"tokensPerStep"`
	// RequestsPerStep is the number of requests the engine works on at a time
	// (vLLM --max-num-seqs).
	RequestsPerStep int64 `json:"requestsPerStep"`
	// StepMs is the time of one step.
	StepMs float64 `json:"stepMs"`
	// RequestMs is the time each request pending on the engine adds to the
	// request. Pending requests share steps, so it is shorter than StepMs.
	// 0 uses StepMs.
	RequestMs float64 `json:"requestMs,omitempty"`
	// ReadMicrosecondsPerToken is the time per prompt token, cached or not, to
	// render and tokenize the prompt on the way to its first step.
	ReadMicrosecondsPerToken float64 `json:"readMicrosecondsPerToken,omitempty"`
	// RestoreMs is the time to bring a cached prefix from outside GPU memory
	// back into it. It is paid once per request, whatever the prefix length.
	RestoreMs float64 `json:"restoreMs,omitempty"`
	// HandoffMs is the time from the step that computed a request's last prompt
	// token to its first response byte.
	HandoffMs float64 `json:"handoffMs,omitempty"`
}

func (s *PrefillSteps) validate() error {
	if s == nil {
		return errors.New("prefillSteps must be set when ttftSource is prefillSteps")
	}
	if s.TokensPerStep <= 0 {
		return fmt.Errorf("prefillSteps.tokensPerStep must be > 0, got %d", s.TokensPerStep)
	}
	if s.RequestsPerStep <= 0 {
		return fmt.Errorf("prefillSteps.requestsPerStep must be > 0, got %d", s.RequestsPerStep)
	}
	if s.StepMs <= 0 {
		return fmt.Errorf("prefillSteps.stepMs must be > 0, got %f", s.StepMs)
	}
	if s.RequestMs < 0 {
		return fmt.Errorf("prefillSteps.requestMs must be >= 0, got %f", s.RequestMs)
	}
	if s.ReadMicrosecondsPerToken < 0 {
		return fmt.Errorf("prefillSteps.readMicrosecondsPerToken must be >= 0, got %f", s.ReadMicrosecondsPerToken)
	}
	if s.RestoreMs < 0 {
		return fmt.Errorf("prefillSteps.restoreMs must be >= 0, got %f", s.RestoreMs)
	}
	if s.HandoffMs < 0 {
		return fmt.Errorf("prefillSteps.handoffMs must be >= 0, got %f", s.HandoffMs)
	}
	return nil
}

// ttftMs estimates the time to first token (ms) of a request on an engine.
// requests is the number of requests pending on the engine ahead of the
// request. tokens is the request's prompt tokens the engine computes,
// cachedTokens the rest of its prompt, and loadMs the wait for its cached prefix to be in
// GPU memory.
func (s *PrefillSteps) ttftMs(requests, tokens, cachedTokens int64, loadMs float64) float64 {
	readMs := s.ReadMicrosecondsPerToken * float64(cachedTokens+tokens) / 1000
	own := max(ceilDiv(tokens, s.TokensPerStep), 1)
	if own > 1 {
		own = ceilDiv(tokens*min(requests+1, s.RequestsPerStep), s.TokensPerStep)
	}
	requestMs := s.RequestMs
	if requestMs == 0 {
		requestMs = s.StepMs
	}
	return loadMs + readMs + float64(requests)*requestMs + float64(own)*s.StepMs + s.HandoffMs
}

func ceilDiv(a, b int64) int64 {
	if a <= 0 {
		return 0
	}
	return (a + b - 1) / b
}
