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

// Package vtc implements a Virtual Token Count (VTC) fairness policy for flow control.
//
// Unlike round-robin (which gives each flow one turn regardless of request size), VTC tracks a
// cumulative virtual token cost per flow. The flow with the lowest virtual counter gets the next
// dispatch opportunity, ensuring tenants sending large prompts do not starve tenants sending small
// ones.
//
// Cost function:
//
//	cost = inputTokenWeight × inputTokens + outputTokenWeight × outputTokens
//
// Input tokens are estimated via a cascade: client-provided hint → pre-tokenized prompt length →
// prompt character count / 4 → request byte size / 4. Output tokens default to 128 unless
// configured otherwise.
//
// Starvation prevention: when a flow's counter does not yet exist (new or rejoining after idle),
// its counter is initialized to the current minimum among active flows rather than zero, preventing
// it from monopolizing dispatch turns when it returns.
package vtc

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const (
	// VTCFairnessPolicyType is the registration key for the VTC fairness policy.
	VTCFairnessPolicyType = "vtc-fairness-policy"

	// normalizationThreshold prevents unbounded float64 growth. When any counter exceeds this,
	// the global minimum is subtracted from all counters to preserve relative differences.
	normalizationThreshold = 1e12

	// defaultOutputTokenEstimate is used when the request does not carry an explicit max_tokens hint.
	defaultOutputTokenEstimate = 128.0
)

// VTCFairnessPolicyFactory creates a VTC fairness policy from optional JSON parameters.
//
// Supported parameters:
//
//	{
//	  "inputTokenWeight": 1.0,
//	  "outputTokenWeight": 3.0
//	}
//
// outputTokenWeight defaults to 3.0 to reflect that autoregressive decode tokens consume
// sequentially more cache and execution time than prefill tokens.
func VTCFairnessPolicyFactory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := vtcConfig{
		InputTokenWeight:  1.0,
		OutputTokenWeight: 3.0,
	}
	if parameters != nil {
		if err := parameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse VTC parameters: %w", err)
		}
	}
	return newVTCFromConfig(name, cfg)
}

// vtcConfig holds the JSON-parsed configuration for the VTC policy.
type vtcConfig struct {
	InputTokenWeight  float64 `json:"inputTokenWeight"`
	OutputTokenWeight float64 `json:"outputTokenWeight"`
}

// vtc implements FairnessPolicy using Virtual Token Counting.
// The struct is immutable after construction and shared across all priority bands (Singleton).
type vtc struct {
	name              string
	inputTokenWeight  float64
	outputTokenWeight float64
}

func newVTC(name string, params json.RawMessage) (*vtc, error) {
	cfg := vtcConfig{
		InputTokenWeight:  1.0,
		OutputTokenWeight: 3.0,
	}

	if len(params) > 0 {
		if err := json.Unmarshal(params, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse VTC parameters: %w", err)
		}
	}

	return newVTCFromConfig(name, cfg)
}

func newVTCFromConfig(name string, cfg vtcConfig) (*vtc, error) {
	if name == "" {
		name = VTCFairnessPolicyType
	}

	if cfg.InputTokenWeight <= 0 {
		cfg.InputTokenWeight = 1.0
	}
	if cfg.OutputTokenWeight <= 0 {
		cfg.OutputTokenWeight = 3.0
	}

	return &vtc{
		name:              name,
		inputTokenWeight:  cfg.InputTokenWeight,
		outputTokenWeight: cfg.OutputTokenWeight,
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *vtc) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: VTCFairnessPolicyType,
		Name: p.name,
	}
}

// vtcBandState holds the mutable per-band state for the VTC policy (Flyweight pattern).
type vtcBandState struct {
	mu       sync.Mutex
	counters map[string]float64 // flow ID → cumulative virtual token cost
}

// NewState initializes the policy state for a specific priority band.
func (p *vtc) NewState(_ context.Context) any {
	return &vtcBandState{
		counters: make(map[string]float64),
	}
}

// Pick selects the flow with the lowest virtual token counter from the given priority band.
//
// Algorithm:
//  1. Sort active flow keys for deterministic tie-breaking.
//  2. For each non-empty queue, initialize its counter (with counter-lift if rejoining).
//  3. Select the flow with the smallest counter.
//  4. Advance the winner's counter by its estimated token cost.
//  5. Prune counters for flows no longer in the active set.
//  6. Normalize counters if any value exceeds the threshold.
func (p *vtc) Pick(
	_ context.Context,
	flowGroup flowcontrol.PriorityBandAccessor,
) (flowcontrol.FlowQueueAccessor, error) {
	if flowGroup == nil {
		return nil, nil //nolint:nilnil
	}

	v := flowGroup.PolicyState()
	s, ok := v.(*vtcBandState)
	if !ok {
		return nil, fmt.Errorf("invalid state type for VTC policy: expected *vtcBandState, got %T", v)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	keys := flowGroup.FlowKeys()
	if len(keys) == 0 {
		return nil, nil //nolint:nilnil
	}

	// Sort for deterministic tie-breaking.
	slices.SortFunc(keys, func(a, b flowcontrol.FlowKey) int { return a.Compare(b) })

	// Compute the minimum counter among active queues that already have a counter.
	// Used to lift rejoining flows so they don't monopolize dispatch turns.
	activeMin := math.MaxFloat64
	for _, key := range keys {
		queue := flowGroup.Queue(key.ID)
		if queue == nil || queue.Len() == 0 {
			continue
		}
		if c, exists := s.counters[key.ID]; exists && c < activeMin {
			activeMin = c
		}
	}
	if activeMin == math.MaxFloat64 {
		activeMin = 0
	}

	// Find the non-empty flow with the lowest virtual counter.
	var bestQueue flowcontrol.FlowQueueAccessor
	bestCounter := math.MaxFloat64
	activeIDs := make(map[string]struct{}, len(keys))

	for _, key := range keys {
		activeIDs[key.ID] = struct{}{}

		queue := flowGroup.Queue(key.ID)
		if queue == nil || queue.Len() == 0 {
			continue
		}

		// Counter-lift: a new or rejoining flow starts at activeMin to avoid unfairly consuming
		// all dispatch capacity upon return.
		if _, exists := s.counters[key.ID]; !exists {
			s.counters[key.ID] = activeMin
		}

		if s.counters[key.ID] < bestCounter {
			bestCounter = s.counters[key.ID]
			bestQueue = queue
		}
	}

	if bestQueue == nil {
		return nil, nil //nolint:nilnil
	}

	// Advance the winner's counter by its estimated token cost.
	winnerID := bestQueue.FlowKey().ID
	s.counters[winnerID] += p.estimateCost(bestQueue)

	// Prune counters for flows no longer in the active set.
	for id := range s.counters {
		if _, active := activeIDs[id]; !active {
			delete(s.counters, id)
		}
	}

	normalizeCounters(s)

	return bestQueue, nil
}

// estimateCost computes the weighted token cost for the head item of a queue.
func (p *vtc) estimateCost(queue flowcontrol.FlowQueueAccessor) float64 {
	head := queue.Peek()
	if head == nil {
		return p.inputTokenWeight + p.outputTokenWeight*defaultOutputTokenEstimate
	}

	req := head.OriginalRequest()
	if req == nil {
		return p.inputTokenWeight + p.outputTokenWeight*defaultOutputTokenEstimate
	}

	inputTokens := estimateInputTokens(req)
	cost := p.inputTokenWeight*inputTokens + p.outputTokenWeight*defaultOutputTokenEstimate
	if cost <= 0 {
		// Floor to prevent counter stagnation on zero-cost requests.
		return 1
	}
	return cost
}

// estimateInputTokens estimates the input token count from the request via a cascade of heuristics.
//
// Priority order:
//  1. Pre-tokenized input token IDs (generate, completions, embeddings).
//  2. Length of a pre-tokenized prompt in InferenceRequestBody.TokenizedPrompt.
//  3. Character-based estimate: prompt text length / 4.
//  4. Raw request size estimate: InferenceRequest.RequestSizeBytes / 4.
//  5. Final fallback: FlowControlRequest.ByteSize() / 4.
func estimateInputTokens(req flowcontrol.FlowControlRequest) float64 {
	ir := req.InferenceRequest()
	if ir != nil && ir.Body != nil {
		if count := preTokenizedInputCount(ir.Body); count > 0 {
			return float64(count)
		}

		if ir.Body.TokenizedPrompt != nil {
			if count := ir.Body.TokenizedPrompt.TokenCount(); count > 0 {
				return float64(count)
			}
		}

		if textLen := promptTextLength(ir.Body); textLen > 0 {
			return float64(textLen) / 4.0
		}

		if ir.RequestSizeBytes > 0 {
			return float64(ir.RequestSizeBytes) / 4.0
		}
	}

	if bs := req.ByteSize(); bs > 0 {
		return float64(bs) / 4.0
	}

	return 1 // minimum to avoid zero-cost dispatch
}

func preTokenizedInputCount(body *fwkrh.InferenceRequestBody) int {
	switch {
	case body.Generate != nil && len(body.Generate.TokenIDs) > 0:
		return len(body.Generate.TokenIDs)
	case body.Completions != nil:
		if hint := body.Completions.Prompt.TokenCountHint(); hint > 0 {
			return hint
		}
	case body.Embeddings != nil:
		if hint := body.Embeddings.Input.TokenCountHint(); hint > 0 {
			return hint
		}
	}
	return 0
}

func promptTextLength(body *fwkrh.InferenceRequestBody) int {
	switch {
	case body.Completions != nil:
		return len(body.Completions.Prompt.PlainText())
	case body.Embeddings != nil:
		return len(body.Embeddings.Input.PlainText())
	case body.Images != nil:
		return len(body.Images.Prompt)
	case body.ChatCompletions != nil:
		n := 0
		for _, msg := range body.ChatCompletions.Messages {
			n += len(msg.Content.PlainText())
		}
		return n
	case body.Messages != nil:
		n := len(body.Messages.System.Raw)
		for _, block := range body.Messages.System.Structured {
			if block.Type == "text" {
				n += len(block.Text)
			}
		}
		for _, msg := range body.Messages.Messages {
			if msg.Content.Raw != "" {
				n += len(msg.Content.Raw)
				continue
			}
			for _, block := range msg.Content.Structured {
				if block.Type == "text" {
					n += len(block.Text)
				}
			}
		}
		return n
	default:
		return 0
	}
}

// normalizeCounters subtracts the global minimum from all counters when any counter exceeds the
// threshold, preserving relative differences while preventing unbounded growth.
func normalizeCounters(s *vtcBandState) {
	var maxC float64
	for _, c := range s.counters {
		if c > maxC {
			maxC = c
		}
	}
	if maxC <= normalizationThreshold {
		return
	}

	minC := math.MaxFloat64
	for _, c := range s.counters {
		if c < minC {
			minC = c
		}
	}
	for id := range s.counters {
		s.counters[id] -= minC
	}
}
