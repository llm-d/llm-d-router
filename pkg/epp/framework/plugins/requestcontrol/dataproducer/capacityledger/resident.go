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

package capacityledger

import (
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// residentState is one request's state on an endpoint.
type residentState struct {
	// prompts holds each prompt's length in tokens; one entry for a single-prompt request.
	prompts []int64
	// uncachedPrompt is the prompt tokens the endpoint must prefill, summed over prompts.
	uncachedPrompt int64
	// n is the number of sequences generated per prompt.
	n int64
	// age is the output tokens each sequence has generated.
	age int64
	// decoding is set once the request has finished prefill.
	decoding bool
}

func (r residentState) sequences() int64 { return r.n * int64(len(r.prompts)) }

// geometry is an endpoint's parameters that convert a request's state into a charge.
type geometry struct {
	blockSize int64
	// stepBudget is the engine's per-step token budget.
	stepBudget int64
	// decodeStep is the tokens one decode step schedules per sequence, and the KV slots one
	// sequence holds beyond its prompt and output: the next token plus the speculative tokens.
	decodeStep int64
}

// footprint is the charge a request in state r places on an endpoint with geometry g.
//   - Memory: each sequence holds its prompt, its output so far and its next decode step, rounded
//     up to blocks per sequence. The prefix match does not reduce memory: a hit on a cached block
//     that no running request references moves that block out of the engine's free pool.
//   - Step: while prefilling, the uncached prompt of every sequence, capped at the step budget,
//     since the engine prefills a larger prompt in chunks across steps; while decoding, one decode
//     step per sequence.
//   - Slots: one per sequence.
func footprint(r residentState, g geometry) vec {
	var c vec
	for _, p := range r.prompts {
		c[axisMemory] += r.n * blocksFor(p+r.age+g.decodeStep, g.blockSize)
	}
	if r.decoding {
		c[axisStep] = r.sequences() * g.decodeStep
	} else {
		c[axisStep] = min(r.n*r.uncachedPrompt, g.stepBudget)
	}
	c[axisSlots] = r.sequences()
	return c
}

// blocksFor rounds tokens up to whole blocks. It returns 0 when the block size is unknown.
func blocksFor(tokens, blockSize int64) int64 {
	if blockSize <= 0 || tokens <= 0 {
		return 0
	}
	return (tokens + blockSize - 1) / blockSize
}

// promptLengths returns each prompt's length in tokens. Without a tokenized prompt it returns one
// entry bounded by bytes, the request's size in bytes, which exceeds the token count for text.
func promptLengths(req *fwksched.InferenceRequest, bytes int64) []int64 {
	if req != nil && req.Body != nil && req.Body.TokenizedRequest != nil {
		lengths := make([]int64, 0, len(req.Body.TokenizedRequest.Prompts))
		for _, p := range req.Body.TokenizedRequest.Prompts {
			lengths = append(lengths, int64(len(p.TokenIDs)))
		}
		if len(lengths) > 0 {
			return lengths
		}
	}
	return []int64{max(bytes, 0)}
}

func sum(xs []int64) int64 {
	var s int64
	for _, x := range xs {
		s += x
	}
	return s
}

// sequencesPerPrompt returns the request's n, the number of sequences it generates per prompt,
// clamped to limit. n is client-supplied, and the engine runs at most its sequence limit at once.
func sequencesPerPrompt(req *fwksched.InferenceRequest, limit int64) int64 {
	if req == nil || req.Body == nil || req.Body.Payload == nil {
		return 1
	}
	m, ok := req.Body.Payload.AsMap()
	if !ok {
		return 1
	}
	if n := fwkrh.MaxOutputTokensFromPayload(m, "n"); n != nil && *n > 1 {
		return max(min(*n, limit), 1)
	}
	return 1
}

// uncachedPromptTokens returns the prompt tokens the endpoint must prefill: the indexed blocks it
// has not cached, plus any prompt beyond the portion the prefix index hashes, and at least one
// token for a non-empty prompt so the engine computes the first output token's logits. Cached
// blocks are counted without device-tier weighting, since a block cached in any tier is not
// prefilled again.
func uncachedPromptTokens(endpoint fwksched.Endpoint, prompt int64, prefixKey fwkplugin.DataKey) int64 {
	if prompt <= 0 {
		return 0
	}
	raw, ok := endpoint.Get(prefixKey)
	if !ok {
		return prompt
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok || info == nil || info.BlockSizeTokens() <= 0 {
		return prompt
	}
	blockSize := int64(info.BlockSizeTokens())
	indexed := int64(info.TotalBlocks()) * blockSize
	cached := min(int64(info.CachedBlockCount())*blockSize, indexed)
	return min(max(max(indexed-cached, 0)+max(prompt-indexed, 0), 1), prompt)
}
