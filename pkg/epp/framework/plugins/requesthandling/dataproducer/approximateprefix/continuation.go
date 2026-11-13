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

package approximateprefix

import (
	"context"
	"slices"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
)

const continuationStateKey = plugin.StateKey("approx-prefix-continuation")

// continuationState holds what ResponseBody needs to index the generated text
// of a request for the endpoint that served it.
type continuationState struct {
	prompt    string
	server    server
	blockSize int
	// promptIndexed receives once PreRequest has inserted the request's prompt
	// keys. At LRU capacity, a later prompt insert would evict continuation
	// blocks with the prompt's zero-padded last block. The value is the prompt's
	// recent-window entry for the endpoint, or nil when none was recorded.
	promptIndexed chan []blockHash
}

func (c *continuationState) Clone() plugin.StateData {
	clone := *c
	return &clone
}

// recordTokenBackend is the Level-2 matcher on the TokenizedPrompt dependency.
// It accepts every backend and records whether it is the estimate backend,
// the only one whose keys this plugin can derive from response text.
func (p *dataProducer) recordTokenBackend(attrs map[string]string) error {
	p.estimateTokens.Store(attrs["backend"] == tokenproducer.BackendEstimate)
	return nil
}

// continuationPrompt returns the prompt text of a request whose generated
// text can be indexed: a non-streamed /v1/completions request with one string
// prompt, echo disabled, and token IDs equal to the estimate encoding of that
// text, which excludes token IDs supplied by the client or a parser. Only then
// do the keys of prompt+completion match the keys a later request carrying
// that text computes.
func continuationPrompt(request *fwksched.InferenceRequest) (string, bool) {
	if request == nil || request.Body == nil || request.Body.Stream {
		return "", false
	}
	c := request.Body.Completions
	if c == nil || len(c.Prompt.TokenIDs) > 0 || !c.EchoDisabled() {
		return "", false
	}
	prompt := c.Prompt.Raw
	if prompt == "" {
		if len(c.Prompt.Strings) != 1 {
			return "", false
		}
		prompt = c.Prompt.Strings[0]
	}
	tr := request.Body.TokenizedRequest
	if tr == nil || len(tr.Prompts) != 1 || !slices.Equal(tr.Prompts[0].TokenIDs, tokenproducer.PackBytes([]byte(prompt))) {
		return "", false
	}
	return prompt, true
}

// trackContinuation records continuation state for an eligible request, or
// releases the request's plugin state otherwise. A nil target is a decode
// endpoint that is not recorded. For an eligible request it returns the channel
// PreRequest sends the prompt's recent-window entry on after inserting the
// prompt keys.
func (p *dataProducer) trackContinuation(ctx context.Context, request *fwksched.InferenceRequest, target *server, blockSize int) chan []blockHash {
	if !p.config.IndexCompletions || target == nil || !p.estimateTokens.Load() {
		p.pluginState.Delete(request.RequestID)
		return nil
	}
	prompt, ok := continuationPrompt(request)
	if !ok {
		p.pluginState.Delete(request.RequestID)
		return nil
	}
	promptIndexed := make(chan []blockHash, 1)
	p.pluginState.Write(request.RequestID, continuationStateKey, &continuationState{prompt: prompt, server: *target, blockSize: blockSize, promptIndexed: promptIndexed})
	p.pluginState.BindLiveness(ctx, request.RequestID)
	return promptIndexed
}

// ResponseBody indexes the prompt followed by the generated text for the
// endpoint that served the request, and replaces the prompt's entry in the
// endpoint's recent window with it. The prefill endpoint of a disaggregated
// request is not credited.
func (p *dataProducer) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrh.Response, _ *fwkdl.EndpointMetadata) {
	if !p.config.IndexCompletions || request == nil || response == nil || !response.EndOfStream {
		return
	}
	state, err := plugin.ReadPluginStateKey[*continuationState](p.pluginState, request.RequestID, continuationStateKey)
	p.pluginState.Delete(request.RequestID)
	if err != nil || response.Completion == nil || response.TerminationCause != fwkrh.TerminationCauseNatural {
		return
	}
	model, salt, text := request.TargetModel, request.Body.TokenizedRequest.CacheSalt, response.Completion.Text
	p.wg.Go(func() {
		keyed := &fwksched.InferenceRequest{
			TargetModel: model,
			Body: &fwkrequest.InferenceRequestBody{TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts:   []fwkrequest.PromptTokens{{TokenIDs: tokenproducer.PackBytes([]byte(state.prompt + text))}},
				CacheSalt: salt,
			}},
		}
		hashes := prefixhash.GetBlockHashes(context.Background(), keyed, state.blockSize, p.resolveMaxBlocks(state.blockSize))
		if len(hashes) == 0 {
			return
		}
		promptEntry := <-state.promptIndexed
		p.indexerInst.Add(hashes[0], state.server)
		if !p.syncerActive() {
			return
		}
		if promptEntry == nil {
			p.recent.record(state.server.String(), hashes[0])
		} else {
			p.recent.replace(state.server.String(), promptEntry, hashes[0])
		}
	})
}
