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

package tokenizer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// tokenInputProducer turns a request body into a TokenizedRequest. Backends vary
// in fidelity (render vs estimate); callers never branch on which produced it.
type tokenInputProducer interface {
	produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedRequest, error)
}

// timeoutAware is implemented by backends (and the tokenizers they wrap) whose
// produce step can exceed the default data-producer timeout and that manage
// their own. The plugin surfaces it so the director extends its budget.
type timeoutAware interface {
	produceTimeout() time.Duration
}

// produceTimeout reports the wrapped tokenizer's timeout when it manages one.
func (b renderBackend) produceTimeout() time.Duration {
	if ta, ok := b.tk.(timeoutAware); ok {
		return ta.produceTimeout()
	}
	return 0
}

const (
	// warmupImage is a 1x1 PNG data URL used to prime the multimodal processor.
	warmupImage = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

	warmupAttempts      = 24
	warmupRetryInterval = 5 * time.Second
)

// warmer is implemented by backends that prime themselves at load time.
type warmer interface {
	warmup(ctx context.Context)
}

// warmup primes the render path so the first request does not pay the cold-start
// cost. It retries a text render until the backend responds, then issues a
// best-effort multimodal render. It returns on success, on the attempt cap, or
// on context cancellation.
func (b renderBackend) warmup(ctx context.Context) {
	logger := log.FromContext(ctx).V(logutil.DEBUG)
	for i := 0; i < warmupAttempts; i++ {
		if _, err := b.produce(ctx, warmupChat()); err == nil {
			_, _ = b.produce(ctx, warmupChat(warmupImage))
			logger.Info("token-producer backend warmed up", "attempts", i+1)
			return
		}
		select {
		case <-time.After(warmupRetryInterval):
		case <-ctx.Done():
			return
		}
	}
	logger.Info("token-producer backend warmup did not complete")
}

// warmupChat builds a single-message chat body carrying the given image URLs.
func warmupChat(imageURLs ...string) *fwkrh.InferenceRequestBody {
	blocks := make([]fwkrh.ContentBlock, 0, 1+len(imageURLs))
	blocks = append(blocks, fwkrh.ContentBlock{Type: "text", Text: "warmup"})
	for _, url := range imageURLs {
		blocks = append(blocks, fwkrh.ContentBlock{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: url}})
	}
	return &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Structured: blocks}}},
		},
	}
}

// renderBackend produces real token IDs and owns protocol dispatch, including
// the pre-tokenized (Generate) passthrough.
type renderBackend struct {
	tk tokenizer
}

func (b renderBackend) produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedRequest, error) {
	switch {
	case body.Completions != nil:
		if ids := body.Completions.Prompt.TokenIDs; len(ids) > 0 {
			return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: ids}}}, nil
		}
		return b.renderCompletions(ctx, body)
	case body.ChatCompletions != nil:
		tokenIDs, mmFeatures, err := b.tk.RenderChat(ctx, chatPayload(body))
		if err != nil {
			return nil, fmt.Errorf("tokenization failed: %w", err)
		}
		return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{
			TokenIDs:           tokenIDs,
			MultiModalFeatures: convertMMFeaturesToUpstream(mmFeatures),
		}}}, nil
	case body.Generate != nil:
		return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{
			TokenIDs:           body.Generate.TokenIDs,
			MultiModalFeatures: convertMMFeaturesToUpstream(body.Generate.Features),
		}}}, nil
	default:
		return nil, errors.New("unsupported request body type, skipping tokenization")
	}
}

// completionsPayload returns the payload for a completions request. Falls back
// to a minimal PayloadMap when the body carries a non-map payload (gRPC, nil).
// Multi-string prompts are passed as an array so the renderer sees the full
// prompt shape.
func completionsPayload(body *fwkrh.InferenceRequestBody) fwkrh.RequestPayload {
	if body.Payload != nil {
		if _, ok := body.Payload.AsMap(); ok {
			return body.Payload
		}
	}
	prompt := body.Completions.Prompt
	if len(prompt.Strings) > 1 {
		return fwkrh.PayloadMap{"prompt": prompt.Strings}
	}
	return fwkrh.PayloadMap{"prompt": prompt.PlainText()}
}

// chatPayload returns the payload for a chat completions request. Falls back
// to an OpenAI-shaped PayloadMap constructed from the typed struct when the body
// carries a non-map payload (gRPC, warmup).
func chatPayload(body *fwkrh.InferenceRequestBody) fwkrh.RequestPayload {
	if body.Payload != nil {
		if _, ok := body.Payload.AsMap(); ok {
			return body.Payload
		}
	}
	rcr := ChatCompletionsToRenderChatRequest(body.ChatCompletions)
	data, _ := json.Marshal(buildChatRenderRequest("", rcr))
	var pm fwkrh.PayloadMap
	_ = json.Unmarshal(data, &pm)
	return pm
}

// CacheSaltFromBody returns the cache salt from whichever protocol is populated.
// The protocol switch lives here so producers populate TokenizedRequest.CacheSalt
// from one place and consumers read only that field.
func CacheSaltFromBody(body *fwkrh.InferenceRequestBody) string {
	switch {
	case body.Conversations != nil:
		return body.Conversations.CacheSalt
	case body.Responses != nil:
		return body.Responses.CacheSalt
	case body.ChatCompletions != nil:
		return body.ChatCompletions.CacheSalt
	case body.Messages != nil:
		return body.Messages.CacheSalt
	case body.Completions != nil:
		return body.Completions.CacheSalt
	case body.Embeddings != nil:
		return body.Embeddings.CacheSalt
	case body.Generate != nil:
		return body.Generate.CacheSalt
	default:
		return ""
	}
}

// renderCompletions tokenizes a completions prompt via a single Render call.
// completionsPayload builds the appropriate payload shape (single string or
// string array), and the renderer returns the tokenized result.
func (b renderBackend) renderCompletions(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedRequest, error) {
	allTokenIDs, _, err := b.tk.Render(ctx, completionsPayload(body))
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}
	prompts := make([]fwkrh.PromptTokens, len(allTokenIDs))
	for i, ids := range allTokenIDs {
		prompts[i] = fwkrh.PromptTokens{TokenIDs: ids}
	}
	return &fwkrh.TokenizedRequest{Prompts: prompts}, nil
}
