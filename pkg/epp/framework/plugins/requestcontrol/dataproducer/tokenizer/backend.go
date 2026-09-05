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
// best-effort multimodal render. It returns on success, on the attempt cap, on
// an authentication rejection, or on context cancellation.
func (b renderBackend) warmup(ctx context.Context) {
	logger := log.FromContext(ctx)
	for i := 0; i < warmupAttempts; i++ {
		_, err := b.produce(ctx, warmupChat(b.modelName))
		if err == nil {
			_, _ = b.produce(ctx, warmupChat(b.modelName, warmupImage))
			logger.V(logutil.DEBUG).Info("token-producer backend warmed up", "attempts", i+1)
			return
		}
		// Warmup carries no credentials; an auth rejection will not clear on retry.
		if isRenderAuthError(err) {
			logger.V(logutil.DEFAULT).Info(
				"token-producer backend requires authentication, skipping warmup; "+
					"the first request pays the cold-start cost",
				"err", err)
			return
		}
		select {
		case <-time.After(warmupRetryInterval):
		case <-ctx.Done():
			return
		}
	}
	logger.V(logutil.DEBUG).Info("token-producer backend warmup did not complete")
}

// warmupChat constructs a probe, not an inference request.
func warmupChat(model string, imageURLs ...string) *fwkrh.InferenceRequestBody {
	parts := make([]any, 0, 1+len(imageURLs))
	parts = append(parts, map[string]any{"type": "text", "text": "warmup"})
	for _, url := range imageURLs {
		parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
	}
	return &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{},
		Payload:         fwkrh.PayloadMap{"model": model, "messages": []any{map[string]any{"role": "user", "content": parts}}},
	}
}

// renderBackend produces real token IDs and owns protocol dispatch, including
// the pre-tokenized (Generate) passthrough.
type renderBackend struct {
	tk        tokenizer
	modelName string
}

func (b renderBackend) produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedRequest, error) {
	switch {
	case body.Completions != nil:
		return b.renderCompletions(ctx, body)
	case body.ChatCompletions != nil:
		tokenIDs, mmFeatures, err := b.tk.RenderChat(ctx, body.WirePayload())
		if err != nil {
			return nil, fmt.Errorf("tokenization failed: %w", err)
		}
		return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{
			TokenIDs:           tokenIDs,
			MultiModalFeatures: convertMMFeaturesToUpstream(mmFeatures),
		}}}, nil
	case body.Messages != nil:
		tokenIDs, mmFeatures, err := b.tk.RenderMessages(ctx, body.WirePayload())
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

// renderCompletions delegates every prompt shape, including token IDs, to vLLM.
func (b renderBackend) renderCompletions(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedRequest, error) {
	payload := body.WirePayload()
	// Native gRPC text has no HTTP envelope. Keep its compatibility path
	// separate from native HTTP rendering; tokenized gRPC bypasses Produce.
	if _, ok := payload.(fwkrh.PayloadProto); ok {
		payload = fwkrh.PayloadMap{"model": b.modelName, "prompt": body.Completions.Prompt.PlainText()}
	}
	allTokenIDs, _, err := b.tk.Render(ctx, payload)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}
	return fwkrh.NewTokenizedRequest(allTokenIDs), nil
}
