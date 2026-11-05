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

package pipeline

import (
	"context"
	"net/http"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
)

// SGLangPrefillStream coordinates a concurrent SGLang gRPC prefill stream with
// the decode step. PrefillStep creates it once response headers arrive; a
// background reader records cached_tokens and any stream or trailer failure.
type SGLangPrefillStream struct {
	cancel     context.CancelFunc
	stopAfter  func() bool
	done       chan struct{}
	closeOnce  sync.Once
	finishOnce sync.Once

	mu             sync.Mutex
	err            error
	grpcStatus     string
	grpcMessage    string
	grpcDetailsBin string
	// cachedTokens0 and promptTokens0 hold index 0, the only index unless the
	// request sets n > 1. The maps are allocated only for other indexes.
	cachedTokens0    uint32
	hasCachedTokens0 bool
	cachedTokens     map[uint32]uint32
	promptTokens0    uint32
	hasPromptTokens0 bool
	promptTokens     map[uint32]uint32
	onFail           func(error)
}

// NewSGLangPrefillStream constructs a stream tracker bound to cancel and
// stopAfter (from context.AfterFunc).
func NewSGLangPrefillStream(cancel context.CancelFunc, stopAfter func() bool) *SGLangPrefillStream {
	return &SGLangPrefillStream{
		cancel:    cancel,
		stopAfter: stopAfter,
		done:      make(chan struct{}),
	}
}

// RecordCachedTokens stores the prompt prefix-cache hit count reported by a
// prefill GenerateResponse frame for item index.
func (s *SGLangPrefillStream) RecordCachedTokens(index, tokens uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index == 0 {
		s.cachedTokens0, s.hasCachedTokens0 = tokens, true
		return
	}
	if s.cachedTokens == nil {
		s.cachedTokens = make(map[uint32]uint32)
	}
	s.cachedTokens[index] = tokens
}

// CachedTokens returns the cached token count reported by prefill for item
// index, if any.
func (s *SGLangPrefillStream) CachedTokens(index uint32) (uint32, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index == 0 {
		return s.cachedTokens0, s.hasCachedTokens0
	}
	tokens, ok := s.cachedTokens[index]
	return tokens, ok
}

// RecordPromptTokens stores the prompt token count reported by a prefill
// GenerateResponse frame for item index.
func (s *SGLangPrefillStream) RecordPromptTokens(index, tokens uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index == 0 {
		s.promptTokens0, s.hasPromptTokens0 = tokens, true
		return
	}
	if s.promptTokens == nil {
		s.promptTokens = make(map[uint32]uint32)
	}
	s.promptTokens[index] = tokens
}

// PromptTokens returns the prompt token count reported by prefill for item
// index, if any.
func (s *SGLangPrefillStream) PromptTokens(index uint32) (uint32, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index == 0 {
		return s.promptTokens0, s.hasPromptTokens0
	}
	tokens, ok := s.promptTokens[index]
	return tokens, ok
}

// Fail records a prefill failure and notifies any registered failure callback.
func (s *SGLangPrefillStream) Fail(err error, grpcStatus, grpcMessage, grpcDetailsBin string) {
	s.mu.Lock()
	if s.err != nil {
		s.mu.Unlock()
		return
	}
	s.err = err
	s.grpcStatus = grpcStatus
	s.grpcMessage = grpcMessage
	s.grpcDetailsBin = grpcDetailsBin
	cb := s.onFail
	s.mu.Unlock()
	if cb != nil {
		cb(err)
	}
}

// OnFailure registers fn to run if prefill fails. If prefill has already failed,
// fn is called immediately before OnFailure returns.
func (s *SGLangPrefillStream) OnFailure(fn func(error)) {
	s.mu.Lock()
	err := s.err
	if err == nil {
		s.onFail = fn
	}
	s.mu.Unlock()
	if err != nil && fn != nil {
		fn(err)
	}
}

// Failure returns any recorded prefill gRPC status code, message, status
// details binary trailer, and error.
func (s *SGLangPrefillStream) Failure() (grpcStatus, grpcMessage, grpcDetailsBin string, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.grpcStatus, s.grpcMessage, s.grpcDetailsBin, s.err
}

// MarkDone signals that the background prefill reader has finished.
func (s *SGLangPrefillStream) MarkDone() {
	s.finishOnce.Do(func() {
		if s.stopAfter != nil {
			s.stopAfter()
		}
		close(s.done)
	})
}

// Done returns a channel closed when the prefill reader finishes.
func (s *SGLangPrefillStream) Done() <-chan struct{} {
	return s.done
}

// Wait blocks until the prefill reader finishes, ctx ends, or timeout elapses.
func (s *SGLangPrefillStream) Wait(ctx context.Context, timeout time.Duration) bool {
	select {
	case <-s.done:
		return true
	default:
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-s.done:
		return true
	case <-ctx.Done():
		return false
	case <-timer.C:
		return false
	}
}

// DetachRequestContext stops forwarding parent request-context cancellation to
// the prefill leg once the client response has completed normally.
func (s *SGLangPrefillStream) DetachRequestContext() {
	if s.stopAfter != nil {
		s.stopAfter()
	}
}

// Close cancels the prefill context and detaches its parent cancellation hook.
func (s *SGLangPrefillStream) Close() {
	s.closeOnce.Do(func() {
		if s.stopAfter != nil {
			s.stopAfter()
		}
		if s.cancel != nil {
			s.cancel()
		}
	})
}

// HTTPStatusToGRPCCode maps an upstream HTTP status code to a gRPC status code.
func HTTPStatusToGRPCCode(status int) codes.Code {
	switch status {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return codes.InvalidArgument
	case http.StatusUnauthorized:
		return codes.Unauthenticated
	case http.StatusForbidden:
		return codes.PermissionDenied
	case http.StatusNotFound, http.StatusMethodNotAllowed:
		return codes.Unimplemented
	case http.StatusRequestEntityTooLarge, http.StatusTooManyRequests:
		return codes.ResourceExhausted
	case http.StatusGatewayTimeout:
		return codes.DeadlineExceeded
	default:
		return codes.Unavailable
	}
}
