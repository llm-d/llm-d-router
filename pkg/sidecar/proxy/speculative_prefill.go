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

package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

const (
	// specPrefillMaxCapture bounds how many response bytes are buffered while
	// extracting the assistant answer, so a large streamed response cannot grow
	// the tee buffer without limit.
	specPrefillMaxCapture = 1 << 20 // 1 MiB

	// specPrefillTimeout bounds the background warm-up request so a slow or
	// hung decoder cannot leak goroutines.
	specPrefillTimeout = 30 * time.Second

	// specPrefillMaxConcurrency keeps speculative prefill opportunistic: one
	// warmup may run at a time, and additional warmups are skipped immediately.
	specPrefillMaxConcurrency = 1
)

var specPrefillAdmission = make(chan struct{}, specPrefillMaxConcurrency)

// speculativePrefillMiddleware wraps the chat-completions handler. When
// speculative prefill is enabled and the request opts in via the
// x-speculative-prefill header, it tees the response so that, once the turn
// finishes, the sidecar warms the KV cache with the predicted next-turn prefix
// ([prior messages + assistant answer]). The client response is never altered
// or delayed; the warm-up runs in the background.
func (s *Server) speculativePrefillMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !s.config.EnableSpeculativePrefill || !specPrefillRequested(r) {
			next(w, r)
			return
		}
		// Router-only header: don't leak it to the model server.
		r.Header.Del(routing.SpeculativePrefillHeader)
		benchmarkValues := benchmarkLogValues(r)
		release, ok := acquireSpeculativePrefillSlot()
		if !ok {
			s.logger.WithName("speculative-prefill").WithValues(benchmarkValues...).Info("skip speculative prefill: concurrency limit reached", "maxConcurrency", specPrefillMaxConcurrency)
			next(w, r)
			return
		}

		// Capture the P/D prefill target (set by EPP) before the downstream
		// handler consumes it: in disaggregated mode the warmup must hit the
		// same prefill worker whose cache the real next turn's prefill reuses.
		prefillHostPorts := r.Header.Values(routing.PrefillEndpointHeader)

		raw, body, ok := s.readJSONBody(r, w)
		if !ok {
			release()
			return
		}
		// Restore the body for the downstream handler, which reads it again.
		r = cloneRequestWithBody(r.Context(), r, raw)

		tee := newSpecPrefillWriter(w)
		next(tee, r)

		answer := accumulateAssistantText(tee.contentType(), tee.captured())
		if answer == "" {
			release()
			return
		}
		// Detach from the request context so the warm-up survives the client
		// connection closing after the turn completes.
		go s.triggerSpeculativePrefill(context.WithoutCancel(r.Context()), body, answer, prefillHostPorts, release, benchmarkValues)
	}
}

func benchmarkLogValues(r *http.Request) []any {
	return []any{
		"benchmarkCase", r.Header.Get("X-Benchmark-Case"),
		"benchmarkUser", r.Header.Get("X-Benchmark-User-ID"),
		"benchmarkTurn", r.Header.Get("X-Benchmark-Turn"),
	}
}

// specPrefillRequested reports whether the request opted into speculative
// prefill via a truthy x-speculative-prefill header.
func specPrefillRequested(r *http.Request) bool {
	enabled, err := strconv.ParseBool(strings.TrimSpace(r.Header.Get(routing.SpeculativePrefillHeader)))
	return err == nil && enabled
}

// triggerSpeculativePrefill builds the predicted next-turn prefix by appending
// the assistant answer to the original messages and issues a max_tokens=1
// request to warm the KV cache. User2 is intentionally not predicted: the prefix
// ends at the assistant answer, matching the cached portion the real next turn
// will reuse. In P/D mode (prefillHostPorts set) the warmup targets the prefill
// worker whose cache the next turn's prefill reuses; otherwise the local decoder.
func (s *Server) triggerSpeculativePrefill(ctx context.Context, originalBody map[string]any, answer string, prefillHostPorts []string, release func(), benchmarkValues []any) {
	logger := s.logger.WithName("speculative-prefill").WithValues(benchmarkValues...)
	defer release()

	messages, err := requestMessages(originalBody)
	if err != nil {
		logger.V(logging.DEBUG).Info("skip speculative prefill: cannot read messages", "error", err)
		return
	}

	assistantMsg, err := json.Marshal(map[string]any{
		requestFieldRole:    roleAssistant,
		requestFieldContent: answer,
	})
	if err != nil {
		logger.V(logging.DEBUG).Info("skip speculative prefill: cannot marshal assistant message", "error", err)
		return
	}
	// Append a trailing placeholder user turn so the chat template renders the
	// assistant answer as history, not as an in-progress generation. In
	// generation context Qwen3 injects an empty <think></think> scaffold right
	// before the assistant content, diverging from the next turn's render at the
	// first answer token so the warmed KV never hits. As history there is no
	// scaffold and the warmed prefix matches the real next turn.
	placeholderUserMsg, err := json.Marshal(map[string]any{
		requestFieldRole:    "user",
		requestFieldContent: " ",
	})
	if err != nil {
		logger.V(logging.DEBUG).Info("skip speculative prefill: cannot marshal placeholder message", "error", err)
		return
	}
	nextMessages := append(append([]json.RawMessage{}, messages...), assistantMsg, placeholderUserMsg)

	prefillBody := maps.Clone(originalBody)
	prefillBody[requestFieldMessages] = nextMessages
	prefillBody[requestFieldMaxTokens] = 1
	prefillBody[requestFieldMaxCompletionTokens] = 1
	prefillBody[requestFieldStream] = false
	// Send [prior messages + assistant answer] as a normal chat request so both
	// vLLM and SGLang render the answer as a completed history turn, matching
	// the [prior + answer] prefix the real next turn reuses. Engine-specific
	// render controls (continue_final_message/add_generation_prompt) are avoided
	// for portability; the trailing generation prompt is past the shared prefix.
	delete(prefillBody, requestFieldStreamOptions)
	delete(prefillBody, requestFieldKVTransferParams)

	// P/D: warm the prefill worker that served this turn (and that the next
	// turn's prefill will reuse).
	host := firstAllowedHostPort(s, prefillHostPorts)
	logger.Info("start speculative prefill warmup", "messages", len(nextMessages), "answerChars", len(answer), "target", host, "viaPrefiller", host != "")
	if host != "" && s.config.KVConnector == KVConnectorSGLang {
		prefillBody = s.addSGLangBootstrapInfo(prefillBody, host, s.generateSGLangRoomID())
	}

	payload, err := json.Marshal(prefillBody)
	if err != nil {
		logger.V(logging.DEBUG).Info("skip speculative prefill: cannot marshal body", "error", err)
		return
	}

	// P/D normally warms only the selected prefill worker. SGLang bootstrap
	// requires a matching decode peer, so it uses a paired P/D warmup.
	if host != "" {
		if s.config.KVConnector == KVConnectorSGLang {
			s.sendPairedPDWarmup(ctx, logger, host, payload)
			return
		}
		s.sendToPrefiller(ctx, logger, host, payload)
		return
	}
	// Fall back to the local decoder in aggregated mode or when the target fails
	// SSRF validation.
	s.sendToDecoder(ctx, logger, payload)
}

func acquireSpeculativePrefillSlot() (func(), bool) {
	select {
	case specPrefillAdmission <- struct{}{}:
		return func() { <-specPrefillAdmission }, true
	default:
		return nil, false
	}
}

// firstAllowedHostPort returns the first prefill host:port that passes SSRF
// allowlist validation, or "" when none is usable. A single header value may
// carry a comma-separated list (matching disaggregatedPrefillHandler).
func firstAllowedHostPort(s *Server, hostPorts []string) string {
	if len(hostPorts) == 1 {
		hostPorts = strings.Split(hostPorts[0], ",")
	}
	for _, hp := range hostPorts {
		hp = strings.TrimSpace(hp)
		if hp != "" && s.allowlistValidator.IsAllowed(hp) {
			return hp
		}
	}
	return ""
}

// sendToPrefiller warms the given prefill worker's KV cache by replaying the
// speculative payload through the cached prefiller reverse proxy. Best-effort:
// failures are logged at debug and never surface to the client.
func (s *Server) sendToPrefiller(ctx context.Context, logger logr.Logger, prefillHostPort string, payload []byte) {
	ctx, cancel := context.WithTimeout(ctx, specPrefillTimeout)
	defer cancel()

	handler, err := s.prefillerProxyHandler(prefillHostPort)
	if err != nil {
		logger.V(logging.DEBUG).Info("speculative prefill: prefiller handler failed", "target", prefillHostPort, "error", err)
		return
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqcommon.PathChatCompletions, bytes.NewReader(payload))
	if err != nil {
		logger.V(logging.DEBUG).Info("speculative prefill: prefiller request build failed", "error", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	bw := &bufferedResponseWriter{}
	handler.ServeHTTP(bw, req)
	logger.V(logging.DEBUG).Info("speculative prefill warmed prefiller KV cache", "target", prefillHostPort, "status", bw.statusCode)
}

// sendPairedPDWarmup warms a disaggregated deployment by issuing a matched
// prefill+decode pair with the same bootstrap room. Connectors that can warm a
// prefiller without a decode peer should use sendToPrefiller instead.
func (s *Server) sendPairedPDWarmup(ctx context.Context, logger logr.Logger, prefillHostPort string, payload []byte) {
	started := time.Now()
	ctx, cancel := context.WithTimeout(ctx, specPrefillTimeout)
	defer cancel()

	prefillHandler, err := s.prefillerProxyHandler(prefillHostPort)
	if err != nil {
		logger.V(logging.DEBUG).Info("speculative prefill: prefiller handler failed", "target", prefillHostPort, "error", err)
		return
	}

	prefillReq, err := http.NewRequestWithContext(context.WithoutCancel(ctx), http.MethodPost, reqcommon.PathChatCompletions, bytes.NewReader(payload))
	if err != nil {
		logger.V(logging.DEBUG).Info("speculative prefill: prefiller request build failed", "error", err)
		return
	}
	prefillReq.Header.Set("Content-Type", "application/json")

	decodeReq, err := http.NewRequestWithContext(ctx, http.MethodPost, reqcommon.PathChatCompletions, bytes.NewReader(payload))
	if err != nil {
		logger.V(logging.DEBUG).Info("speculative prefill: decoder request build failed", "error", err)
		return
	}
	decodeReq.Header.Set("Content-Type", "application/json")

	prefillWriter := &bufferedResponseWriter{}
	decodeWriter := &bufferedResponseWriter{}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		prefillHandler.ServeHTTP(prefillWriter, prefillReq)
	}()
	s.decoderProxy.ServeHTTP(decodeWriter, decodeReq)
	wg.Wait()

	if prefillWriter.statusCode >= http.StatusBadRequest {
		logger.V(logging.DEBUG).Info("speculative prefill prefiller returned error", "target", prefillHostPort, "status", prefillWriter.statusCode)
	}
	if decodeWriter.statusCode >= http.StatusBadRequest {
		logger.V(logging.DEBUG).Info("speculative prefill decoder returned error", "status", decodeWriter.statusCode)
	}
	logger.Info("speculative prefill warmed P/D KV cache", "target", prefillHostPort, "prefillStatus", prefillWriter.statusCode, "decodeStatus", decodeWriter.statusCode, "duration", time.Since(started).String(), "payloadBytes", len(payload))
}

// sendToDecoder posts the warm-up request to the local decoder and drains the
// response. It is best-effort: failures are logged at debug and never surface
// to the client.
func (s *Server) sendToDecoder(ctx context.Context, logger logr.Logger, payload []byte) {
	ctx, cancel := context.WithTimeout(ctx, specPrefillTimeout)
	defer cancel()

	target := s.config.DecoderURL.String() + reqcommon.PathChatCompletions
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(payload))
	if err != nil {
		logger.V(logging.DEBUG).Info("speculative prefill request build failed", "error", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{
		Transport: s.newProxyTransport(s.config.DecoderURL.Scheme, s.config.InsecureSkipVerifyForDecoder),
		Timeout:   specPrefillTimeout,
	}
	resp, err := client.Do(req)
	if err != nil {
		logger.V(logging.DEBUG).Info("speculative prefill request failed", "error", err)
		return
	}
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.Copy(io.Discard, resp.Body)
	logger.V(logging.DEBUG).Info("speculative prefill warmed decoder KV cache", "status", resp.StatusCode)
}

// accumulateAssistantText extracts the assistant's generated text from a
// captured chat-completions response, handling both non-streaming JSON and
// streaming SSE bodies.
func accumulateAssistantText(contentType string, body []byte) string {
	if strings.Contains(contentType, "text/event-stream") {
		return accumulateSSEText(body)
	}
	var response map[string]any
	if json.Unmarshal(body, &response) != nil {
		// Fall back to SSE parsing in case the content type was absent.
		return accumulateSSEText(body)
	}
	return extractChoiceText(firstChoice(response))
}

// accumulateSSEText reassembles choices[0].delta.content across SSE events.
func accumulateSSEText(body []byte) string {
	var out strings.Builder
	for _, line := range strings.Split(string(body), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, sseDataPrefix) {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, sseDataPrefix))
		if data == "" || data == "[DONE]" {
			continue
		}
		var event map[string]any
		if json.Unmarshal([]byte(data), &event) != nil {
			continue
		}
		choice := firstChoice(event)
		if choice == nil {
			continue
		}
		if delta, ok := choice[responseFieldDelta].(map[string]any); ok {
			if content, ok := delta[requestFieldContent].(string); ok {
				out.WriteString(content)
			}
		}
	}
	return out.String()
}

// specPrefillWriter is a pass-through ResponseWriter that also captures up to
// specPrefillMaxCapture bytes of the response body so the assistant answer can
// be extracted after the turn completes. It preserves streaming by forwarding
// Flush to the underlying writer.
type specPrefillWriter struct {
	http.ResponseWriter
	buf       bytes.Buffer
	truncated bool
}

func newSpecPrefillWriter(w http.ResponseWriter) *specPrefillWriter {
	return &specPrefillWriter{ResponseWriter: w}
}

func (w *specPrefillWriter) Write(b []byte) (int, error) {
	if remaining := specPrefillMaxCapture - w.buf.Len(); remaining > 0 {
		if len(b) <= remaining {
			w.buf.Write(b)
		} else {
			w.buf.Write(b[:remaining])
			w.truncated = true
		}
	}
	return w.ResponseWriter.Write(b)
}

func (w *specPrefillWriter) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *specPrefillWriter) captured() []byte { return w.buf.Bytes() }

func (w *specPrefillWriter) contentType() string {
	return w.ResponseWriter.Header().Get("Content-Type")
}
