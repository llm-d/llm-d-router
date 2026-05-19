/*
Copyright 2025 The llm-d Authors.

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
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/felixge/httpsnoop"
)

type cachedTokensUsageRewriter struct {
	header       http.Header
	cachedTokens int
	wroteHeader  bool
	streaming    bool
	streamBuffer []byte
}

// OpenAI-compatible chat usage reports prompt cache hits at
// usage.prompt_tokens_details.cached_tokens.
// See: https://platform.openai.com/docs/guides/prompt-caching
const promptTokensDetailsField = "prompt_tokens_details"

func newCachedTokensResponseWriter(w http.ResponseWriter, cachedTokens int) http.ResponseWriter {
	writer, _ := newCachedTokensResponseWriterWithFinalize(w, cachedTokens)
	return writer
}

func newCachedTokensResponseWriterWithFinalize(w http.ResponseWriter, cachedTokens int) (http.ResponseWriter, func() error) {
	rewriter := &cachedTokensUsageRewriter{
		header:       w.Header(),
		cachedTokens: cachedTokens,
	}
	// httpsnoop preserves optional ResponseWriter interfaces such as
	// http.Flusher, http.Hijacker, http.Pusher, and io.ReaderFrom.
	writer := httpsnoop.Wrap(w, httpsnoop.Hooks{
		WriteHeader: func(next httpsnoop.WriteHeaderFunc) httpsnoop.WriteHeaderFunc {
			return func(statusCode int) {
				rewriter.writeHeader(next, statusCode)
			}
		},
		Write: func(next httpsnoop.WriteFunc) httpsnoop.WriteFunc {
			return func(body []byte) (int, error) {
				return rewriter.write(next, body)
			}
		},
		ReadFrom: func(_ httpsnoop.ReadFromFunc) httpsnoop.ReadFromFunc {
			return func(src io.Reader) (int64, error) {
				// ReadFrom would otherwise bypass Write and skip usage rewriting.
				return rewriter.readFrom(w.Write, src)
			}
		},
	})
	return writer, func() error {
		return rewriter.flushSSEBuffer(w.Write)
	}
}

func (r *cachedTokensUsageRewriter) writeHeader(next httpsnoop.WriteHeaderFunc, statusCode int) {
	// Rewriting may change the body size, so any upstream Content-Length is stale.
	r.header.Del("Content-Length")
	r.wroteHeader = true
	next(statusCode)
}

func (r *cachedTokensUsageRewriter) write(next httpsnoop.WriteFunc, body []byte) (int, error) {
	updated := r.rewrite(body)
	if !r.wroteHeader {
		r.header.Del("Content-Length")
	}
	n, err := next(updated)
	if err != nil {
		return n, err
	}
	return len(body), nil
}

func (r *cachedTokensUsageRewriter) readFrom(next httpsnoop.WriteFunc, src io.Reader) (int64, error) {
	if r.isSSE(nil) {
		// SSE can be long-lived, so keep it streaming and rewrite complete lines.
		n, err := io.Copy(cachedTokensStreamWriter{
			write:   r.write,
			forward: next,
		}, src)
		if err != nil {
			return n, err
		}
		if err := r.flushSSEBuffer(next); err != nil {
			return n, err
		}
		return n, nil
	}

	// Non-streaming JSON must be complete before we can safely rewrite usage.
	body, err := io.ReadAll(src)
	if err != nil {
		return 0, err
	}
	_, err = r.write(next, body)
	if err != nil {
		return 0, err
	}
	return int64(len(body)), nil
}

func (r *cachedTokensUsageRewriter) rewrite(body []byte) []byte {
	if r.isSSE(body) {
		return r.rewriteSSEChunk(body)
	}
	return replaceCachedTokens(body, r.cachedTokens)
}

type cachedTokensStreamWriter struct {
	write   func(httpsnoop.WriteFunc, []byte) (int, error)
	forward httpsnoop.WriteFunc
}

func (w cachedTokensStreamWriter) Write(body []byte) (int, error) {
	return w.write(w.forward, body)
}

func (r *cachedTokensUsageRewriter) isSSE(body []byte) bool {
	if r.streaming {
		return true
	}
	contentType := r.header.Get("Content-Type")
	// Some handlers may not set the content type before the first Write.
	if strings.Contains(contentType, "text/event-stream") || bytes.HasPrefix(body, []byte("data:")) {
		r.streaming = true
		return true
	}
	return false
}

func (r *cachedTokensUsageRewriter) rewriteSSEChunk(body []byte) []byte {
	r.streamBuffer = append(r.streamBuffer, body...)
	updated := make([]byte, 0, len(r.streamBuffer))
	for {
		lineEnd := bytes.IndexByte(r.streamBuffer, '\n')
		if lineEnd < 0 {
			break
		}
		// Upstream may split chunks mid-event; only rewrite complete SSE lines.
		line := r.streamBuffer[:lineEnd+1]
		replacedLine, _ := replaceCachedTokensSSELine(line, r.cachedTokens)
		updated = append(updated, replacedLine...)
		r.streamBuffer = r.streamBuffer[lineEnd+1:]
	}
	return updated
}

func (r *cachedTokensUsageRewriter) flushSSEBuffer(next httpsnoop.WriteFunc) error {
	if len(r.streamBuffer) == 0 {
		return nil
	}
	// EOF without a final newline is malformed SSE, but forward it rather than
	// dropping bytes. If it is a data line, still try to apply the usage rewrite.
	replacedLine, _ := replaceCachedTokensSSELine(r.streamBuffer, r.cachedTokens)
	r.streamBuffer = nil
	_, err := next(replacedLine)
	return err
}

func extractCachedTokens(response map[string]any) (int, bool) {
	usage, ok := response["usage"].(map[string]any)
	if !ok {
		return 0, false
	}
	return cachedTokensFromUsage(usage)
}

func cachedTokensFromUsage(usage map[string]any) (int, bool) {
	// Only the documented OpenAI-compatible field is used as the source of truth.
	details, ok := usage[promptTokensDetailsField].(map[string]any)
	if !ok {
		return 0, false
	}
	if cachedTokens, ok := intValue(details["cached_tokens"]); ok {
		return cachedTokens, true
	}
	return 0, false
}

func intValue(value any) (int, bool) {
	switch v := value.(type) {
	case float64:
		return int(v), true
	case int:
		return v, true
	default:
		return 0, false
	}
}

func replaceCachedTokens(body []byte, cachedTokens int) []byte {
	if len(bytes.TrimSpace(body)) == 0 {
		return body
	}
	// Prefer full JSON first; streamed responses are handled line-by-line below.
	if updated, ok := replaceCachedTokensJSON(body, cachedTokens); ok {
		return updated
	}
	if updated, ok := replaceCachedTokensSSE(body, cachedTokens); ok {
		return updated
	}
	return body
}

func replaceCachedTokensJSON(body []byte, cachedTokens int) ([]byte, bool) {
	var response map[string]any
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, false
	}
	if !setCachedTokens(response, cachedTokens) {
		return body, true
	}
	updated, err := json.Marshal(response)
	if err != nil {
		return body, true
	}
	return updated, true
}

func replaceCachedTokensSSE(body []byte, cachedTokens int) ([]byte, bool) {
	lines := bytes.SplitAfter(body, []byte("\n"))
	updated := make([]byte, 0, len(body))
	changed := false
	processed := false

	for _, line := range lines {
		if len(line) == 0 {
			continue
		}
		replacedLine, ok := replaceCachedTokensSSELine(line, cachedTokens)
		if !ok {
			updated = append(updated, replacedLine...)
			continue
		}
		processed = true
		if !bytes.Equal(replacedLine, line) {
			changed = true
		}
		updated = append(updated, replacedLine...)
	}

	if !processed {
		return nil, false
	}
	if !changed {
		return body, true
	}
	return updated, true
}

func replaceCachedTokensSSELine(line []byte, cachedTokens int) ([]byte, bool) {
	trimmedLine := bytes.TrimRight(line, "\r\n")
	lineEnding := line[len(trimmedLine):]
	data, ok := bytes.CutPrefix(trimmedLine, []byte("data: "))
	if !ok {
		return line, false
	}
	if bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
		return line, true
	}
	// Only JSON data frames can carry usage; other SSE frames pass through.
	replacedData, isJSON := replaceCachedTokensJSON(data, cachedTokens)
	if !isJSON {
		return line, true
	}
	updated := make([]byte, 0, len(line))
	updated = append(updated, []byte("data: ")...)
	updated = append(updated, replacedData...)
	updated = append(updated, lineEnding...)
	return updated, true
}

func setCachedTokens(response map[string]any, cachedTokens int) bool {
	usage, ok := response["usage"].(map[string]any)
	if !ok {
		return false
	}
	changed := false
	details, ok := usage[promptTokensDetailsField].(map[string]any)
	if !ok {
		// Some decoder chunks omit details entirely; create the standard field.
		usage[promptTokensDetailsField] = map[string]any{"cached_tokens": cachedTokens}
		return true
	}
	if current, ok := intValue(details["cached_tokens"]); !ok || current != cachedTokens {
		details["cached_tokens"] = cachedTokens
		changed = true
	}
	return changed
}
