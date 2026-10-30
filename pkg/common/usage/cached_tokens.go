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

// Package usage reads and rewrites the OpenAI-compatible token usage block of
// an inference response.
package usage

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/felixge/httpsnoop"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

type cachedTokensUsageRewriter struct {
	header        http.Header
	cachedTokens  *int
	onUsage       func(block map[string]any)
	dropUsageOnly bool
	wroteHeader   bool
	streaming     bool
	streamBuffer  []byte
	jsonBuffer    []byte
}

// Options selects what the response writer does with the usage blocks it sees.
type Options struct {
	// CachedTokens, when set, is written into every usage block as its
	// cached-token count (see NewCachedTokensResponseWriter).
	CachedTokens *int
	// OnUsage is called with each usage block the response carries, decoded,
	// before any rewrite. Streamed responses can carry more than one.
	OnUsage func(block map[string]any)
	// DropUsageOnlyFrame drops a streamed chat or completions chunk that
	// carries usage and no choices, for a stream whose include_usage the
	// coordinator added itself so the client never asked for that frame.
	DropUsageOnlyFrame bool
}

// usageKey is the JSON key that must be present before a frame can carry usage.
// Streamed responses send one frame per token and only the final frame has usage,
// so scanning for this is much cheaper than unmarshalling every frame to find out.
// The openai and anthropic stream parsers scan for the same word without the
// quotes; a JSON serializer always writes the key quoted, so keeping them here
// skips more content frames.
var usageKey = []byte(`"` + reqcommon.FieldUsage + `"`)

// NewCachedTokensResponseWriter wraps w so that every usage block it carries
// reports cachedTokens as its cached-token count, in the details object the
// block's own schema documents (see tokenDetails). streaming selects SSE
// line-by-line rewriting over whole-document rewriting; it comes from the
// request, since the upstream content type is not yet known.
//
// The returned function must be called once the response is complete: both
// modes buffer an incomplete tail, which it flushes.
func NewCachedTokensResponseWriter(
	w http.ResponseWriter, cachedTokens int, streaming bool,
) (http.ResponseWriter, func() error) {
	return NewResponseWriter(w, Options{CachedTokens: &cachedTokens}, streaming)
}

// NewResponseWriter wraps w according to opts. See NewCachedTokensResponseWriter
// for the streaming flag and the returned flush function.
func NewResponseWriter(
	w http.ResponseWriter, opts Options, streaming bool,
) (http.ResponseWriter, func() error) {
	rewriter := &cachedTokensUsageRewriter{
		header:        w.Header(),
		cachedTokens:  opts.CachedTokens,
		onUsage:       opts.OnUsage,
		dropUsageOnly: opts.DropUsageOnlyFrame,
		streaming:     streaming,
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
		if rewriter.streaming {
			return rewriter.flushSSEBuffer(w.Write)
		}
		return rewriter.flushJSONBuffer(w.Write)
	}
}

func (r *cachedTokensUsageRewriter) writeHeader(next httpsnoop.WriteHeaderFunc, statusCode int) {
	// Rewriting may change the body size, so any upstream Content-Length is stale.
	r.header.Del("Content-Length")
	r.wroteHeader = true
	next(statusCode)
}

func (r *cachedTokensUsageRewriter) write(next httpsnoop.WriteFunc, body []byte) (int, error) {
	if !r.streaming {
		// ReverseProxy may split one JSON document across many Write calls; it is
		// parsed once, by flushJSONBuffer.
		r.jsonBuffer = append(r.jsonBuffer, body...)
		return len(body), nil
	}
	updated := r.rewriteSSEChunk(body)
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
	if r.streaming {
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
	r.jsonBuffer = append(r.jsonBuffer, body...)
	if err := r.flushJSONBuffer(next); err != nil {
		return 0, err
	}
	return int64(len(body)), nil
}

type cachedTokensStreamWriter struct {
	write   func(httpsnoop.WriteFunc, []byte) (int, error)
	forward httpsnoop.WriteFunc
}

func (w cachedTokensStreamWriter) Write(body []byte) (int, error) {
	return w.write(w.forward, body)
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
		replacedLine, _, drop := r.processSSELine(line)
		if !drop {
			updated = append(updated, replacedLine...)
		}
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
	replacedLine, _, drop := r.processSSELine(r.streamBuffer)
	r.streamBuffer = nil
	if drop {
		return nil
	}
	_, err := next(replacedLine)
	return err
}

func (r *cachedTokensUsageRewriter) flushJSONBuffer(next httpsnoop.WriteFunc) error {
	if len(r.jsonBuffer) == 0 {
		return nil
	}
	updated := r.process(r.jsonBuffer)
	r.jsonBuffer = nil
	if !r.wroteHeader {
		r.header.Del("Content-Length")
	}
	_, err := next(updated)
	return err
}

// CachedTokens reads the cached-token count out of a response's usage block,
// which callers pass as the decoded value of the response's "usage" field. It
// reports false when the block is absent or carries no count; vLLM sends the
// details object as null when the count is zero, so callers that need a number
// treat that as zero.
func CachedTokens(block any) (int, bool) {
	fields, ok := block.(map[string]any)
	if !ok {
		return 0, false
	}
	details, ok := tokenDetails(fields)
	if !ok {
		return 0, false
	}
	return intValue(details[reqcommon.FieldCachedTokens])
}

// tokenDetails returns the object a usage block carries its cached-token count
// in. Responses and Conversations report input_tokens_details, the chat,
// completions, and generate routes report prompt_tokens_details; a block
// carrying both is read as Responses, which is also how the EPP's openai parser
// resolves it. Only the documented OpenAI-compatible fields are used as the
// source of truth.
// See: https://platform.openai.com/docs/guides/prompt-caching
func tokenDetails(block map[string]any) (map[string]any, bool) {
	for _, field := range []string{reqcommon.FieldInputTokensDetails, reqcommon.FieldPromptTokensDetails} {
		if details, ok := block[field].(map[string]any); ok {
			return details, true
		}
	}
	return nil, false
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

// The replaceCachedTokens* functions are the cached-token rewrite alone, with
// no observer and no frame dropping.
func replaceCachedTokens(body []byte, cachedTokens int) []byte {
	return (&cachedTokensUsageRewriter{cachedTokens: &cachedTokens}).process(body)
}

func replaceCachedTokensSSELine(line []byte, cachedTokens int) ([]byte, bool) {
	updated, isData, _ := (&cachedTokensUsageRewriter{cachedTokens: &cachedTokens}).processSSELine(line)
	return updated, isData
}

// process applies the rewriter to a complete body: a JSON document first,
// otherwise SSE lines.
func (r *cachedTokensUsageRewriter) process(body []byte) []byte {
	if len(bytes.TrimSpace(body)) == 0 {
		return body
	}
	if updated, ok := r.processJSON(body); ok {
		return updated
	}
	if updated, ok := r.processSSE(body); ok {
		return updated
	}
	return body
}

// processJSON handles one JSON document. Only the usage block is decoded: the
// rest of the document is passed through as raw JSON, so values outside usage
// keep the key order and number formatting the worker sent. The block is
// reported to the observer, then rewritten if a cached-token count is set.
// ok is false when body is not a JSON object.
func (r *cachedTokensUsageRewriter) processJSON(body []byte) ([]byte, bool) {
	var response map[string]json.RawMessage
	if err := json.Unmarshal(body, &response); err != nil || response == nil {
		return nil, false
	}
	block, wrapper := r.observe(response)
	if r.cachedTokens == nil || !rewritable(wrapper) || !setCachedTokens(block, *r.cachedTokens) {
		return body, true
	}
	return marshalWithUsage(response, wrapper, block, body), true
}

func (r *cachedTokensUsageRewriter) processSSE(body []byte) ([]byte, bool) {
	lines := bytes.SplitAfter(body, []byte("\n"))
	updated := make([]byte, 0, len(body))
	changed := false
	processed := false

	for _, line := range lines {
		if len(line) == 0 {
			continue
		}
		replacedLine, isData, drop := r.processSSELine(line)
		if !isData {
			updated = append(updated, replacedLine...)
			continue
		}
		processed = true
		if drop {
			changed = true
			continue
		}
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

// processSSELine handles one SSE line. isData is true for a data: line, which
// is the only kind that can carry usage. drop is true when the line is a
// usage-only chunk the coordinator asked for itself and the client should not
// see; the returned line is then meaningless.
func (r *cachedTokensUsageRewriter) processSSELine(line []byte) (out []byte, isData bool, drop bool) {
	trimmedLine := bytes.TrimRight(line, "\r\n")
	lineEnding := line[len(trimmedLine):]
	data, ok := bytes.CutPrefix(trimmedLine, []byte(reqcommon.SSEDataPrefix))
	if !ok {
		return line, false, false
	}
	if bytes.Equal(bytes.TrimSpace(data), []byte(reqcommon.SSEDoneMarker)) {
		return line, true, false
	}
	if !bytes.Contains(data, usageKey) {
		// No usage in this frame, so unmarshalling it could not change anything.
		return line, true, false
	}
	// Only JSON data frames can carry usage; other SSE frames pass through.
	var response map[string]json.RawMessage
	if err := json.Unmarshal(data, &response); err != nil || response == nil {
		return line, true, false
	}
	block, wrapper := r.observe(response)
	if r.dropUsageOnly && block != nil && wrapper == "" && hasEmptyChoices(response) {
		return nil, true, true
	}
	if r.cachedTokens == nil || !rewritable(wrapper) || !setCachedTokens(block, *r.cachedTokens) {
		return line, true, false
	}
	updated := make([]byte, 0, len(line))
	updated = append(updated, []byte(reqcommon.SSEDataPrefix)...)
	updated = append(updated, marshalWithUsage(response, wrapper, block, data)...)
	updated = append(updated, lineEnding...)
	return updated, true, false
}

// observe decodes the usage block a response carries and reports it to the
// observer. It is the top-level "usage" for chat, completions, generate and
// the final Responses and Messages objects; "response.usage" for Responses
// stream events; "message.usage" for the Messages message_start event. wrapper
// names the object that holds a nested block and is empty for a top-level one.
// A null usage, which Responses sends on its early events, counts as absent.
func (r *cachedTokensUsageRewriter) observe(response map[string]json.RawMessage) (block map[string]any, wrapper string) {
	if raw, ok := response[reqcommon.FieldUsage]; ok {
		if err := json.Unmarshal(raw, &block); err != nil {
			block = nil
		}
	}
	if block == nil {
		for _, key := range []string{fieldResponse, fieldMessage} {
			raw, ok := response[key]
			if !ok {
				continue
			}
			var inner map[string]json.RawMessage
			if err := json.Unmarshal(raw, &inner); err != nil {
				continue
			}
			var nested map[string]any
			if err := json.Unmarshal(inner[reqcommon.FieldUsage], &nested); err == nil && nested != nil {
				block, wrapper = nested, key
				break
			}
		}
	}
	if block != nil && r.onUsage != nil {
		r.onUsage(block)
	}
	return block, wrapper
}

// rewritable reports whether the cached-token rewrite applies to a usage block
// found under wrapper: the top-level block, and response.usage on Responses
// stream events such as response.completed.
func rewritable(wrapper string) bool {
	return wrapper == "" || wrapper == fieldResponse
}

// marshalWithUsage re-encodes response with block as its usage, or as the
// usage of response[wrapper] when wrapper is set, leaving every other value as
// the raw JSON it arrived as. On an encoding error the original bytes are
// returned unchanged.
func marshalWithUsage(response map[string]json.RawMessage, wrapper string, block map[string]any, original []byte) []byte {
	usage, err := json.Marshal(block)
	if err != nil {
		return original
	}
	if wrapper != "" {
		var inner map[string]json.RawMessage
		if err := json.Unmarshal(response[wrapper], &inner); err != nil {
			return original
		}
		inner[reqcommon.FieldUsage] = usage
		if usage, err = json.Marshal(inner); err != nil {
			return original
		}
		response[wrapper] = usage
	} else {
		response[reqcommon.FieldUsage] = usage
	}
	updated, err := json.Marshal(response)
	if err != nil {
		return original
	}
	return updated
}

// hasEmptyChoices reports whether a streamed chat or completions chunk is the
// trailing usage frame vLLM sends for stream_options.include_usage: an empty
// choices list beside the usage block.
func hasEmptyChoices(response map[string]json.RawMessage) bool {
	raw, ok := response[fieldChoices]
	if !ok {
		return false
	}
	var choices []json.RawMessage
	return json.Unmarshal(raw, &choices) == nil && len(choices) == 0
}

func setCachedTokens(block map[string]any, cachedTokens int) bool {
	if block == nil {
		return false
	}
	details, ok := tokenDetails(block)
	if !ok {
		// Some decoder chunks report prompt_tokens with the details object
		// omitted or null; create the field those routes document. A block
		// carrying neither details object nor prompt_tokens follows another
		// schema (the Messages API reports cache_read_input_tokens) and is left
		// alone rather than given a field it does not define.
		if _, documented := block[reqcommon.FieldPromptTokens]; !documented {
			return false
		}
		block[reqcommon.FieldPromptTokensDetails] = map[string]any{reqcommon.FieldCachedTokens: cachedTokens}
		return true
	}
	if current, ok := intValue(details[reqcommon.FieldCachedTokens]); !ok || current != cachedTokens {
		details[reqcommon.FieldCachedTokens] = cachedTokens
		return true
	}
	return false
}
