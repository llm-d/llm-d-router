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

package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"strings"
	"sync"
	"time"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

const (
	AnthropicParserType = "anthropic-parser"

	messagesAPI    = "messages"
	countTokensAPI = "messages/count_tokens"

	eventMessageStart = "message_start"
	eventMessageDelta = "message_delta"

	// maxTrackedStreams bounds the stream map when streams end without a
	// final frame faster than the staleness sweep releases them.
	maxTrackedStreams = 4096
	// compactMinPeak is the peak map size below which the map is never
	// reallocated to release empty buckets.
	compactMinPeak = 64
)

// compile-time type validation
var (
	_ fwkrh.Parser            = &AnthropicParser{}
	_ fwkrh.ModelNameRewriter = &AnthropicParser{}
)

type AnthropicParser struct {
	typedName fwkplugin.TypedName

	// now is the stream staleness clock; nil uses time.Now.
	now func() time.Time

	mu          sync.Mutex
	streams     map[context.Context]*streamState
	peakStreams int
	lastSweep   time.Time
}

func NewAnthropicParser() *AnthropicParser {
	return &AnthropicParser{
		typedName: fwkplugin.TypedName{
			Type: AnthropicParserType,
			Name: AnthropicParserType,
		},
	}
}

func (p *AnthropicParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *AnthropicParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths:     []string{messagesAPI, countTokensAPI, messagesAPI + "/render"},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}
}

func AnthropicParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewAnthropicParser().WithName(name), nil
}

func (p *AnthropicParser) WithName(name string) *AnthropicParser {
	p.typedName.Name = name
	return p
}

func (p *AnthropicParser) ParseRequest(_ context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	path := request.GetRequestPath(headers)
	if request.MatchPathSuffix(path, messagesAPI+"/render") {
		return parserutil.ParseRenderRequest(body)
	}

	countTokens := request.MatchPathSuffix(path, countTokensAPI)
	if !countTokens && !request.MatchPathSuffix(path, messagesAPI) {
		return nil, fmt.Errorf("unsupported API endpoint: %s", path)
	}

	bodyMap, err := parserutil.UnmarshalEnvelope(body, "system")
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling request body: %w", err)
	}

	result := &request.InferenceRequestBody{
		Payload:         request.PayloadMap(bodyMap),
		RawBody:         body,
		MaxOutputTokens: request.MaxOutputTokensFromPayload(bodyMap, "max_tokens"),
	}
	if model, ok := bodyMap["model"].(string); ok {
		result.Model = model
	}

	// count_tokens delegates token counting to the server and passes its response
	// through, so only the envelope is read: Messages stays nil, while the model
	// still resolves and rewrites.
	if countTokens {
		return &fwkrh.ParseResult{Body: result, JSONValidated: true, SkipResponseProcessing: true}, nil
	}

	var messagesReq request.MessagesRequest
	if err := json.Unmarshal(body, &messagesReq); err != nil {
		var typeErr *json.UnmarshalTypeError
		if errors.As(err, &typeErr) && typeErr.Field != "" {
			err = &fwkrh.RequestFieldError{Field: typeErr.Field, Err: err}
		}
		return nil, fmt.Errorf("error parsing messages request: %w", err)
	}
	if len(messagesReq.Messages) == 0 {
		return nil, errors.New("invalid messages request: must have at least one message")
	}
	result.Messages = &messagesReq
	if stream, ok := bodyMap["stream"].(bool); ok && stream {
		result.Stream = true
	}

	return &fwkrh.ParseResult{Body: result, JSONValidated: true, SkipResponseProcessing: false}, nil
}

// RewriteModelName writes the resolved model into the request payload map.
func (p *AnthropicParser) RewriteModelName(payload request.MarshalablePayload, model string) (request.MarshalablePayload, error) {
	m, ok := payload.(request.PayloadMap)
	if !ok {
		return payload, nil
	}
	m["model"] = model
	return m, nil
}

// ParseResponse extracts usage from a response body. For an event stream, ctx
// identifies the response across chunks: the input token buckets seen before
// end of stream are kept per ctx and released at end of stream, when ctx is
// done, or when the stream goes stale.
func (p *AnthropicParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endOfStream bool) (*fwkrh.ParsedResponse, error) {
	if !isEventStream(headers) {
		if endOfStream {
			p.updateStream(ctx, chunkSummary{}, true)
		}
		if len(body) == 0 {
			return nil, nil //nolint:nilnil
		}
		usage, err := extractUsage(body)
		if err != nil {
			return nil, err
		}
		return &fwkrh.ParsedResponse{Usage: usage}, nil
	}

	s := scanChunk(body)
	if !endOfStream && !s.sawStart && !s.sawDelta {
		if len(body) == 0 {
			return nil, nil //nolint:nilnil
		}
		return &fwkrh.ParsedResponse{StreamedEvents: s.events}, nil
	}
	// The final frame is often an empty body, so the stream entry is released
	// before the empty check.
	tok := p.updateStream(ctx, s, endOfStream)
	if len(body) == 0 {
		return nil, nil //nolint:nilnil
	}
	return &fwkrh.ParsedResponse{Usage: s.usage(tok), StreamedEvents: s.events}, nil
}

func isEventStream(headers map[string]string) bool {
	for k, v := range headers {
		if !strings.EqualFold(k, reqcommon.HeaderContentType) {
			continue
		}
		mediaType, _, _ := strings.Cut(v, ";")
		if strings.EqualFold(strings.TrimSpace(mediaType), reqcommon.MediaTypeEventStream) {
			return true
		}
	}
	return false
}

// streamState carries the input token buckets of one streamed response across
// its body chunks. message_start reports the cache buckets and message_delta
// may omit them, and the two events often arrive in separate chunks.
type streamState struct {
	tok       inputTokens
	updatedAt time.Time
	// stop deregisters the ctx cancellation hook; nil when ctx is never done.
	stop func() bool
}

// trackable reports whether ctx can key the stream map without panicking. The
// dynamic value is checked because a comparable struct type can hold an
// interface field whose value is a map, slice, or func.
func trackable(ctx context.Context) bool {
	return ctx != nil && reflect.ValueOf(ctx).Comparable()
}

func (p *AnthropicParser) clock() time.Time {
	if p.now == nil {
		return time.Now()
	}
	return p.now()
}

// updateStream folds a chunk carrying message_start, a message_delta usage, or
// end of stream into the state kept for ctx, and returns the input token
// buckets of the stream so far.
func (p *AnthropicParser) updateStream(ctx context.Context, s chunkSummary, endOfStream bool) inputTokens {
	if !trackable(ctx) {
		return s.tok
	}
	var stops []func() bool
	defer func() {
		for _, stop := range stops {
			stop()
		}
	}()
	p.mu.Lock()
	defer p.mu.Unlock()

	entry := p.streams[ctx]
	tok := s.tok
	if entry != nil && !s.sawStart {
		tok = entry.tok
		tok.merge(s.tok)
	}
	if endOfStream || ctx.Err() != nil || tok == (inputTokens{}) {
		if entry != nil {
			stops = p.removeLocked(ctx, entry, stops)
		}
		return tok
	}

	now := p.clock()
	if entry != nil {
		entry.tok = tok
		entry.updatedAt = now
		return tok
	}
	stops = p.sweepStaleLocked(now, len(p.streams) >= maxTrackedStreams, stops)
	if len(p.streams) >= maxTrackedStreams {
		stops = p.evictOldestLocked(stops)
	}
	entry = &streamState{tok: tok, updatedAt: now}
	if ctx.Done() != nil {
		entry.stop = context.AfterFunc(ctx, func() { p.deleteStream(ctx, entry) })
	}
	if p.streams == nil {
		p.streams = make(map[context.Context]*streamState)
	}
	p.streams[ctx] = entry
	p.peakStreams = max(p.peakStreams, len(p.streams))
	return tok
}

// deleteStream removes entry only if it still owns ctx, so a late cancellation
// hook cannot drop a newer entry stored under the same ctx.
func (p *AnthropicParser) deleteStream(ctx context.Context, entry *streamState) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.streams[ctx] == entry {
		delete(p.streams, ctx)
		p.compactLocked()
	}
}

// removeLocked deletes the entry for ctx and appends its cancellation hook
// deregistration to stops, which the caller runs after releasing p.mu.
func (p *AnthropicParser) removeLocked(ctx context.Context, entry *streamState, stops []func() bool) []func() bool {
	delete(p.streams, ctx)
	p.compactLocked()
	if entry.stop != nil {
		stops = append(stops, entry.stop)
	}
	return stops
}

// sweepStaleLocked drops every entry that saw no chunk within the staleness
// threshold, including entries with a cancellation hook: a per-request ctx
// derived from a long-lived parent is never cancelled when its stream aborts.
// Unless force is set, it runs at most once per threshold.
func (p *AnthropicParser) sweepStaleLocked(now time.Time, force bool, stops []func() bool) []func() bool {
	if !force && now.Sub(p.lastSweep) < fwkplugin.DefaultStalenessThreshold {
		return stops
	}
	p.lastSweep = now
	for ctx, entry := range p.streams {
		if now.Sub(entry.updatedAt) > fwkplugin.DefaultStalenessThreshold {
			delete(p.streams, ctx)
			if entry.stop != nil {
				stops = append(stops, entry.stop)
			}
		}
	}
	p.compactLocked()
	return stops
}

func (p *AnthropicParser) evictOldestLocked(stops []func() bool) []func() bool {
	var oldest *streamState
	for _, entry := range p.streams {
		if oldest == nil || entry.updatedAt.Before(oldest.updatedAt) {
			oldest = entry
		}
	}
	for ctx, entry := range p.streams {
		if entry == oldest {
			return p.removeLocked(ctx, entry, stops)
		}
	}
	return stops
}

// compactLocked releases the map once it empties, and reallocates it once it
// shrinks well below its peak, because a Go map never frees its buckets.
func (p *AnthropicParser) compactLocked() {
	if len(p.streams) == 0 {
		p.streams = nil
		p.peakStreams = 0
		return
	}
	if p.peakStreams > compactMinPeak && len(p.streams)*4 < p.peakStreams {
		n := make(map[context.Context]*streamState, len(p.streams))
		maps.Copy(n, p.streams)
		p.streams = n
		p.peakStreams = len(n)
	}
}

func extractUsage(responseBytes []byte) (*request.Usage, error) {
	var responseBody map[string]any
	if err := json.Unmarshal(responseBytes, &responseBody); err != nil {
		return nil, err
	}

	usg, ok := responseBody[reqcommon.FieldUsage].(map[string]any)
	if !ok {
		return nil, nil //nolint:nilnil
	}

	var tokens inputTokens
	tokens.add(usg)
	var usage request.Usage
	tokens.apply(&usage)
	usage.CompletionTokens = jsonInt(usg, reqcommon.FieldOutputTokens)
	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens

	return &usage, nil
}

// inputTokens accumulates the three Messages API input counts across stream
// frames. input_tokens counts only the tokens that were neither read from nor
// written to the prompt cache, so the prompt the server processed is
// input_tokens + cache_read_input_tokens + cache_creation_input_tokens.
//
// Each bucket is updated only by a positive value so a later frame that omits
// or zeroes a bucket keeps the count an earlier frame reported.
type inputTokens struct {
	input         int
	cacheRead     int
	cacheCreation int
	hasRead       bool
}

func (t *inputTokens) add(usg map[string]any) {
	if input := jsonInt(usg, reqcommon.FieldInputTokens); input > 0 {
		t.input = input
	}
	if read, ok := jsonIntPresent(usg, reqcommon.FieldCacheReadInputTokens); ok {
		t.hasRead = true
		if read > 0 {
			t.cacheRead = read
		}
	}
	if creation := jsonInt(usg, reqcommon.FieldCacheCreationInputTokens); creation > 0 {
		t.cacheCreation = creation
	}
}

// merge folds o, accumulated from later frames, onto t with the same rule as add.
func (t *inputTokens) merge(o inputTokens) {
	if o.input > 0 {
		t.input = o.input
	}
	if o.cacheRead > 0 {
		t.cacheRead = o.cacheRead
	}
	if o.cacheCreation > 0 {
		t.cacheCreation = o.cacheCreation
	}
	t.hasRead = t.hasRead || o.hasRead
}

func (t inputTokens) apply(usage *request.Usage) {
	if total := t.input + t.cacheRead + t.cacheCreation; total > 0 {
		usage.PromptTokens = total
	}
	// An explicit zero records a cache miss in PromptTokenDetails. The metrics
	// layer only records request_cached_tokens when PromptTokenDetails is set,
	// so dropping the zero leaves the misses out of the histogram and inflates
	// the reported hit rate.
	if t.hasRead {
		usage.PromptTokenDetails = &request.PromptTokenDetails{CachedTokens: t.cacheRead}
	}
}

func jsonInt(m map[string]any, key string) int {
	v, _ := m[key].(float64)
	return int(v)
}

// jsonIntPresent reports the integer stored under key and whether the key was
// present, so a caller can tell an explicit zero from a missing field.
func jsonIntPresent(m map[string]any, key string) (int, bool) {
	v, ok := m[key].(float64)
	return int(v), ok
}

// chunkSummary is what one stream chunk reports, read without touching the
// per-stream state.
type chunkSummary struct {
	events int
	// sawStart reports a message_start event; tok then holds only the buckets
	// reported from the last message_start on.
	sawStart bool
	// sawDelta reports a message_delta event carrying usage.
	sawDelta bool
	// sawUsage reports any usage object, which makes the chunk report Usage.
	sawUsage   bool
	tok        inputTokens
	completion int
}

func (s chunkSummary) usage(tok inputTokens) *request.Usage {
	if !s.sawUsage {
		return nil
	}
	usage := &request.Usage{CompletionTokens: s.completion}
	tok.apply(usage)
	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	return usage
}

// Anthropic SSE streaming format:
//
//	event: message_start
//	data: {"type":"message_start","message":{"usage":{"input_tokens":25},...}}
//
//	event: message_delta
//	data: {"type":"message_delta","delta":{...},"usage":{"input_tokens":30,"output_tokens":15}}
//
//	event: message_stop
//	data: {"type":"message_stop"}
func scanChunk(chunk []byte) chunkSummary {
	var s chunkSummary
	for line := range bytes.SplitSeq(chunk, []byte("\n")) {
		content, ok := bytes.CutPrefix(line, []byte(reqcommon.SSEDataPrefix))
		if !ok || bytes.Equal(bytes.TrimSuffix(content, []byte("\r")), []byte(reqcommon.SSEDoneMarker)) {
			continue
		}
		s.events++
		// message_start may omit usage but still begins a new stream on ctx.
		if !bytes.Contains(content, []byte(reqcommon.FieldUsage)) && !bytes.Contains(content, []byte(eventMessageStart)) {
			continue
		}

		var event struct {
			Type    string `json:"type"`
			Message struct {
				Usage map[string]any `json:"usage"`
			} `json:"message"`
			Usage map[string]any `json:"usage"`
		}
		if err := json.Unmarshal(content, &event); err != nil {
			continue
		}

		switch event.Type {
		case eventMessageStart:
			s.sawStart = true
			s.tok = inputTokens{}
			if event.Message.Usage != nil {
				s.sawUsage = true
				s.tok.add(event.Message.Usage)
			}
		case eventMessageDelta:
			if event.Usage != nil {
				s.sawDelta = true
				s.sawUsage = true
				s.tok.add(event.Usage)
				s.completion = jsonInt(event.Usage, reqcommon.FieldOutputTokens)
			}
		}
	}
	return s
}
