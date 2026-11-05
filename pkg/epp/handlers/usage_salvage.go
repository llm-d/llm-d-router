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

package handlers

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// maxSalvageObjectBytes caps the usage object collected from a dropped line.
// Every wire format reports a flat token block well below this size.
const maxSalvageObjectBytes = 64 << 10

// salvageDroppedSSELine feeds the bytes of an SSE line dropped for exceeding
// maxSSELineBytes. A dropped line still carries the request's only usage event
// (a response.completed repeats instructions and output before it), so its
// usage object is scanned out and merged instead of being lost with the line.
// lineDone marks the end of the dropped line, flushing the collected usage.
// The scan starts at the head of the dropped line, so string parsing stays in
// phase and a "usage" key echoed inside a string value never matches.
func salvageDroppedSSELine(ctx context.Context, reqCtx *RequestContext, dropped []byte, lineDone bool) {
	if reqCtx.sseSalvage == nil {
		if len(dropped) == 0 {
			return
		}
		reqCtx.sseSalvage = &usageSalvager{}
	}
	reqCtx.sseSalvage.feed(dropped)
	if !lineDone {
		return
	}
	usage := reqCtx.sseSalvage.finish()
	reqCtx.sseSalvage = nil
	if usage != nil {
		mergeUsage(&reqCtx.Usage, *usage)
		log.FromContext(ctx).V(logutil.DEFAULT).Info("salvaged usage from a dropped oversized SSE line",
			"limit", maxSSELineBytes)
	}
}

// scanDroppedSSEUsage returns the usage object carried by a full dropped SSE
// line, the last one winning like the streaming parsers.
func scanDroppedSSEUsage(data []byte) *fwkrh.Usage {
	salvager := &usageSalvager{}
	salvager.feed(data)
	return salvager.finish()
}

// usageSalvager scans the byte stream of one dropped SSE line as it arrives,
// collecting the value of every "usage" key that maps to a JSON object. Only
// state that spans chunks is retained: a partial string being read (as a
// key candidate) and a usage object being collected. A usage value over
// maxSalvageObjectBytes is discarded and skipped to its closing brace.
// Everything else is processed byte by byte, so the memory overhead stays flat.
type usageSalvager struct {
	state       int // one of the states below
	escape      bool
	str         []byte // a string key being read, truncated just past len("usage")
	obj         []byte // the usage object being collected
	objInString bool
	objEscape   bool
	objDepth    int
	last        *fwkrh.Usage
}

const (
	scanOutsideString = iota // outside a JSON string
	scanInString             // inside a string, accumulating it
	scanAfterUsageKey        // a "usage" key just closed, expecting ':'
	scanAfterColon           // the colon was seen, expecting '{' or a non-object value
	scanCollect              // inside the usage object
	scanSkipObject           // usage value exceeded the cap; skip to its closing brace
)

// feed advances the scan by the given bytes of the dropped line.
func (s *usageSalvager) feed(data []byte) {
	for _, c := range data {
		s.step(c)
	}
}

func (s *usageSalvager) step(c byte) {
	switch s.state {
	case scanOutsideString:
		if c == '"' {
			s.str = s.str[:0]
			s.escape = false
			s.state = scanInString
		}
	case scanInString:
		switch {
		case s.escape:
			s.escape = false
		case c == '\\':
			s.escape = true
		case c == '"':
			if string(s.str) == "usage" {
				s.state = scanAfterUsageKey
			} else {
				s.state = scanOutsideString
			}
		default:
			// Longer strings are not the key, but keep one extra byte so an
			// overlong string never equals the candidate.
			if len(s.str) <= len("usage") {
				s.str = append(s.str, c)
			}
		}
	case scanAfterUsageKey:
		if c == ':' {
			s.state = scanAfterColon
		} else if c != ' ' && c != '\t' {
			s.state = scanOutsideString
		}
	case scanAfterColon:
		switch c {
		case '{':
			s.obj = s.obj[:0]
			s.obj = append(s.obj, c)
			s.objInString, s.objEscape, s.objDepth = false, false, 1
			s.state = scanCollect
		case '"':
			// A string-valued "usage" key is not the token block. Enter the
			// string so its closing quote stays in phase; otherwise that quote
			// is read as the start of the next string and the real usage object
			// later in the line is missed.
			s.str = s.str[:0]
			s.escape = false
			s.state = scanInString
		case ' ', '\t', '\n', '\r':
			// keep waiting for the object
		default:
			s.state = scanOutsideString
		}
	case scanCollect:
		s.obj = append(s.obj, c)
		s.trackUsageValue(c)
		if s.objDepth == 0 {
			if usage := parseSalvagedUsage(s.obj); usage != nil {
				s.last = usage
			}
			s.state = scanOutsideString
		} else if len(s.obj) > maxSalvageObjectBytes {
			// Not a token block. Drop the buffer and skip the rest of this
			// value with string and depth tracking still in phase, so a later
			// usage object on the same line can still match.
			s.obj = nil
			s.state = scanSkipObject
		}
	case scanSkipObject:
		s.trackUsageValue(c)
		if s.objDepth == 0 {
			s.objInString = false
			s.objEscape = false
			s.state = scanOutsideString
		}
	}
}

// trackUsageValue updates string and brace depth for one byte inside a usage
// value. The caller owns when the value ends.
func (s *usageSalvager) trackUsageValue(c byte) {
	switch {
	case s.objEscape:
		s.objEscape = false
	case s.objInString && c == '\\':
		s.objEscape = true
	case c == '"':
		s.objInString = !s.objInString
	case !s.objInString && c == '{':
		s.objDepth++
	case !s.objInString && c == '}':
		s.objDepth--
	}
}

// finish closes the dropped line and returns the last usage object found.
func (s *usageSalvager) finish() *fwkrh.Usage {
	// A usage object cut short by the end of the line cannot be parsed.
	return s.last
}

// parseSalvagedUsage decodes a usage object covering the OpenAI chat
// (prompt_tokens/completion_tokens), Responses and Anthropic
// (input_tokens/output_tokens) field namings.
func parseSalvagedUsage(obj []byte) *fwkrh.Usage {
	var fields map[string]any
	if err := json.Unmarshal(obj, &fields); err != nil {
		return nil
	}
	usage := &fwkrh.Usage{}
	for _, key := range []string{"prompt_tokens", "input_tokens"} {
		if f, ok := fields[key].(float64); ok {
			usage.PromptTokens = int(f)
			break
		}
	}
	// Anthropic reports the cache counts inside the prompt total.
	for _, key := range []string{"cache_creation_input_tokens", "cache_read_input_tokens"} {
		if f, ok := fields[key].(float64); ok {
			usage.PromptTokens += int(f)
		}
	}
	for _, key := range []string{"completion_tokens", "output_tokens"} {
		if f, ok := fields[key].(float64); ok {
			usage.CompletionTokens = int(f)
			break
		}
	}
	if usage.PromptTokens == 0 && usage.CompletionTokens == 0 {
		// An object with none of the token fields is not a usage block.
		return nil
	}
	if f, ok := fields["total_tokens"].(float64); ok {
		usage.TotalTokens = int(f)
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	var details map[string]any
	for _, key := range []string{"prompt_tokens_details", "input_tokens_details"} {
		if d, ok := fields[key].(map[string]any); ok {
			details = d
			break
		}
	}
	if f, ok := details["cached_tokens"].(float64); ok {
		usage.PromptTokenDetails = &fwkrh.PromptTokenDetails{CachedTokens: int(f)}
	}
	// Anthropic reports the cache read at the root of the usage object.
	// CachedTokens counts that read inside PromptTokens, matching applyInputTokens.
	if usage.PromptTokenDetails == nil {
		if f, ok := fields["cache_read_input_tokens"].(float64); ok && int(f) != 0 {
			usage.PromptTokenDetails = &fwkrh.PromptTokenDetails{CachedTokens: int(f)}
		}
	}
	return usage
}
